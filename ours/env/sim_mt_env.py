"""
Adopted from fairseq v0.8.0 train.py
"""
import collections
import math
import os
import random
import sys
import traceback
from typing import Tuple, List, Union

import numpy as np
import torch
from fairseq import checkpoint_utils, tasks, utils, progress_bar, distributed_utils
from fairseq.data import iterators
from fairseq.meters import AverageMeter
from fairseq.trainer import Trainer
from utils import verify_dir

from .fairseq_utils import get_training_stats, get_valid_stats


class SimMTEnvironment:
    DEFAULT_STATE_REPR = ["train-stage", "src-length", "tgt-length",
                          "train-loss-target-k", "historical-train-loss-target-k",
                          "last-epoch-valid-loss", "historical-last-epoch-valid-loss", ]
    ALL_STATE_REPR = ["train-stage", "src-length", "tgt-length",
                      "train-loss-target-k", "historical-train-loss-target-k",
                      "last-epoch-valid-loss", "historical-last-epoch-valid-loss", ]

    @staticmethod
    def add_args(parser):
        parser.add_argument("--sim-mt-target-k", type=int, required=True)
        parser.add_argument("--sim-mt-sample-k-max", default=13, type=int)
        parser.add_argument("--sim-mt-sample-k-min", default=1, type=int)
        parser.add_argument("--rl-search-state-repr", nargs="+", default=SimMTEnvironment.DEFAULT_STATE_REPR,
                            choices=SimMTEnvironment.ALL_STATE_REPR)
        args, unknown = parser.parse_known_args()

    def wrap_action(self, action: Union[int, List[int]]) -> Union[int, List[int]]:
        # action: in { 0 ... sim_mt_sample_k_max - sim_mt_sample_k_min }
        # return: action + self.args.sim_mt_sample_k_min, in { sim_mt_sample_k_min ... sim_mt_sample_k_max }
        if isinstance(action, int):
            return action + self.args.sim_mt_sample_k_min
        else:
            return [self.wrap_action(int(a)) for a in action]

    @staticmethod
    def build_env(args):
        if args.distributed_world_size == 1:
            return SimMTEnvironmentSingleGPU(args)
        else:
            return SimMTEnvironmentMultiGPU(args)

    @property
    def state_size(self) -> int:
        state_repr_lens = {
            "train-stage": 1,
            "src-length": 1,
            "tgt-length": 1,
            "last-epoch-valid-loss": 1,
            "historical-last-epoch-valid-loss": 1,
            "train-loss-target-k": 1,
            "historical-train-loss-target-k": 1,
        }
        return sum([state_repr_lens[k] for k in set(self.args.rl_search_state_repr)])

    @property
    def action_size(self) -> int:
        return self.args.sim_mt_sample_k_max - self.args.sim_mt_sample_k_min + 1


def get_static_k(k):
    return (lambda: k)


class ListKGetter:
    def __init__(self, klist):
        self.klist = klist
        self.i = 0

    def get_k(self):
        k = int(self.klist[self.i])
        self.i += 1
        return k


class SimMTEnvironmentSingleGPU(SimMTEnvironment):
    def __init__(self, args):
        self.args = args
        args.wait_k = args.wait_k if args.wait_k is not None else args.sim_mt_target_k

        # necessary setup and check
        utils.import_user_module(args)
        assert args.max_tokens is not None or args.max_sentences is not None, \
            'Must specify batch size either with --max-tokens or --max-sentences'

        # Initialize CUDA and distributed training
        if torch.cuda.is_available() and not args.cpu:
            torch.cuda.set_device(args.device_id)
        torch.manual_seed(args.seed)
        if args.distributed_world_size > 1:
            args.distributed_rank = distributed_utils.distributed_init(args)

        self.max_epoch = args.max_epoch or math.inf
        self.max_update = args.max_update or math.inf

        self.task = None
        self.model = None
        self.trainer = None
        self.epoch_itr = None
        self.itr = None
        self.extra_meters = None
        self._done = True
        # recorded for state repr
        self.data = None
        self.last_loss = None
        self._state = None
        self.historical_train_loss = [[] for _ in range(args.sim_mt_sample_k_min, args.sim_mt_sample_k_max + 1)]
        self.historical_valid_loss = []
        self.historical_last_epoch_valid_loss = []
        self.k_coverage_num = [0 for _ in range(self.action_size)]
        self.k_coverage_num_current_epoch = [0 for _ in range(self.action_size)]

    def save(self, path: str):
        verify_dir(path)
        extra_state = dict(
            train_iterator=self.epoch_itr.state_dict(),
            data=self.data,
            last_loss=self.last_loss,
            historical_train_loss=self.historical_train_loss,
            historical_valid_loss=self.historical_valid_loss,
            state_repr=self._state,
            historical_last_epoch_valid_loss=self.historical_last_epoch_valid_loss,
            k_coverage_num=self.k_coverage_num,
            k_coverage_num_current_epoch=self.k_coverage_num_current_epoch,
        )
        self.trainer.save_checkpoint(path, extra_state)

    def save_state(self, path: str):
        verify_dir(path)
        torch.save(dict(
            data=self.data,
            last_loss=self.last_loss,
            historical_train_loss=self.historical_train_loss,
            historical_valid_loss=self.historical_valid_loss,
            state_repr=self._state,
            historical_last_epoch_valid_loss=self.historical_last_epoch_valid_loss,
            k_coverage_num=self.k_coverage_num,
            k_coverage_num_current_epoch=self.k_coverage_num_current_epoch,
        ), path)

    def load(self, path: str, state_path=None):
        """Load a checkpoint and restore the training iterator."""
        extra_state = self.trainer.load_checkpoint(path)
        if state_path is not None:
            extra_state.update(torch.load(state_path))

        self.data = extra_state["data"]
        self.last_loss = extra_state["last_loss"]
        self.historical_train_loss = extra_state.get("historical_train_loss", [])
        self.historical_valid_loss = extra_state.get("historical_valid_loss", [])
        self.historical_last_epoch_valid_loss = extra_state.get("historical_last_epoch_valid_loss", [])
        self.k_coverage_num = extra_state.get("k_coverage_num", [0 for _ in range(self.action_size)])
        self.k_coverage_num_current_epoch = extra_state.get("k_coverage_num_current_epoch",
                                                            [0 for _ in range(self.action_size)])
        self._state = extra_state.get("state_repr", None)

        itr_state = extra_state['train_iterator']
        self.epoch_itr = self.trainer.get_train_iterator(epoch=itr_state['epoch'])
        self.epoch_itr.load_state_dict(itr_state)
        status = self.start_epoch()
        assert status, "Load fail by starting epoch"
        self.trainer.lr_step(self.epoch_itr.epoch)

        if self.data is None:
            assert self._state is None
            self.step_data()
            self.step_state()
        elif self._state is None:
            self.step_state()

    def reset(self):
        args = self.args
        print("Setup training process in SimMT Environment")

        # Setup task, e.g., translation, language modeling, etc.
        task = tasks.setup_task(args)

        # Load valid dataset (we load training data below, based on the latest checkpoint)
        for valid_sub_split in self.args.valid_subset.split(','):
            task.load_dataset(valid_sub_split, combine=False, epoch=0)

        # Build model and criterion
        model = task.build_model(args)
        criterion = task.build_criterion(args)

        print("Fairseq model:")
        print(model)
        print('| model {}, criterion {}'.format(args.arch, criterion.__class__.__name__))
        print('| num. model params: {} (num. trained: {})'.format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        ))

        # Build trainer
        trainer = Trainer(args, task, model, criterion)
        print('| training on {} GPUs'.format(args.distributed_world_size))
        print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
            args.max_tokens,
            args.max_sentences,
        ))

        # Load the latest checkpoint if one is available and restore the
        # corresponding train iterator
        if args.restore_file == "checkpoint_last.pt" or args.restore_file == "/dummy":
            args.restore_file = "/dummy"
            assert not os.path.exists(args.restore_file)  # don't let fairseq to load model
        else:
            print("=" * 50)
            print("!" * 50)
            print("Load pretrained model from %s" % args.restore_file)
            assert os.path.exists(args.restore_file)
        _, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

        self.task = task
        self.model = model
        self.trainer = trainer
        self.epoch_itr = epoch_itr
        self._done = False
        self.data = None
        self.historical_train_loss = [[] for _ in range(args.sim_mt_sample_k_min, args.sim_mt_sample_k_max + 1)]
        self.historical_valid_loss = []
        self.historical_last_epoch_valid_loss = []
        self.k_coverage_num = [0 for _ in range(self.action_size)]
        self.k_coverage_num_current_epoch = [0 for _ in range(self.action_size)]

        valid_loss, _ = self.validate()
        self.last_loss = valid_loss
        self.historical_last_epoch_valid_loss = [valid_loss, ]
        self.step_data()
        self.step_state()

    def step(self, action: List[int]) -> Tuple[bool, dict]:
        for a in action:
            self.k_coverage_num[a] += 1
            self.k_coverage_num_current_epoch[a] += 1
        if self.data is None:
            return False, {}
        i, samples = self.data
        logging = self.step_train(i, samples, [self.wrap_action(a) for a in action])
        self.step_data()
        self.step_state()
        return True, logging

    def step_train(self, i: int, samples, action: List[int]):
        self.task.get_wait_k = ListKGetter(action).get_k
        # print(action)
        log_output = self.trainer.train_step(samples)

        # log mid-epoch stats
        stats = get_training_stats(self.trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k or k == 'accuracy':
                self.extra_meters[k].update(v, log_output['sample_size'])
            else:
                self.extra_meters[k].update(v)
            stats[k] = self.extra_meters[k].avg
        logging_stats = collections.OrderedDict({"epoch": self.epoch_itr.epoch, "step": i})
        logging_stats.update({k: progress_bar.format_stat(v) for k, v in stats.items()})

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            self.trainer.get_meter('wps').reset()

        num_updates = self.trainer.get_num_updates()
        if num_updates >= self.max_update:  # training finished
            self._done = True

        return logging_stats

    def step_data(self):
        self.data = None
        if self._done:  # env done. return nothing
            return
        elif self.itr is None:
            status = self.start_epoch()
            if not status:
                return
        try:
            i, samples = next(self.itr)  # next at current epoch
        except StopIteration:  # current epoch is finished
            self.finalize_epoch()  # validate, save, step lr
            status = self.start_epoch()  # get next epoch
            if not status:
                return
            i, samples = next(self.itr)
        self.data = (i, samples)

    def step_state(self):
        if self.data is None:
            self._state = None
        else:
            self._state = self.calculate_state(self.data)
            # print(self._state)

    def valid_samples_loss(self, action, samples):
        # self.task.action_k = self.wrap_action(action)
        self.task.get_wait_k = get_static_k(self.wrap_action(action))
        # log_output = self.trainer.train_step(samples)
        self.model.eval()
        loss = 0
        sample_size = 0
        with torch.no_grad():  # copy codes from train_step, except that we don't accumulate grad, don't optimize
            for s in samples:
                loss_, sample_size_, _ = self.trainer.criterion(self.model, self.trainer._prepare_sample(s))
                loss += loss_.item()
                sample_size += sample_size_
        return loss / sample_size

    def calculate_state(self, data):
        # Calculate self._state
        i, samples = data
        n_samples = len([s for s in samples if len(s) > 0])

        if n_samples == 0 or sum([len(s) for s in samples]) == 0:
            # empty data
            return torch.zeros((0, self.state_size), dtype=torch.float)

        state_tensor = []
        if "train-stage" in self.args.rl_search_state_repr:
            state_tensor.append([self.trainer.get_num_updates() / self.args.max_update, ] * n_samples)
        if "src-length" in self.args.rl_search_state_repr:
            src_lengths = [s['net_input']['src_lengths'] for s in samples if len(s) > 0]
            src_means = [s.float().mean() for s in src_lengths]
            src_dataset_mean = self.task.dataset(self.args.train_subset).src_sizes.mean()
            state_tensor.append([s / src_dataset_mean for s in src_means])
        if "tgt-length" in self.args.rl_search_state_repr:
            tgt_lengths = [(s['target'] != self.task.target_dictionary.pad()).sum(1) for s in samples if len(s) > 0]
            tgt_means = [t.float().mean() for t in tgt_lengths]
            tgt_dataset_mean = self.task.dataset(self.args.train_subset).tgt_sizes.mean()
            state_tensor.append([t / tgt_dataset_mean for t in tgt_means])
        if "train-loss-target-k" in self.args.rl_search_state_repr or "historical-train-loss-target-k" in self.args.rl_search_state_repr:
            action_k_index = self.args.sim_mt_target_k - self.args.sim_mt_sample_k_min
            losses = [self.valid_samples_loss(action_k_index, [s, ]) for s in samples if len(s) > 0]
            self.historical_train_loss[action_k_index] += losses
            if "train-loss-target-k" in self.args.rl_search_state_repr:
                state_tensor.append(losses)
            if "historical-train-loss-target-k" in self.args.rl_search_state_repr:
                state_tensor.append([np.mean(self.historical_train_loss[action_k_index]), ] * n_samples)
        if "last-epoch-valid-loss" in self.args.rl_search_state_repr:
            state_tensor.append([self.historical_last_epoch_valid_loss[-1], ] * n_samples)
        if "historical-last-epoch-valid-loss" in self.args.rl_search_state_repr:
            state_tensor.append([np.mean(self.historical_last_epoch_valid_loss), ] * n_samples)
        return torch.FloatTensor(state_tensor).transpose(0, 1)

    def finalize_epoch(self):
        # only use first validation loss to update the learning rate
        valid_loss, _ = self.validate()

        self.historical_last_epoch_valid_loss.append(valid_loss)
        print("Epoch %d is finished; loss: %.4f" % (self.epoch_itr.epoch, valid_loss))
        self.trainer.lr_step(self.epoch_itr.epoch, valid_loss)
        if ':' in getattr(self.args, 'data', ''):
            # sharded data: get train iterator for next epoch
            self.epoch_itr = self.trainer.get_train_iterator(self.epoch_itr.epoch)
        self.k_coverage_num_current_epoch = [0 for _ in range(self.action_size)]

    def start_epoch(self) -> bool:
        if not (self.trainer.get_lr() > self.args.min_lr and self.epoch_itr.epoch < self.max_epoch and
                self.trainer.get_num_updates() < self.max_update):
            self._done = True
            return False

        args = self.args
        update_freq = args.update_freq[self.epoch_itr.epoch - 1] \
            if self.epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]
        # Initialize data iterator
        itr = self.epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=args.fix_batches_to_gpus,
            shuffle=(self.epoch_itr.epoch >= args.curriculum),
        )
        itr = iterators.GroupedIterator(itr, update_freq)
        # meters in the epoch
        self.extra_meters = collections.defaultdict(lambda: AverageMeter())
        # enumerate
        self.itr = enumerate(itr, start=self.epoch_itr.iterations_in_epoch)
        return True

    def reward(self) -> Tuple[float, float, dict]:
        curr_loss, logging = self.validate()
        last_loss = self.last_loss
        self.last_loss = curr_loss
        r = last_loss - curr_loss
        return r, curr_loss, logging

    def done(self) -> bool:
        return self._done

    def validate(self) -> Tuple[float, dict]:
        args = self.args
        self.task.get_wait_k = get_static_k(self.args.sim_mt_target_k)

        subsets = self.args.valid_subset.split(',')
        valid_losses = []
        log_output = collections.OrderedDict()

        for i, subset in enumerate(subsets):
            # Initialize data iterator
            itr = self.task.get_batch_iterator(
                dataset=self.task.dataset(subset),
                max_tokens=args.max_tokens_valid,
                max_sentences=args.max_sentences_valid,
                max_positions=utils.resolve_max_positions(
                    self.task.max_positions(),
                    self.trainer.get_model().max_positions(),
                ),
                ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
                required_batch_size_multiple=args.required_batch_size_multiple,
                seed=args.seed,
                num_shards=args.distributed_world_size,
                shard_id=args.distributed_rank,
                num_workers=args.num_workers,
            ).next_epoch_itr(shuffle=False)

            # reset validation loss meters
            for k in ['valid_loss', 'valid_nll_loss']:
                meter = self.trainer.get_meter(k)
                if meter is not None:
                    meter.reset()
            extra_meters = collections.defaultdict(lambda: AverageMeter())

            for sample in itr:
                if self.trainer._dummy_batch is None:  # fairseq v0.8.0 is so stupid
                    self.trainer._dummy_batch = sample
                log_output = self.trainer.valid_step(sample)

                for k, v in log_output.items():
                    if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                        continue
                    extra_meters[k].update(v)

            # log validation stats
            stats = get_valid_stats(self.trainer, args, extra_meters)
            for k, meter in extra_meters.items():
                stats[k] = meter.avg

            valid_losses.append(
                stats[args.best_checkpoint_metric].avg
                if args.best_checkpoint_metric == 'loss'
                else stats[args.best_checkpoint_metric]
            )
            for k, v in stats.items():
                log_output[k if len(subsets) == 1 else "valid_{:d}_{:s}".format(i, k)] = progress_bar.format_stat(v)

        # save checkpoint
        # checkpoint_utils.save_checkpoint(args, self.trainer, self.epoch_itr, valid_losses[0])
        return valid_losses[0], log_output

    def state(self) -> torch.Tensor:
        return self._state

    def close(self):
        pass


EXCEPTION_STR = "EXCEPTION"


def distributed_fn(device_id, args, conn_list):
    conn = conn_list[device_id]
    try:
        args.device_id = device_id
        args.distributed_rank = device_id
        env = SimMTEnvironmentSingleGPU(args)
        conn.send(True)
        while True:
            signal = conn.recv()
            if signal[0] == "reset":
                env.reset()
                conn.send(True)
            elif signal[0] == "done":
                conn.send(env.done())
            elif signal[0] == "load":
                env.load(signal[1], state_path=signal[2])
                conn.send(True)
            elif signal[0] == "step":
                conn.send(env.step(signal[1]))
            elif signal[0] == "save":
                env.save(signal[1])
                conn.send(True)
            elif signal[0] == "save_state":
                env.save_state(signal[1])
                conn.send(True)
            elif signal[0] == "state":
                conn.send(env.state())
            elif signal[0] == "validate":
                result = env.validate()
                if device_id == 0:  # is master
                    conn.send(result)
            elif signal[0] == "reward":
                result = env.reward()
                if device_id == 0:  # is master
                    conn.send(result)
    except Exception as e:
        conn.send((EXCEPTION_STR, e, traceback.format_exc()))
    conn.send((EXCEPTION_STR, "Exiting distributed_fn", ""))


class ConnWrapper:
    def __init__(self, conn):
        self.conn = conn

    def send(self, *args, **kwargs):
        self.conn.send(*args, **kwargs)

    def _recv(self, timeout, info):
        if self.conn.poll(timeout):
            return self.conn.recv()
        print("Connection timeout %.2f. Info: %s" % (timeout, info))
        sys.exit(1)

    def recv(self, timeout=180, info="-"):
        ret = self._recv(timeout, info)
        if isinstance(ret, tuple) and len(ret) == 3 and ret[0] == EXCEPTION_STR:
            print("Error in subprocess!\n\n")
            print(ret[2])
            sys.exit(1)
        return ret


class SimMTEnvironmentMultiGPU(SimMTEnvironment):

    def __init__(self, args):
        self.args = args
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        # init distributed
        self.conns = []
        conns_other = []
        for _ in range(args.distributed_world_size):
            conn1, conn2 = torch.multiprocessing.Pipe()
            self.conns.append(ConnWrapper(conn1))
            conns_other.append(conn2)
        context = torch.multiprocessing.spawn(fn=distributed_fn, args=(args, conns_other),
                                              nprocs=args.distributed_world_size, join=False)
        self.ps = context.processes
        for conn in self.conns:
            assert conn.recv(360, "Init")
        print("| Distributed initialization done")
        self.historical_train_loss = []
        self._state = None

    def reset(self) -> None:
        for conn in self.conns:
            conn.send(("reset",))
        for i, conn in enumerate(self.conns):
            assert conn.recv(600, "Reset")
            print("======== Reset process %d" % i)
        self.historical_train_loss = []
        self.step_state()

    def state(self) -> torch.Tensor:
        return self._state

    def step_state(self):
        states = []
        for conn in self.conns:
            conn.send(("state",))
            states.append(conn.recv(300, "Step state"))
        # merge state
        # print(states)
        if any([x is None for x in states]):  # step done
            assert all([x is None for x in states])
            self._state = None
            return

        state_tensor = torch.cat(states, dim=0)

        # fix historical train loss target k
        if "historical-train-loss-target-k" in self.args.rl_search_state_repr:
            # gather train-loss-target-k from state repr
            assert "train-loss-target-k" in self.args.rl_search_state_repr
            # train-loss index
            train_loss_idx = int("train-stage" in self.args.rl_search_state_repr) + \
                             int("src-length" in self.args.rl_search_state_repr) + \
                             int("tgt-length" in self.args.rl_search_state_repr)
            train_loss_hist_idx = train_loss_idx + 1
            self.historical_train_loss += [float(x) for x in state_tensor[:, train_loss_idx]]
            state_tensor[:, train_loss_hist_idx] = np.mean(self.historical_train_loss)
        self._state = state_tensor

    def step(self, action) -> Tuple[bool, dict]:
        for conn in self.conns:
            conn.send(("step", action))
        status = True
        logging = {}
        for i, conn in enumerate(self.conns):
            status_, logging_ = conn.recv(600, "Step")
            status = status and status_
            logging["process{:d}".format(i)] = logging_
        self.step_state()
        return status, logging

    def reward(self) -> Tuple[float, float, dict]:
        # Return: environment reward, student model metric, reward logging
        for conn in self.conns:
            conn.send(("reward",))
        rw, m, logging = self.conns[0].recv(600, "Reward")
        return rw, m, logging

    def validate(self) -> Tuple[float, dict]:
        # Validate student model
        # Return: student model metric, validate logging
        for conn in self.conns:
            conn.send(("validate",))
        m, logging = self.conns[0].recv(360, "Validate")
        return m, logging

    def done(self) -> bool:
        self.conns[0].send(("done",))
        done_ = self.conns[0].recv(180, "Done")
        for conn in self.conns[1:]:
            conn.send(("done",))
            assert done_ == conn.recv(180, "Done")
        return done_

    def save(self, path: str):
        self.conns[0].send(("save", path))
        assert self.conns[0].recv(2400, "Save main ckpt")
        for i, conn in enumerate(self.conns[1:]):
            conn.send(("save_state", path + ".state{:d}".format(i + 1)))
            assert conn.recv(600, "Save state {:d}".format(i))
        verify_dir(path)
        torch.save(self.historical_train_loss, path + ".state-dist-controller")

    def load(self, path: str):
        for i, conn in enumerate(self.conns):
            conn.send(("load", path, (path + ".state{:d}".format(i)) if i > 0 else None))
        for i, conn in enumerate(self.conns):
            assert conn.recv(600, "Load {:d}".format(i))
            print("======== Process %d load done" % i)
        self.historical_train_loss = torch.load(path + ".state-dist-controller")
        self.step_state()

    def close(self):
        for p in self.ps:
            p.terminate()
        self.ps = None
