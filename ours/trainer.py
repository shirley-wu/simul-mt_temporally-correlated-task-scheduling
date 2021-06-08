import math
import os

import torch
import tqdm
from agent import REINFORCE
from env import SimMTEnvironment
from utils import log_str, verify_dir, format_seconds


class Trainer:
    def __init__(self, args):
        # log directly to file. Not to stdout
        self.logfile = os.path.join(args.rl_search_save_dir, "log")
        print("Log to %s" % self.logfile)
        self.log_buffer = []
        # assert args.rl_search_save_interval % args.rl_search_reward_interval == 0
        assert args.rl_search_learn_interval % args.rl_search_reward_interval == 0
        assert args.rl_search_learn_interval % args.rl_search_save_interval == 0
        # build env, agent
        self.env = SimMTEnvironment.build_env(args)
        self.agent = REINFORCE(self.env, args).to(args.rl_search_device)
        self.args = args
        self.step = 0
        self.best_student_metric = math.inf
        self.env.reset()
        self.agent.reset()
        # load checkpoint
        ckpt_last = os.path.join(args.rl_search_save_dir, "checkpoint_last.txt")
        if os.path.exists(ckpt_last):
            self.step = self.read_ckpt_info(ckpt_last)
            self.load(self.step)

    def print(self, *args):
        self.log_buffer.append(args)

    def clear_print_buffer(self):
        # write to logfile
        with open(self.logfile, "a") as f:
            for args in self.log_buffer:
                print(*args, file=f)
        self.log_buffer = []

    def checkpoint_name(self, step):
        return os.path.join(self.args.rl_search_save_dir, "teacher", "checkpoint{:d}.pt".format(step)), \
               os.path.join(self.args.rl_search_save_dir, "student", "checkpoint{:d}.pt".format(step)), \
               os.path.join(self.args.rl_search_save_dir, "trainer", "checkpoint{:d}.pt".format(step))

    @staticmethod
    def write_ckpt_info(path, step):
        d = os.path.dirname(path)
        if not os.path.exists(d):
            os.makedirs(d)
        with open(path, "w") as f:
            f.write("{:d}\n".format(step))

    @staticmethod
    def read_ckpt_info(path):
        with open(path) as f:
            s = f.read().strip()
        return int(s)

    def save(self, student_metric):
        self.clear_print_buffer()
        # pdb.set_trace()
        teacher, student, trainer = self.checkpoint_name(self.step)
        self.agent.save(teacher)
        self.env.save(student)
        verify_dir(trainer)
        torch.save(dict(best_student_metric=self.best_student_metric), trainer)
        self.write_ckpt_info(os.path.join(self.args.rl_search_save_dir, "checkpoint_last.txt"), self.step)
        if student_metric < self.best_student_metric:
            print("= New best student! step %d, %r -> %r" % (self.step, self.best_student_metric, student_metric))
            self.best_student_metric = student_metric
            self.write_ckpt_info(os.path.join(self.args.rl_search_save_dir, "best.txt"), self.step)

    def load(self, step):
        # pdb.set_trace()
        print("| Load agent and model at %d step" % step)
        teacher, student, trainer = self.checkpoint_name(step)
        # pdb.set_trace()
        self.agent.load(teacher)
        self.env.load(student)
        ckpt = torch.load(trainer)
        self.best_student_metric = ckpt["best_student_metric"]

    def train(self):
        env = self.env
        agent = self.agent
        args = self.args

        # pdb.set_trace()
        t = tqdm.tqdm()
        while not env.done():
            self.step += 1
            state = env.state()
            action = agent.get_action(state)
            _, log = env.step(action)
            self.print("> Environment step logging:\tstep %d\taction: %r\t%s" % (
                self.step, env.wrap_action(action), log_str(log)))
            # pdb.set_trace()
            if self.step % args.rl_search_reward_interval == 0:
                reward, student_metric, log = env.reward()
                self.print("> Environment reward logging:\tstep %d\treward: %.4f, student metric: %.4f\t%s" % (
                    self.step, reward, student_metric, log_str(log)))
                agent.update_reward(reward)
                if self.step % args.rl_search_learn_interval == 0:
                    agent.learn()
                    agent.reset()
                    assert self.step % args.rl_search_save_interval == 0
                    self.save(student_metric)
            elif self.step % args.rl_search_save_interval == 0:
                student_metric, log = env.validate()
                self.print("> Environment validate logging:\tstep %d\tstudent metric: %.4f\t%s" % (
                    self.step, student_metric, log_str(log)))
                self.save(student_metric)
            t.update(1)
        self.clear_print_buffer()
        print("Avg time: %s" % format_seconds(t.avg_time))
        print("Total: %d steps" % t.n)
        print("Total time: %s" % format_seconds(t.avg_time * t.n))
        t.close()
