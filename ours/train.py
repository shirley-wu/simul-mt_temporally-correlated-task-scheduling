from env import SimMTEnvironment
from fairseq import options
from models import FFModel
from trainer import Trainer


def parse_args():
    parser = options.get_training_parser()
    SimMTEnvironment.add_args(parser)
    FFModel.add_args(parser)
    parser.add_argument("--rl-search-device", default="cuda", choices=["cuda", "cpu", ])
    parser.add_argument("--rl-search-reward-interval", default=750, type=int)
    parser.add_argument("--rl-search-learn-interval", default=750, type=int)
    parser.add_argument("--rl-search-save-interval", default=750, type=int)  # ~ an epoch
    parser.add_argument("--rl-search-save-dir", required=True)  # NOTE: different from fairseq save dir
    parser.add_argument("--rl-search-lr", default=5e-04, type=float)
    args = options.parse_args_and_arch(parser)
    # args.distributed_world_size = 1
    return args


def main(args):
    print(args)
    trainer = Trainer(args)
    try:
        trainer.train()
    finally:
        trainer.env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
