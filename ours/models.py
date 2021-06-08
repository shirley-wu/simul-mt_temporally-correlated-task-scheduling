from argparse import ArgumentParser

from torch import nn


class FFModel(nn.Module):
    def __init__(self, input_size, output_size, args):
        super().__init__()
        # pdb.set_trace()
        ff_hidden_size = args.ff_hidden_size
        self.layers = nn.Sequential(nn.Linear(input_size, ff_hidden_size), nn.Tanh(),
                                    nn.Linear(ff_hidden_size, output_size), )

    def forward(self, x):
        # pdb.set_trace()
        return self.layers(x)

    @staticmethod
    def add_args(parser: ArgumentParser):
        # pdb.set_trace()
        parser.add_argument("--ff-hidden-size", default=512, type=int)
