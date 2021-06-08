from typing import List

import numpy as np
import torch
import torch.optim as optim
from models import FFModel
from torch.distributions import Categorical
from utils import verify_dir


class REINFORCE:
    def __init__(self, environment, args):
        # set arguments
        self.args = args
        self.device = "cpu"
        # reset
        self.reset()
        # build policy
        self.policy = FFModel(environment.state_size, environment.action_size, self.args)
        print("Policy network")
        print(self.policy)
        self.optimizer = optim.SGD(self.policy.parameters(), lr=self.args.rl_search_lr)

    def reset(self):
        self.episode_rewards = []
        self.episode_log_probabilities = []

    def get_action(self, state) -> List[int]:
        action, log_probabilities = self.pick_action_and_get_log_probabilities(state)
        self.episode_log_probabilities.append(log_probabilities)
        return action

    def get_probabilities(self, state):
        action_probabilities = self.policy(state.float().to(self.device)).softmax(dim=-1)
        return action_probabilities

    def pick_action_and_get_log_probabilities(self, state):
        """Picks actions and then calculates the log probabilities of the actions it picked given the policy"""
        # PyTorch only accepts mini-batches and not individual observations so we have to add
        # a "fake" dimension to our observation using unsqueeze
        # action_probabilities = action_probabilities.cpu()
        action_probabilities = self.get_probabilities(state)
        action_distribution = Categorical(action_probabilities)  # this creates a distribution to sample from
        action = action_distribution.sample()
        return [int(a) for a in action.reshape(-1)], action_distribution.log_prob(action)

    def update_reward(self, reward: float) -> None:
        self.episode_rewards.append(reward)

    def learn(self):
        # calculate reward
        # NOTE: we update the network every time we get a reward, so len(self.episode_rewards) == 1
        discount_rate = 1.0
        discounts = discount_rate ** np.arange(len(self.episode_rewards))
        total_discounted_reward = np.dot(discounts, self.episode_rewards)
        # calculate loss
        policy_loss = - torch.cat(self.episode_log_probabilities).sum() * total_discounted_reward
        # optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def save(self, path: str) -> None:
        verify_dir(path)
        torch.save(self.state_dict(), path)
        print("| save agent to %s" % path)

    def state_dict(self) -> dict:
        return dict(policy=self.policy.state_dict(), optimizer=self.optimizer.state_dict())

    def load(self, path: str):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)

    def load_state_dict(self, state_dict: dict):
        self.policy.load_state_dict(state_dict["policy"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def to(self, device):
        self.device = device
        self.policy = self.policy.to(device)
        return self
