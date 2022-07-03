
import torch as th

import numpy as np

from common.Memory import ReplayMemory
from common.utils import identity


class Agent(object):
    """
    A unified agent interface:
    - interact: interact with the environment to collect experience
        - _take_one_step: take one step
        - _take_n_steps: take n steps
        - _discount_reward: discount roll out rewards
    - train: train on a sample batch
        - _soft_update_target: soft update the target network
    - exploration_action: choose an action based on state with random noise
                            added for exploration in training
    - action: choose an action based on state for execution
    - value: evaluate value for a state-action pair
    - evaluation: evaluation a learned agent
    """
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=10000,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=identity, critic_loss="mse",
                 actor_lr=0.01, critic_lr=0.01,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True):

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env_state = self.env.reset()
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.roll_out_n_steps = 1

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.done_penalty = done_penalty

        self.memory = ReplayMemory(memory_capaci