
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
    - exploration_action: 