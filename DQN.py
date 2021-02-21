
import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np

from common.Agent import Agent
from common.Model import ActorNetwork
from common.utils import identity, to_tensor_var


class DQN(Agent):
    """
    An agent learned with DQN using replay memory and temporal difference
    - use a value network to estimate the state-action value
    """
    def __init__(self, env, state_dim, action_dim,
                 memory_ca