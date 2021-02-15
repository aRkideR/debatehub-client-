
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
    - use a value netw