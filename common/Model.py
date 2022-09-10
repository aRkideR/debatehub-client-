
import torch as th
from torch import nn


class ActorNetwork(nn.Module):
    """
    A network for actor
    """
    def __init__(self, state_dim, hidde