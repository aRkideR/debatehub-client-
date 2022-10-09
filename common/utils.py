
import torch as th
from torch.autograd import Variable
import numpy as np


def identity(x):
    return x


def entropy(p):
    return -t