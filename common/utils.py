
import torch as th
from torch.autograd import Variable
import numpy as np


def identity(x):
    return x


def entropy(p):
    return -th.sum(p * th.log(p), 1)


def kl_log_probs(log_p1, log_p2):
    return -th.sum(th.exp(log_p1)*(log_p2 - log_p1), 1)


def index_to_one_hot(index, dim):
    if isinstance(index, np.int) or isinstance(index, np.int64):
        one_hot = np.zeros(dim)
        one_hot[index] = 1.
    else:
        one_hot = np.zeros((len(index), dim))
        one_hot[np.arange(len(index)), index] = 1.