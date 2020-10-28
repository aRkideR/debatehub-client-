
import torch as th
from torch import nn

import numpy as np

from A2C import A2C
from common.Model import ActorCriticNetwork
from common.kfac import KFACOptimizer
from common.utils import index_to_one_ho