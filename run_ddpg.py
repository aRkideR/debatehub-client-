
from DDPG import DDPG
from common.utils import agg_double_list

import gym
import sys
import numpy as np
import matplotlib.pyplot as plt


MAX_EPISODES = 5000
EPISODES_BEFORE_TRAIN = 100
EVAL_EPISODES = 10
EVAL_INTERVAL = 100

# max steps in each episode, prevent from running too long
MAX_STEPS = 10000 # None

MEMORY_CAPACITY = 10000
BATCH_SIZE = 100
CRITIC_LOSS = "mse"
MAX_GRAD