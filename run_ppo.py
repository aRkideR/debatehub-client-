
from PPO import PPO
from common.utils import agg_double_list

import sys
import gym
import numpy as np
import matplotlib.pyplot as plt


MAX_EPISODES = 5000
EPISODES_BEFORE_TRAIN = 0
EVAL_EPISODES = 10
EVAL_INTERVAL = 100

# roll out n steps
ROLL_OUT_N_STEPS = 10
# only remember the latest ROLL_OUT_N_STEPS
MEMORY_CAPACITY = ROLL_OUT_N_STEPS
# only use the latest ROLL_OUT_N_STEPS for training PPO
BATCH_SIZE = ROLL_OUT_N_STEPS

TARGET_UPDATE_STEPS = 5
TARGET_TAU = 1.0

REWARD_DISCOUNTED_GAMMA = 0.99
ENTROPY_REG = 0.00
#
DONE_PENALTY = -10.

CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

EPSILO