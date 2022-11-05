
from ACKTR import DisjointACKTR as ACKTR
from ACKTR import JointACKTR as ACKTR
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
# only use the latest ROLL_OUT_N_STEPS for training A2C
BATCH_SIZE = ROLL_OUT_N_STEPS

REWARD_DISCOUNTED_GAMMA = 0.99
ENTROPY_REG = 0.00
#
DONE_PENALTY = -10.

CRITIC_LOSS = "mse"
MAX_GRAD_NORM = None

EPSILON_START = 0.99
EPSILON_END = 0.05
EPSILON_DECAY = 500

RANDOM_SEED = 2017


def run(env_id="CartPole-v0"):

    env = gym.make(env_id)
    env.seed(RANDOM_SEED)
    env_eval = gym.make(env