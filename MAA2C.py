
import torch as th
from torch import nn
from torch.optim import Adam, RMSprop

import numpy as np

from common.Agent import Agent
from common.Model import ActorNetwork, CriticNetwork
from common.utils import entropy, index_to_one_hot, to_tensor_var


class MAA2C(Agent):
    """
    An multi-agent learned with Advantage Actor-Critic
    - Actor takes its local observations as input
    - agent interact with environment to collect experience
    - agent training with experience to update policy

    Parameters
    - training_strategy:
        - cocurrent
            - each agent learns its own individual policy which is independent
            - multiple policies are optimized simultaneously
        - centralized (see MADDPG in [1] for details)
            - centralized training and decentralized execution
            - decentralized actor map it's local observations to action using individual policy
            - centralized critic takes both state and action from all agents as input, each actor
                has its own critic for estimating the value function, which allows each actor has
                different reward structure, e.g., cooperative, competitive, mixed task
    - actor_parameter_sharing:
        - True: all actors share a single policy which enables parameters and experiences sharing,
            this is mostly useful where the agents are homogeneous. Please see Sec. 4.3 in [2] and
            Sec. 4.1 & 4.2 in [3] for details.
        - False: each actor use independent policy
    - critic_parameter_sharing:
        - True: all actors share a single critic which enables parameters and experiences sharing,
            this is mostly useful where the agents are homogeneous and reward sharing holds. Please
            see Sec. 4.1 in [3] for details.
        - False: each actor use independent critic (though each critic can take other agents actions
            as input, see MADDPG in [1] for details)

    Reference:
    [1] Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
    [2] Cooperative Multi-Agent Control Using Deep Reinforcement Learning
    [3] Parameter Sharing Deep Deterministic Policy Gradient for Cooperative Multi-agent Reinforcement Learning

    """
    def __init__(self, env, n_agents, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=10,
                 reward_gamma=0.99, reward_scale=1., 