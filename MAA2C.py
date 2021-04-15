
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
                different reward structure, e.g., cooperative, competitive, mix