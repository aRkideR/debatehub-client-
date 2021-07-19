
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
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True, training_strategy="cocurrent",
                 actor_parameter_sharing=False, critic_parameter_sharing=False):
        super(MAA2C, self).__init__(env, state_dim, action_dim,
                 memory_capacity, max_steps,
                 reward_gamma, reward_scale, done_penalty,
                 actor_hidden_size, critic_hidden_size,
                 actor_output_act, critic_loss,
                 actor_lr, critic_lr,
                 optimizer_type, entropy_reg,
                 max_grad_norm, batch_size, episodes_before_train,
                 epsilon_start, epsilon_end, epsilon_decay,
                 use_cuda)

        assert training_strategy in ["cocurrent", "centralized"]

        self.n_agents = n_agents
        self.roll_out_n_steps = roll_out_n_steps
        self.training_strategy = training_strategy
        self.actor_parameter_sharing = actor_parameter_sharing
        self.critic_parameter_sharing = critic_parameter_sharing

        self.actors = [ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim, self.actor_output_act)] * self.n_agents
        if self.training_strategy == "cocurrent":
            self.critics = [CriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1)] * self.n_agents
        elif self.training_strategy == "centralized":
            critic_state_dim = self.n_agents * self.state_dim
            critic_action_dim = self.n_agents * self.action_dim
            self.critics = [CriticNetwork(critic_state_dim, critic_action_dim, self.critic_hidden_size, 1)] * self.n_agents
        if optimizer_type == "adam":
            self.actor_optimizers = [Adam(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critic_optimizers = [Adam(c.parameters(), lr=self.critic_lr) for c in self.critics]
        elif optimizer_type == "rmsprop":
            self.actor_optimizers = [RMSprop(a.parameters(), lr=self.actor_lr) for a in self.actors]
            self.critic_optimizers = [RMSprop(c.parameters(), lr=self.critic_lr) for c in self.critics]

        # tricky and memory consumed implementation of parameter sharing
        if self.actor_parameter_sharing:
            for agent_id in range(1, self.n_agents):
                self.actors[agent_id] = self.actors[0]
                self.actor_optimizers[agent_id] = self.actor_optimizers[0]
        if self.critic_parameter_sharing:
            for agent_id in range(1, self.n_agents):
                self.critics[agent_id] = self.critics[0]
                self.critic_optimizers[agent_id] = self.critic_optimizers[0]

        if self.use_cuda:
            for a in self.actors:
                a.cuda()
            for c in self.critics:
                c.cuda()

    # agent interact with the environment to collect experience
    def interact(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            self.env_state = self.env.reset()
            self.n_steps = 0
        states = []
        actions = []
        rewards = []
        # take n steps
        for i in range(self.roll_out_n_steps):
            states.append(self.env_state)
            action = self.exploration_action(self.env_state)
            next_state, reward, done, _ = self.env.step(action)
            done = done[0]
            actions.append([index_to_one_hot(a, self.action_dim) for a in action])
            rewards.append(reward)
            final_state = next_state
            self.env_state = next_state
            if done:
                self.env_state = self.env.reset()
                break
        # discount reward
        if done:
            final_r = [0.0] * self.n_agents
            self.n_episodes += 1
            self.episode_done = True
        else:
            self.episode_done = False
            final_action = self.action(final_state)
            one_hot