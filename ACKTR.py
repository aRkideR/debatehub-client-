
import torch as th
from torch import nn

import numpy as np

from A2C import A2C
from common.Model import ActorCriticNetwork
from common.kfac import KFACOptimizer
from common.utils import index_to_one_hot, entropy, to_tensor_var


class DisjointACKTR(A2C):
    """
    An agent learned with ACKTR
    Using disjoint actor and critic results in instability in training.
    """
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=10,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001, vf_fisher_coef=0.5,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True):
        super(DisjointACKTR, self).__init__(env, state_dim, action_dim,
                 memory_capacity, max_steps, roll_out_n_steps,
                 reward_gamma, reward_scale, done_penalty,
                 actor_hidden_size, critic_hidden_size,
                 actor_output_act, critic_loss,
                 actor_lr, critic_lr,
                 optimizer_type, entropy_reg,
                 max_grad_norm, batch_size, episodes_before_train,
                 epsilon_start, epsilon_end, epsilon_decay,
                 use_cuda)

        self.actor_optimizer = KFACOptimizer(self.actor, lr=self.actor_lr)
        self.critic_optimizer = KFACOptimizer(self.critic, lr=self.critic_lr)
        self.vf_fisher_coef = vf_fisher_coef

    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            pass

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.state_dim)
        one_hot_actions = index_to_one_hot(batch.actions, self.action_dim)
        actions_var = to_tensor_var(one_hot_actions, self.use_cuda).view(-1, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, 1)

        # update actor network
        action_log_probs = self.actor(states_var)
        entropy_loss = th.mean(entropy(th.exp(action_log_probs)))
        action_log_probs = th.sum(action_log_probs * actions_var, 1)
        # fisher loss
        if self.actor_optimizer.steps % self.actor_optimizer.Ts == 0:
            self.actor.zero_grad()
            pg_fisher_loss = th.mean(action_log_probs)
            self.actor_optimizer.acc_stats = True
            pg_fisher_loss.backward(retain_graph=True)
            self.actor_optimizer.acc_stats = False
        self.actor_optimizer.zero_grad(