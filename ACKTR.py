
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
        self.actor_optimizer.zero_grad()
        # actor loss
        values = self.critic(states_var, actions_var).detach()
        advantages = rewards_var - values
        pg_loss = -th.mean(action_log_probs * advantages)
        actor_loss = pg_loss - entropy_loss * self.entropy_reg
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        # update critic network
        target_values = rewards_var
        values = self.critic(states_var, actions_var)
        # fisher loss
        if self.critic_optimizer.steps % self.critic_optimizer.Ts == 0:
            self.critic.zero_grad()
            values_noise = to_tensor_var(np.random.randn(values.size()[0]), self.use_cuda)
            sample_values = (values + values_noise.view(-1, 1)).detach()
            if self.critic_loss == "huber":
                vf_fisher_loss = - self.vf_fisher_coef * nn.functional.smooth_l1_loss(values, sample_values)
            else:
                vf_fisher_loss = - self.vf_fisher_coef * nn.MSELoss()(values, sample_values)
            self.critic_optimizer.acc_stats = True
            vf_fisher_loss.backward(retain_graph=True)
            self.critic_optimizer.acc_stats = False
        self.critic_optimizer.zero_grad()
        # critic loss
        if self.critic_loss == "huber":
            critic_loss = nn.functional.smooth_l1_loss(values, target_values)
        else:
            critic_loss = nn.MSELoss()(values, target_values)
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

class JointACKTR(A2C):
    """
    An agent learned with ACKTR
    Compared with DisjointACKTR, JointACKTR is more stable.
    """
    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=None,
                 roll_out_n_steps=10,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.001, critic_lr=0.001, vf_coef=0.5, vf_fisher_coef=1.0,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 epsilon_start=0.9, epsilon_end=0.01, epsilon_decay=200,
                 use_cuda=True):
        super(JointACKTR, self).__init__(env, state_dim, action_dim,
                 memory_capacity, max_steps, roll_out_n_steps,
                 reward_gamma, reward_scale, done_penalty,
                 actor_hidden_size, critic_hidden_size,
                 actor_output_act, critic_loss,
                 actor_lr, critic_lr,
                 optimizer_type, entropy_reg,
                 max_grad_norm, batch_size, episodes_before_train,
                 epsilon_start, epsilon_end, epsilon_decay,
                 use_cuda)

        self.actor_critic = ActorCriticNetwork(self.state_dim, self.action_dim,
                                               min(self.actor_hidden_size, self.critic_hidden_size),
                                               self.actor_output_act)
        self.optimizer = KFACOptimizer(self.actor_critic, lr=min(self.actor_lr, self.critic_lr))
        self.vf_coef = vf_coef
        self.vf_fisher_coef = vf_fisher_coef

    # train on a roll out batch
    def train(self):
        