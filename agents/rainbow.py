import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops import math_ops
from collections import deque
from agents.dqn import DQN
from utilities.rl.buffers import PriorityReplayBuffer
from utilities.rl.models import RainbowModel


class RainbowDQN(DQN):
    def __init__(self, n_inputs, n_actions,
                 layer1_dims=128, layer2_dims=128, adam_lr=1e-3, adam_epsilon=1.5e-4, adam_clip=None,
                 gamma=0.99, epsilon_decay=0.001, epsilon_min=0.01, epsilon_eval=0.001, tau=0.001,
                 batch_size=64, update_steps=100, update_soft=False, reward_max=None,
                 mem_size=100000, warmup=1000,
                 per_alpha=0.7, per_beta=0.5, per_beta_inc=1e-4, per_offset=0.001, per_error_max=1.0,
                 n_step=1, sigma_init=0.5, n_atoms=51, v_max=10, v_min=None):

        self.sigma_init = sigma_init
        self.n_atoms = n_atoms
        super().__init__(n_inputs, n_actions, layer1_dims, layer2_dims, adam_lr, adam_epsilon, adam_clip,
                         gamma, epsilon_decay, epsilon_min, epsilon_eval, tau, batch_size,
                         update_steps, update_soft, reward_max, mem_size, warmup)

        self.name = "Rainbow"
        self.memory = PriorityReplayBuffer(mem_size=mem_size, n_inputs=n_inputs, discrete=True)
        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_inc = per_beta_inc
        self.per_offset = per_offset
        self.per_error_max = per_error_max

        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=n_step)
        self.reward_max = None if reward_max is None else reward_max * self.n_step

        self.v_max = v_max
        self.v_min = v_min if v_min is not None else -v_max
        self.support_z = np.linspace(self.v_min, self.v_max, n_atoms)
        self.delta_z = (self.v_max - self.v_min) / (n_atoms - 1)

    def build_networks(self, n_actions, layer1_dims, layer2_dims):
        """ Build networks using noisy-dense layers with dueling architecture """
        self.model = RainbowModel(
            n_actions, layer1_dims, layer2_dims, sigma_init=self.sigma_init, n_atoms=self.n_atoms)
        self.target_model = RainbowModel(
            n_actions, layer1_dims, layer2_dims, sigma_init=self.sigma_init, n_atoms=self.n_atoms)
        opt = Adam(lr=self.adam_lr, epsilon=self.adam_epsilon, clipvalue=self.adam_clip)
        self.model.compile(optimizer=opt)
        self.hard_target_update()
        return self.model, self.target_model

    def choose_action(self, state, evaluate=False):
        """ Returns action following greedy policy """
        if self.memory.mem_cntr < self.warmup:
            action = np.random.choice(self.action_space)
        else:
            noise = True if evaluate is False else False
            action = self.get_best_action(state, noise)
        return action

    def get_best_action(self, state, noise=True):
        """ Returns action with highest Q-value for state or batch of states """
        probs = self.model(state).numpy() if noise is True else self.model(state, noise=False).numpy()
        q_values = self.get_q_values(probs)
        action = np.argmax(q_values, axis=-1)
        if action.shape[0] > 1:
            return action
        else:
            return action[0]

    def get_q_values(self, dist):
        """ Calculate Q-values from distribution """
        return np.sum(dist * self.support_z, axis=2)

    def learn(self):
        """ Train agent using categorical algorithm and KLD loss function """
        if self.memory.mem_cntr < self.warmup:
            return

        states, actions, rewards, states_, dones, weights, indicies = \
            self.memory.sample_buffer(batch_size=self.batch_size, priority_scale=self.per_alpha)
        rewards = rewards if self.reward_max is None else np.clip(rewards, -self.reward_max, self.reward_max)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        scaled_weights = np.expand_dims(weights ** self.per_beta, axis=1)

        target_dist = self.target_model(states_).numpy()
        max_actions = self.get_best_action(states_)
        target_dist_max = target_dist[batch_index, max_actions]

        with tf.GradientTape() as tape:
            eval_dist = self.model(states)
            proj_dist = self.get_projected_dist(target_dist_max, rewards, dones)
            train_dist = np.zeros_like(target_dist)
            train_dist[batch_index, actions] = proj_dist
            train_dist = tf.convert_to_tensor(train_dist)
            loss = tf.keras.losses.kld(y_true=train_dist, y_pred=eval_dist)
            per_loss = tf.multiply(loss, scaled_weights)
        gradients = tape.gradient(per_loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        kld_loss = loss.numpy()
        kl_error = np.abs(kld_loss[batch_index, actions])
        clipped_error = kl_error if self.per_error_max is None else np.clip(kl_error, 0, self.per_error_max)
        self.memory.set_priorities(indicies, clipped_error, self.per_offset)

        self.update_mem_beta()
        self.update_target()

    def update_mem_beta(self):
        """ Update importance sampling parameter """
        if self.per_beta * (1 + self.per_beta_inc) <= 1:
            self.per_beta *= (1 + self.per_beta_inc)
        else:
            self.per_beta = 1

    def store_transition(self, state, action, reward, state_, done):
        """ Store transition in n-step buffer and add to replay buffer """
        if self.n_step > 1:
            self.n_step_buffer.append([state, action, reward, state_, done])
            if len(self.n_step_buffer) == self.n_step:
                ns_return, ns_state_, ns_done = self.get_return()
                ns_state, ns_action = self.n_step_buffer[0][:2]
                self.memory.store_transition(ns_state, ns_action, ns_return, ns_state_, ns_done)
        else:
            self.memory.store_transition(state, action, reward, state_, done)

    def get_return(self):
      """ Calculate n-step return and resulting state """
        g = 0
        s_ = 0
        d = 0
        for i in range(self.n_step):
            r, s_, d = self.n_step_buffer[i][2:]
            g += r * (self.gamma ** i)
            if d == 1:
                return g, s_, d
        return g, s_, d

    def get_projected_dist(self, target_dist, rewards, dones):
        """ Categorical algorithm to get target distribution for KLD loss """
        rewards = np.expand_dims(rewards, 1)
        dones = np.expand_dims(dones, 1)
        m = np.zeros(shape=(self.batch_size, self.n_atoms), dtype=np.float32)
        tz = rewards + self.support_z * (self.gamma ** self.n_step) * (1 - dones)
        tz = np.clip(tz, self.v_min, self.v_max)
        b = (tz - self.v_min) / self.delta_z
        l = np.floor(b).astype(np.int64)
        u = np.ceil(b).astype(np.int64)
        l_dist = np.squeeze(target_dist * [u - b], axis=0)
        u_dist = np.squeeze(target_dist * [b - l], axis=0)
        for i in range(self.batch_size):
            for j in range(self.n_atoms):
                m[i, l[i, j]] += l_dist[i, j]
                m[i, u[i, j]] += u_dist[i, j]
        m = np.clip(m, 0.0, 1.0)
        return m
