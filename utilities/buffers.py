import numpy as np


class ReplayBuffer(object):
    def __init__(self, mem_size, n_inputs, discrete=True):
        self.mem_cntr = 0
        self.mem_size = mem_size
        a_dtype = np.int8 if discrete is True else np.float32

        self.state_memory = np.zeros((self.mem_size, n_inputs), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=a_dtype)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.next_state_memory = np.zeros((self.mem_size, n_inputs), dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.reward_memory[index] = reward
        self.next_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.action_memory[index] = action
        self.mem_cntr += 1

    def sample_buffer(self, batch_size, priority_scale=None):
        batch = np.random.choice(self.max_mem(), size=batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, states_, dones

    def max_mem(self):
        return min(self.mem_cntr, self.mem_size)


class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self, mem_size, n_inputs, discrete=True):
        super().__init__(mem_size, n_inputs, discrete)
        self.priorities = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.priorities[index] = max(self.priorities[:self.max_mem()], default=1)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size, priority_scale=1.0):
        sample_probabilities = self.get_probabilities(priority_scale)
        batch = np.random.choice(self.max_mem(), size=batch_size, replace=False, p=sample_probabilities)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        dones = self.terminal_memory[batch]
        weight = self.get_importance_weight(sample_probabilities[batch])
        return states, actions, rewards, states_, dones, weight, batch

    def get_probabilities(self, priority_scale):
        priorities = self.priorities[:self.max_mem()]
        scaled_priorities = priorities ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance_weight(self, probabilities):
        weight = 1 / (self.max_mem() * probabilities)
        weight /= np.max(weight)
        return weight

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset
