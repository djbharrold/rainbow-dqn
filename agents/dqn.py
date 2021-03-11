import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from utilities.buffers import ReplayBuffer
from utilities.models import BasicModel


class DQN(object):
    def __init__(self, n_inputs, n_actions,
                 layer1_dims=128, layer2_dims=128, adam_lr=1e-3, adam_epsilon=1.5e-4, adam_clip=None,
                 gamma=0.99, epsilon_decay=0.001, epsilon_min=0.01, epsilon_eval=0.001, tau=0.001,
                 batch_size=64, update_steps=100, update_soft=False, reward_max=None,
                 mem_size=100000, warmup=1000):

        tf.random.set_seed(1)
        np.random.seed(1)

        self.name = "DQN"
        self.n_inputs = n_inputs                            # Number of inputs
        self.n_actions = n_actions                          # Number of actions
        self.adam_lr = adam_lr                              # Optimiser learning rate
        self.adam_epsilon = adam_epsilon                    # Optimiser epsilon value
        self.adam_clip = adam_clip                          # Max norm value for gradient clipping
        self.gamma = gamma                                  # Discount factor
        self.epsilon = 1.0                                  # Epsilon start value for epsilon-greedy policy
        self.epsilon_decay = epsilon_decay                  # Epsilon decay per step
        self.epsilon_min = epsilon_min                      # Epsilon minimum value for training
        self.epsilon_eval = epsilon_eval                    # Epsilon value for evaluation
        self.tau = tau                                      # Rate of model weight transfer if update_soft is True
        self.batch_size = batch_size                        # Number of batches sampled from replay buffer
        self.update_steps = update_steps                    # Number of steps before model weights are copied to target
        self.update_soft = update_soft                      # Flag for using hard or soft target update
        self.reward_max = reward_max                        # Maximum reward value for clipping
        self.warmup = warmup                                # Step at which learning will begin
        self.action_space = [i for i in range(n_actions)]   # Action space used for random action selection

        self.memory = ReplayBuffer(mem_size=mem_size, n_inputs=n_inputs, discrete=True)
        self.model, self.target_model = self.build_networks(n_actions, layer1_dims, layer2_dims)

    def build_networks(self, n_actions, layer1_dims, layer2_dims):
        """ Create models and copy model weights to target """
        self.model = BasicModel(n_actions=n_actions, layer1_dims=layer1_dims, layer2_dims=layer2_dims)
        self.target_model = BasicModel(n_actions=n_actions, layer1_dims=layer1_dims, layer2_dims=layer2_dims)
        opt = Adam(learning_rate=self.adam_lr, epsilon=self.adam_epsilon, clipnorm=self.adam_clip)
        self.model.compile(optimizer=opt)
        self.hard_target_update()
        return self.model, self.target_model

    def choose_action(self, state, evaluate=False):
        """ Returns action following epsilon-greedy policy """
        rand = np.random.random()
        epsilon = self.epsilon if evaluate is False else self.epsilon_eval
        if rand < epsilon:
            action = np.random.choice(self.action_space)
        else:
            action = self.get_best_action(state)
        return action

    def get_best_action(self, state, noise=False):
        """ Returns action with highest Q-value for state or batch of states """
        q_values = self.model(state)
        action = tf.math.argmax(q_values, axis=1).numpy()
        if state.shape[0] > 1:
            return action
        else:
            return action[0]

    def learn(self):
        """ Train agent using GradientTape """
        if self.memory.mem_cntr < self.warmup:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(batch_size=self.batch_size)
        rewards = rewards if self.reward_max is None else np.clip(rewards, -self.reward_max, self.reward_max)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        with tf.GradientTape() as tape:
            q_eval = self.model(states)
            q_target = self.target_model(states_)
            q_train = q_eval.numpy()
            targets = rewards + self.gamma * tf.math.reduce_max(q_target, axis=1) * (1 - dones)
            q_train[batch_index, actions] = targets
            q_train = tf.convert_to_tensor(q_train)
            loss = tf.keras.losses.mse(y_true=q_train, y_pred=q_eval)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.update_epsilon()
        self.update_target()

    def update_epsilon(self):
        """ Decays epsilon until epsilon_min is reached """
        if self.epsilon * (1 - self.epsilon_decay) >= self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)
        else:
            self.epsilon = self.epsilon_min

    def update_target(self):
        """ Update target network weights """
        if self.update_soft is True:
            self.soft_target_update()
        elif self.memory.mem_cntr % self.update_steps == 0:
            self.hard_target_update()

    def hard_target_update(self):
        """ Copy model weights directly to target model """
        self.target_model.set_weights(self.model.get_weights())

    def soft_target_update(self):
        """ Increment target network weights towards model weights """
        weights_model = self.model.get_weights()
        weights_target = self.target_model.get_weights()
        for i in range(len(weights_model)):
            weights_target[i] = self.tau * weights_model[i] + (1 - self.tau) * weights_target[i]
        self.target_model.set_weights(weights_target)

    def store_transition(self, state, action, reward, state_, done):
        """ Store transition in replay buffer """
        self.memory.store_transition(state, action, reward, state_, done)
  
