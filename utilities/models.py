import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from utilities.layers import NoisyNetDense


class BasicModel(keras.Model):
    def __init__(self, n_actions, layer1_dims, layer2_dims):
        super(BasicModel, self).__init__()
        self.dense1 = Dense(layer1_dims, activation="relu")
        self.dense2 = Dense(layer2_dims, activation="relu")
        self.Q = Dense(n_actions, activation="linear")

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        q = self.Q(x)
        return q

      
class RainbowModel(keras.Model):
    def __init__(self, n_actions, layer1_dims, layer2_dims, n_atoms, sigma_init=0.5):
        super(RainbowModel, self).__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.dense1 = Dense(layer1_dims, activation="relu")
        self.dense2 = Dense(layer2_dims, activation="relu")
        self.V = NoisyNetDense(n_atoms, activation="linear", sigma_zero=sigma_init)
        self.A = NoisyNetDense(n_actions * n_atoms, activation="linear", sigma_zero=sigma_init)

    def call(self, state, noise=True):
        x = self.dense1(state)
        x = self.dense2(x)
        v_logits = self.V(x, noise=noise)
        a_logits = self.A(x, noise=noise)
        v = tf.reshape(v_logits, [-1, 1, self.n_atoms])
        a = tf.reshape(a_logits, [-1, self.n_actions, self.n_atoms])
        q = v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True))
        probs = tf.keras.activations.softmax(q)
        return probs
