import tensorflow.keras.backend as K
import tensorflow.keras.layers as kl
from tensorflow.keras import activations, constraints, initializers, regularizers


class NoisyNetDense(kl.Layer):
    def __init__(self, units, sigma_zero=0.5, activation=None, use_bias=True,
                 kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, activity_regularizer=None, bias_regularizer=None,
                 kernel_constraint=None, bias_constraint=None,
                 **kwargs):

        super(NoisyNetDense, self).__init__(**kwargs)

        self.units = units
        self.sigma_init = sigma_zero
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        sqr_inputs = self.input_dim ** 0.5
        mu_range = 1 / sqr_inputs
        self.sigma_initializer = initializers.Constant(value=(self.sigma_init / sqr_inputs))
        self.mu_initializer = initializers.RandomUniform(minval=-mu_range, maxval=mu_range)

        self.mu_weight = self.add_weight(
            shape=(self.input_dim, self.units), initializer=self.mu_initializer, name='mu_weights',
            constraint=self.kernel_constraint, regularizer=self.kernel_regularizer)

        self.sigma_weight = self.add_weight(
            shape=(self.input_dim, self.units), initializer=self.sigma_initializer, name='sigma_weights',
            constraint=self.kernel_constraint, regularizer=self.kernel_regularizer)

        self.mu_bias = self.add_weight(
            shape=(self.units,), initializer=self.mu_initializer, name='mu_bias',
            constraint=self.bias_constraint, regularizer=self.bias_regularizer)

        self.sigma_bias = self.add_weight(
            shape=(self.units,), initializer=self.sigma_initializer, name='sigma_bias',
            constraint=self.bias_constraint, regularizer=self.bias_regularizer)

        super(NoisyNetDense, self).build(input_shape=input_shape)

    def call(self, x, noise=True):
        e_w, e_b = self.get_noise_params(noise)
        noise_injected_weights = K.dot(x, self.mu_weight + (self.sigma_weight * e_w))
        noise_injected_bias = self.mu_bias + (self.sigma_bias * e_b)
        output = K.bias_add(noise_injected_weights, noise_injected_bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_noise_params(self, noise=True):
        if noise is True:
            e_i = K.random_normal((self.input_dim, self.units))
            e_j = K.random_normal((self.units,))
            e_w = K.sign(e_i) * K.sqrt(K.abs(e_i)) * K.sign(e_j) * K.sqrt(K.abs(e_j))
            e_b = K.sign(e_j) * K.sqrt(K.abs(e_j))
            return e_w, e_b
        else:
            return 0, 0

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)
