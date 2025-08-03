import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class TemporalAttention(Layer):
    def __init__(self, units, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.units = units
        self.W = Dense(units)
        self.U = Dense(units)
        self.v = Dense(1)

    def build(self, input_shape):
        # Build underlying Dense layers
        self.W.build(input_shape)
        self.U.build(input_shape)
        # For v, input shape is (batch, time_steps, units)
        self.v.build((input_shape[0], input_shape[1], self.units))
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch, time_steps, features)
        score = tf.nn.tanh(self.W(inputs))               # (batch, time_steps, units)
        weights = tf.nn.softmax(self.v(score), axis=1)    # (batch, time_steps, 1)
        context = tf.reduce_sum(weights * inputs, axis=1) # (batch, features)
        return context

    def get_config(self):
        config = super(TemporalAttention, self).get_config()
        config.update({'units': self.units})
        return config

