import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout

class TemporalAttention(Layer):
    def __init__(self, units, dropout=0.2, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout
        
        # Couches pour le mécanisme d'attention
        self.W = Dense(units, activation='tanh')
        self.U = Dense(units, activation='sigmoid')
        self.V = Dense(1)
        self.dropout = Dropout(dropout)

    def build(self, input_shape):
        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs):
        # Calcul des scores d'attention
        score = self.V(self.dropout(
            self.W(inputs) * self.U(inputs)
        ))
        
        # Softmax pour les poids d'attention
        weights = tf.nn.softmax(score, axis=1)
        
        # Contexte pondéré
        context = tf.reduce_sum(weights * inputs, axis=1)
        return context

    def get_config(self):
        config = super(TemporalAttention, self).get_config()
        config.update({
            'units': self.units,
            'dropout': self.dropout_rate
        })
        return config