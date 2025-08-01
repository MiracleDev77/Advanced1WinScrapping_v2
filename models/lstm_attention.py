import tensorflow as tf
from tensorflow.keras.layers import Layer, LSTM, Dense, Dropout
from config.paths import Paths
from config.params import LSTMParams

class TemporalAttention(Layer):
    def __init__(self, units):
        super().__init__()
        self.W = Dense(units)
        self.V = Dense(1)

    def call(self, inputs):
        score = self.V(tf.nn.tanh(self.W(inputs)))
        attention_weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(attention_weights * inputs, axis=1)

class LSTMModel:
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)
        
    def _build_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        
        # Couche Bidirectionnelle
        lstm_out = tf.keras.layers.Bidirectional(
            LSTM(LSTMParams.UNITS, return_sequences=True)
        )(inputs)
        
        # Mécanisme d'Attention
        attention = TemporalAttention(LSTMParams.ATTENTION_UNITS)(lstm_out)
        
        # Classification
        outputs = Dense(6, activation='softmax')(attention)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LSTMParams.LEARNING_RATE),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train(self, X_train, y_train, X_val, y_val, class_weights=None):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=LSTMParams.PATIENCE,  # Paramètre corrigé
                monitor='val_loss',
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                Paths.LSTM_MODEL,
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=LSTMParams.EPOCHS,
            batch_size=LSTMParams.BATCH_SIZE,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        return history
    
    def load(self):
        self.model = tf.keras.models.load_model(
            Paths.LSTM_MODEL,
            custom_objects={'TemporalAttention': TemporalAttention}
        )
        return self