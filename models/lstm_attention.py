import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from config.paths import Paths
from config.params import LSTMParams
from models.temporal_attention import TemporalAttention

class LSTMModel:
    def __init__(self, input_shape, n_classes):
        self.model = self._build_model(input_shape, n_classes)

    def _build_model(self, input_shape, n_classes):
        inputs = tf.keras.Input(shape=input_shape)
        lstm_out = tf.keras.layers.Bidirectional(
            LSTM(LSTMParams.UNITS, return_sequences=True)
        )(inputs)
        attention = TemporalAttention(LSTMParams.ATTENTION_UNITS)(lstm_out)
        outputs = Dense(n_classes, activation='softmax')(attention)

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
                patience=LSTMParams.EARLY_STOPPING_PATIENCE,
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
