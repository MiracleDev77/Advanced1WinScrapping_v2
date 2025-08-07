import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from config.paths import Paths
from config.params import LSTMParams
from models.temporal_attention import TemporalAttention

class LSTMModel:
    def __init__(self, input_shape, n_classes):
        self.model = self._build_model(input_shape, n_classes)

    def _build_model(self, input_shape, n_classes):
        inputs = tf.keras.Input(shape=input_shape)
        
        # Couche LSTM bidirectionnelle avec régularisation
        lstm_out = tf.keras.layers.Bidirectional(
            LSTM(
                LSTMParams.UNITS, 
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(LSTMParams.L2_REG),
                dropout=LSTMParams.DROPOUT_RATE
            )
        )(inputs)
        
        # Mécanisme d'attention
        attention = TemporalAttention(LSTMParams.ATTENTION_UNITS)(lstm_out)
        attention = Dropout(LSTMParams.DROPOUT_RATE)(attention)
        
        # Couche dense intermédiaire
        dense = Dense(64, activation='relu')(attention)
        dense = Dropout(LSTMParams.DROPOUT_RATE)(dense)
        
        outputs = Dense(n_classes, activation='softmax')(dense)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Utiliser un taux d'apprentissage fixe pour compatibilité avec ReduceLROnPlateau
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LSTMParams.LEARNING_RATE),
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
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
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
