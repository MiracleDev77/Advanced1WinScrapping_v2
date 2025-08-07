# lstm_attention.py
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from config.paths import Paths
from config.params import LSTMParams
from models.temporal_attention import TemporalAttention

class LSTMModel:
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        
        # Shared LSTM layer
        lstm_out = tf.keras.layers.Bidirectional(
            LSTM(
                LSTMParams.UNITS, 
                return_sequences=True,
                kernel_regularizer=tf.keras.regularizers.l2(LSTMParams.L2_REG),
                dropout=LSTMParams.DROPOUT_RATE
            )
        )(inputs)
        
        # Attention mechanism
        attention = TemporalAttention(LSTMParams.ATTENTION_UNITS)(lstm_out)
        attention = Dropout(LSTMParams.DROPOUT_RATE)(attention)
        
        # Score classification branch (3 classes)
        class_branch = Dense(64, activation='relu')(attention)
        class_branch = Dropout(LSTMParams.DROPOUT_RATE)(class_branch)
        class_output = Dense(3, activation='softmax', name='score_class')(class_branch)
        
        # Period prediction branch (binary)
        period_branch = Dense(32, activation='relu')(attention)
        period_branch = Dropout(LSTMParams.DROPOUT_RATE)(period_branch)
        period_output = Dense(1, activation='sigmoid', name='period')(period_branch)
        
        # Score regression branch
        score_branch = Dense(32, activation='relu')(attention)
        score_branch = Dropout(LSTMParams.DROPOUT_RATE)(score_branch)
        score_output = Dense(1, name='score_reg')(score_branch)

        model = tf.keras.Model(
            inputs=inputs, 
            outputs=[class_output, period_output, score_output]
        )
        
        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=LSTMParams.LEARNING_RATE,
            decay_steps=LSTMParams.DECAY_STEPS,
            decay_rate=LSTMParams.DECAY_RATE
        )
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss={
                'score_class': 'sparse_categorical_crossentropy',
                'period': 'binary_crossentropy',
                'score_reg': 'mse'
            },
            loss_weights={
                'score_class': 0.4,
                'period': 0.3,
                'score_reg': 0.3
            },
            metrics={
                'score_class': 'accuracy',
                'period': 'binary_accuracy',
                'score_reg': 'mae'
            }
        )
        return model

    def train(self, X_train, y_train, X_val, y_val):
        y_train_class, y_train_period, y_train_score = y_train
        y_val_class, y_val_period, y_val_score = y_val
        
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
            X_train,
            {
                'score_class': y_train_class,
                'period': y_train_period,
                'score_reg': y_train_score
            },
            validation_data=(
                X_val, 
                {
                    'score_class': y_val_class,
                    'period': y_val_period,
                    'score_reg': y_val_score
                }
            ),
            epochs=LSTMParams.EPOCHS,
            batch_size=LSTMParams.BATCH_SIZE,
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