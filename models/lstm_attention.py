import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import Loss
from config.paths import Paths
from config.params import LSTMParams, TrainingParams
from models.temporal_attention import TemporalAttention

class FocalLoss(Loss):
    def __init__(self, gamma=3.0, alpha=None, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)
        
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
        # Calcul de p_t
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=3)
        p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        focal_loss = tf.pow(1 - p_t, self.gamma) * ce
        
        if self.alpha is not None:
            # Convertir alpha en tensor
            alpha_t = tf.convert_to_tensor(self.alpha, dtype=tf.float32)
            alpha_t = tf.gather(alpha_t, tf.cast(y_true, tf.int32))
            focal_loss = alpha_t * focal_loss
            
        return tf.reduce_mean(focal_loss)
    
    def get_config(self):
        return {'gamma': self.gamma, 'alpha': self.alpha, 'from_logits': self.from_logits}

class LSTMModel:
    def __init__(self, input_shape):
        self.model = self._build_model(input_shape)

    def _build_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape, dtype=tf.float32)
        
        # Architecture simplifiée
        lstm_out = tf.keras.layers.Bidirectional(
            LSTM(LSTMParams.UNITS, return_sequences=True, dropout=LSTMParams.DROPOUT_RATE)
        )(inputs)
        
        attention = TemporalAttention(LSTMParams.ATTENTION_UNITS)(lstm_out)
        attention = Dropout(LSTMParams.DROPOUT_RATE)(attention)
        
        # Branche classification avec régularisation renforcée
        class_branch = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(attention)
        class_branch = Dropout(0.4)(class_branch)
        class_output = Dense(3, activation='softmax', name='score_class')(class_branch)
        
        # Branche période simplifiée
        period_branch = Dense(32, activation='relu')(attention)
        period_output = Dense(1, activation='sigmoid', name='period')(period_branch)
        
        # Branche régression
        score_output = Dense(1, name='score_reg')(attention)

        model = tf.keras.Model(inputs=inputs, outputs=[class_output, period_output, score_output])
        
        # Compilation avec Focal Loss améliorée
        model.compile(
            optimizer=tf.keras.optimizers.Adam(LSTMParams.LEARNING_RATE),
            loss={
                'score_class': FocalLoss(
                    gamma=TrainingParams.FOCAL_LOSS_GAMMA,
                    alpha=list(TrainingParams.CLASS_WEIGHTS.values())
                ),
                'period': 'binary_crossentropy',
                'score_reg': 'mse'
            },
            loss_weights={
                'score_class': 0.6,  # Poids accru
                'period': 0.2,
                'score_reg': 0.2
            },
            metrics={
                'score_class': 'accuracy',
                'period': 'binary_accuracy',
                'score_reg': 'mae'
            }
        )
        return model

    def train(self, train_dataset, val_dataset):
        """Entraîne le modèle sans utiliser de poids de classe"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=LSTMParams.EARLY_STOPPING_PATIENCE,
                monitor='val_loss',
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(Paths.LSTM_MODEL),
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(Paths.LOGS_DIR),
                histogram_freq=1
            )
        ]
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=LSTMParams.EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        return history

    def load(self):
        self.model = tf.keras.models.load_model(
            str(Paths.LSTM_MODEL),
            custom_objects={
                'TemporalAttention': TemporalAttention,
                'FocalLoss': FocalLoss
            }
        )
        return self