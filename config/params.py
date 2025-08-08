import tensorflow as tf

class LSTMParams:
    UNITS = 64  # Réduit la complexité
    DROPOUT_RATE = 0.3
    L2_REG = 0.005
    LEARNING_RATE = 0.001
    EPOCHS = 50  # Réduit le nombre d'epochs
    BATCH_SIZE = 32  # Batch size plus petit pour convergence
    ATTENTION_UNITS = 32
    EARLY_STOPPING_PATIENCE = 5
    DECAY_STEPS = 1000
    DECAY_RATE = 0.9

class XGBoostParams:
    TUNING_ITERATIONS = 50
    EARLY_STOPPING = 20
    BASE_PARAMS = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'tree_method': 'gpu_hist' if tf.config.list_physical_devices('GPU') else 'hist',
        'verbosity': 0,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'n_estimators': 500
    }
    REGRESSOR_PARAMS = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'gpu_hist' if tf.config.list_physical_devices('GPU') else 'hist',
        'verbosity': 0,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'n_estimators': 500
    }

class DataParams:
    WINDOW_SIZE = 10  # Fenêtre réduite
    TEST_SIZE = 0.10
    VAL_SIZE = 0.10
    AUGMENT_FACTOR = 5  # Augmentation plus importante
    AUGMENT_MINORITY_FACTOR = 10  # Facteur spécifique pour la classe minoritaire
    AUGMENT_NOISE = 0.05
    USE_SMOTE = True
    STRATIFY_SPLIT = True
    BALANCE_CLASSES = True  # Activer le rééquilibrage

class TrainingParams:
    EARLY_STOPPING_PATIENCE = 5
    MIN_DELTA = 0.001
    MIN_CONFIDENCE = 0.99
    NUM_FOLDS = 5  # Plus de folds pour une meilleure validation
    FOCAL_LOSS_GAMMA = 3.0  # Gamma augmenté
    CLASS_WEIGHTS = {0: 1.0, 1: 10.0, 2: 1.0}  # Poids pour la classe 1 augmenté