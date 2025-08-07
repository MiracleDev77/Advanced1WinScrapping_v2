class LSTMParams:
    UNITS = 128
    DROPOUT_RATE = 0.4  # Renommé de DROPOUT
    L2_REG = 0.01       # Nouveau: régularisation L2
    LEARNING_RATE = 0.001
    EPOCHS = 150
    BATCH_SIZE = 64
    ATTENTION_UNITS = 64
    EARLY_STOPPING_PATIENCE = 10
    DECAY_STEPS = 1000   # Nouveau: pour le scheduling LR
    DECAY_RATE = 0.9     # Nouveau: pour le scheduling LR

class XGBoostParams:
    TUNING_ITERATIONS = 100
    EARLY_STOPPING = 30
    BASE_PARAMS = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'tree_method': 'gpu_hist',
        'verbosity': 0,
        'learning_rate': 0.1,       # Nouveau
        'max_depth': 5,             # Nouveau
        'subsample': 0.8,           # Nouveau
        'colsample_bytree': 0.8,    # Nouveau
        'n_estimators': 1000        # Nouveau
    }
    REGRESSOR_PARAMS = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'gpu_hist',
        'verbosity': 0,
        'learning_rate': 0.1,       # Nouveau
        'max_depth': 5,             # Nouveau
        'subsample': 0.8,           # Nouveau
        'colsample_bytree': 0.8,    # Nouveau
        'n_estimators': 1000        # Nouveau
    }

class DataParams:
    WINDOW_SIZE = 15
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    AUGMENT_FACTOR = 3   # Nouveau: facteur d'augmentation
    AUGMENT_NOISE = 0.05 # Nouveau: niveau de bruit

class TrainingParams:
    EARLY_STOPPING_PATIENCE = 10
    MIN_DELTA = 0.001
    MIN_CONFIDENCE = 0.99
    NUM_FOLDS = 3        # Nouveau: pour la validation croisée