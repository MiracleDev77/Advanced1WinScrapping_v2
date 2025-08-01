class LSTMParams:
    UNITS = 128
    DROPOUT = 0.4
    LEARNING_RATE = 0.001
    EPOCHS = 150
    BATCH_SIZE = 64
    ATTENTION_UNITS = 64
    EARLY_STOPPING_PATIENCE = 10

class XGBoostParams:
    TUNING_ITERATIONS = 100
    EARLY_STOPPING = 30
    BASE_PARAMS = {
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'tree_method': 'gpu_hist',
        'verbosity': 0
    }
    REGRESSOR_PARAMS = {  # Nouveau
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'gpu_hist',
        'verbosity': 0
    }

class DataParams:
    WINDOW_SIZE = 15
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15

class TrainingParams:
    EARLY_STOPPING_PATIENCE = 10
    MIN_DELTA = 0.001
    MIN_CONFIDENCE = 0.99  # Nouveau: seuil de confiance