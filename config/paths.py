# # paths.py
# from pathlib import Path

# class Paths:
#     BASE_DIR = Path(__file__).resolve().parent.parent
#     DATA_DIR = BASE_DIR / 'data'
#     MODEL_DIR = BASE_DIR / 'models'
#     REPORT_DIR = BASE_DIR / 'reports'
#     LOGS_DIR = BASE_DIR / 'logs'
    
#     DATABASE = DATA_DIR / 'dataset.db'
#     SCALER = MODEL_DIR / 'scaler.joblib'
#     ENCODER = MODEL_DIR / 'encoder.joblib'
#     PREPROCESSOR = MODEL_DIR / 'preprocessor.joblib'
#     LSTM_MODEL = MODEL_DIR / 'lstm_model.keras'
#     XGB_MODEL = MODEL_DIR / 'xgb_classifier.json'
#     XGB_PERIOD = MODEL_DIR / 'xgb_period.json'
#     XGB_REGRESSOR = MODEL_DIR / 'xgb_regressor.json'

from pathlib import Path

class Paths:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODEL_DIR = BASE_DIR / 'models'
    LOGS_DIR = BASE_DIR / 'logs'  # Nouveau chemin ajouté
    REPORT_DIR = BASE_DIR / 'reports'
    
    DATABASE = DATA_DIR / 'dataset.db'
    SCALER = MODEL_DIR / 'scaler.joblib'
    PREPROCESSOR = MODEL_DIR / 'preprocessor.joblib'
    LSTM_MODEL = MODEL_DIR / 'lstm_model.keras'
    XGB_MODEL = MODEL_DIR / 'xgb_classifier.json'
    XGB_PERIOD = MODEL_DIR / 'xgb_period.json'
    XGB_REGRESSOR = MODEL_DIR / 'xgb_regressor.json'

    # Créer les répertoires si nécessaire
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    MODEL_DIR.mkdir(exist_ok=True, parents=True)
    LOGS_DIR.mkdir(exist_ok=True, parents=True)  # Création du répertoire de logs
    REPORT_DIR.mkdir(exist_ok=True, parents=True)
