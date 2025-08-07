from pathlib import Path

class Paths:
    # Répertoires de base
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / 'data'
    MODEL_DIR = BASE_DIR / 'models'
    REPORT_DIR = BASE_DIR / 'reports'
    
    # Fichiers de données
    DATABASE = BASE_DIR / 'dataset.db'
    
    # Modèles
    LSTM_MODEL = MODEL_DIR / 'lstm_model.keras'
    XGB_MODEL = MODEL_DIR / 'xgb_model.json'
    XGB_REGRESSOR = MODEL_DIR / 'xgb_regressor.json' 
    SCALER = MODEL_DIR / 'scaler.joblib'
    ENCODER = MODEL_DIR / 'encoder.joblib'
    PREPROCESSOR = MODEL_DIR / 'preprocessor.joblib'
    
    # Rapports
    DRIFT_REPORT = REPORT_DIR / 'data_drift_report.html'