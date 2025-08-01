import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import xgboost as xgb
import joblib
from datetime import datetime
from config.paths import Paths
from typing import Dict, Any
from models.xgboost import XGBoostModel, XGBRegressorModel

class CasinoPredictor:
    def __init__(self):
        # Charger tous les artefacts nécessaires
        self.preprocessor = joblib.load(Paths.PREPROCESSOR)
        self.lstm = load_model(Paths.LSTM_MODEL)
        self.config = joblib.load(Paths.MODEL_DIR / 'config.joblib')
        
        # Charger les classificateurs
        self.xgb_clf = XGBoostModel.load(
            Paths.XGB_MODEL,
            len(self.preprocessor.encoder.classes_)
        ).model
        
        # Charger le régresseur pour le score
        self.xgb_reg = XGBRegressorModel.load(Paths.XGB_REGRESSOR).model
        
    def predict(self, raw_data: pd.DataFrame) -> Dict[str, Any]:
        # 1. Préprocessing
        processed = self.preprocessor.transform(raw_data)
        
        # 2. Préparer la séquence LSTM
        sequence = processed[['Score', 'MoyenneMobileDixDernier', 'Ecart_Type']].tail(self.config['window_size']).values
        lstm_input = sequence[np.newaxis, ...]
        lstm_features = self.lstm.predict(lstm_input)
        
        # 3. Features temporelles (fenêtre historique)
        temporal_features = self._compute_temporal_features(processed)
        
        # 4. Features hybrides
        hybrid_features = np.concatenate([lstm_features, temporal_features], axis=1)
        
        # 5. Prédiction de classe
        xgb_pred = self.xgb_clf.predict(hybrid_features)
        xgb_proba = self.xgb_clf.predict_proba(hybrid_features)
        max_probability = np.max(xgb_proba)
        class_pred = xgb_pred[0]
        
        # Décodage du type de score
        prediction_label = self.preprocessor.encoder.inverse_transform([class_pred])[0]
        
        # 6. Initialisation du résultat
        result = {
            'prediction': prediction_label,
            'probability': float(max_probability),
            'probabilities': dict(zip(
                self.preprocessor.encoder.classes_,
                np.round(xgb_proba[0], 3).tolist()
            )),
            'timestamp': datetime.now().isoformat()
        }
        
        # 7. Prédiction de score si confiance suffisante
        if max_probability >= self.config['min_confidence']:
            predicted_score = self.xgb_reg.predict(hybrid_features)[0]
            result['Score probable'] = float(predicted_score)
        
        return result
    
    def _compute_temporal_features(self, df: pd.DataFrame) -> np.ndarray:
        """Calcule les statistiques temporelles sur la fenêtre historique"""
        # Utiliser uniquement les données passées
        window = df.iloc[-self.config['window_size']:-1]
        
        features = [
            window['Score'].mean(),
            window['Score'].std(),
            (window['Type_encoded'] == 0).sum()
        ]
        
        return np.array([features])