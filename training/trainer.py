import numpy as np
from data.preprocessor import DataPreprocessor
from models.lstm_attention import LSTMModel
from models.xgboost import XGBoostModel, XGBRegressorModel  # Nouveau modèle
from config.paths import Paths
from config.params import DataParams
import joblib
from sklearn.model_selection import train_test_split

class TemporalFeatureBuilder:
    """Constructeur de features temporelles sans fuite de données"""
    def __init__(self, window_size):
        self.window_size = window_size
    
    def build_features(self, df, start_index):
        """Construit des features à partir d'une fenêtre temporelle décalée"""
        # Indices valides (évite les valeurs futures)
        end_index = start_index - 1
        start_index = max(0, end_index - self.window_size + 1)
        
        window = df.iloc[start_index:end_index]
        
        features = []
        # Statistiques de base
        if len(window) > 0:
            features.extend([
                window['Score'].mean(),
                window['Score'].std(),
                (window['Type_encoded'] == 0).sum()
            ])
        else:
            features.extend([0, 0, 0])  # Valeurs par défaut
            
        return np.array(features)

class CasinoTrainer:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.feature_builder = TemporalFeatureBuilder(DataParams.WINDOW_SIZE)
        
    def train(self, raw_df):
        # 1. Split temporel strict
        train_raw, val_raw, test_raw = self.preprocessor.split_data(raw_df)
        
        # 2. Préprocessing sans fuite
        self.preprocessor.fit(train_raw)
        train = self.preprocessor.transform(train_raw)
        val = self.preprocessor.transform(val_raw)
        
        # 3. Préparation des séquences (avec scores réels)
        X_train, y_train, train_scores = self.preprocessor.prepare_sequences(train)
        X_val, y_val, val_scores = self.preprocessor.prepare_sequences(val)
        
        # 4. Entraînement LSTM
        lstm = LSTMModel(input_shape=(DataParams.WINDOW_SIZE, 3))
        lstm.train(X_train, y_train, X_val, y_val)
        
        # 5. Extraction des features hybrides
        X_train_features = self._extract_hybrid_features(lstm.model, X_train, train)
        X_val_features = self._extract_hybrid_features(lstm.model, X_val, val)
        
        # 6. Entraînement XGBoost classificateur
        xgb_clf = XGBoostModel(n_classes=len(np.unique(y_train)))
        xgb_clf.train(
            np.vstack([X_train_features, X_val_features]),
            np.concatenate([y_train, y_val])
        )
        xgb_clf.save(Paths.XGB_MODEL)
        
        # 7. Entraînement XGBoost régresseur (pour prédiction de score)
        xgb_reg = XGBRegressorModel()
        xgb_reg.train(
            np.vstack([X_train_features, X_val_features]),
            np.concatenate([train_scores, val_scores])
        )
        xgb_reg.save(Paths.XGB_REGRESSOR)
        
        # 8. Sauvegarde des artefacts
        self.preprocessor.save_artifacts()
        joblib.dump({
            'window_size': DataParams.WINDOW_SIZE,
            'min_confidence': 0.99  # Seuil de confiance
        }, Paths.MODEL_DIR / 'config.joblib')
        
        return lstm, xgb_clf, xgb_reg
    
    def _extract_hybrid_features(self, lstm_model, sequences, df):
        """Extrait les features hybrides avec alignement temporel correct"""
        lstm_features = lstm_model.predict(sequences)
        temporal_features = []
        
        # Les séquences commencent à l'index window_size
        for i in range(DataParams.WINDOW_SIZE, len(df)):
            features = self.feature_builder.build_features(df, i)
            temporal_features.append(features)
            
        return np.hstack([lstm_features, np.array(temporal_features)])