import numpy as np
import pandas as pd
from data.preprocessor import DataPreprocessor
from models.lstm_attention import LSTMModel
from models.xgboost import XGBoostModel, XGBRegressorModel
from config.paths import Paths
from config.params import DataParams, TrainingParams
import joblib
import os

class TemporalFeatureBuilder:
    def __init__(self, window_size):
        self.window_size = window_size

    def build_features(self, df, start_index):
        end_idx = start_index - 1
        start_idx = max(0, end_idx - self.window_size + 1)
        window = df.iloc[start_idx:end_idx]

        # Compute time span: if datetime index, use seconds, else fallback to sequence length
        if isinstance(window.index, pd.DatetimeIndex) and len(window) > 1:
            time_span = (window.index[-1] - window.index[0]).total_seconds()
        else:
            time_span = max(len(window) - 1, 0)

        if len(window) > 0:
            return np.array([
                # Basic statistics
                window['Score'].mean(),
                window['Score'].std(),
                window['Score'].max(),
                window['Score'].min(),
                np.median(window['Score']),
                
                # Type counts
                (window['Type_encoded'] == 0).sum(),
                (window['Type_encoded'] == 1).sum(),
                (window['Type_encoded'] == 2).sum(),
                
                # Time span feature
                time_span,
                
                # Derived features
                (window['Score'] > 5).sum(),  # high scores
                (window['Score'] < 2).sum(),  # low scores
                window['Score'].diff().mean(),  # average trend
                
                # Volatility
                window['Score'].rolling(3).mean().std() if len(window) >= 3 else 0
            ])
        else:
            # Return zeros for all features if no window
            return np.zeros(13)

class CasinoTrainer:
    def __init__(self, fold=0):
        self.preprocessor = DataPreprocessor()
        self.feature_builder = TemporalFeatureBuilder(DataParams.WINDOW_SIZE)
        self.fold = fold

    def train(self, raw_df):
        # Split temporel
        train_raw, val_raw, test_raw = self.preprocessor.split_data(raw_df)
        
        # Augmentation des données d'entraînement
        if DataParams.AUGMENT_FACTOR > 1:
            train_raw = self.augment_data(train_raw)
        
        print(f"📊 FOLD {self.fold}: train_raw: {len(train_raw)}, val_raw: {len(val_raw)}, test_raw: {len(test_raw)}")

        self.preprocessor.fit(train_raw)
        train = self.preprocessor.transform(train_raw)
        val = self.preprocessor.transform(val_raw)

        X_train, y_train, train_scores = self.preprocessor.prepare_sequences(train)
        X_val, y_val, val_scores = self.preprocessor.prepare_sequences(val)

        n_classes = len(np.unique(y_train))
        
        # Calcul des poids de classes
        class_counts = np.bincount(y_train)
        total = len(y_train)
        class_weights = {i: total/(count * len(class_counts)) for i, count in enumerate(class_counts)}
        
        # Entraînement LSTM
        lstm_model = LSTMModel(
            input_shape=(DataParams.WINDOW_SIZE, len(self.preprocessor.feature_columns)),
            n_classes=n_classes
        )
        lstm_model.train(X_train, y_train, X_val, y_val, class_weights=class_weights)

        # Extraction de features hybrides
        X_train_feat = self._extract_hybrid_features(lstm_model.model, X_train, train)
        X_val_feat = self._extract_hybrid_features(lstm_model.model, X_val, val)

        # Entraînement XGBoost Classifier
        xgb_clf = XGBoostModel(n_classes=n_classes)
        xgb_clf.train(
            np.vstack([X_train_feat, X_val_feat]),
            np.concatenate([y_train, y_val]),
            eval_set=[(X_val_feat, y_val)]
        )
        
        # Entraînement XGBoost Regressor
        xgb_reg = XGBRegressorModel()
        xgb_reg.train(
            np.vstack([X_train_feat, X_val_feat]),
            np.concatenate([train_scores, val_scores]),
            eval_set=[(X_val_feat, val_scores)]
        )

        # Sauvegarde des modèles pour ce fold
        self.save_models(xgb_clf, xgb_reg, suffix=f"_fold_{self.fold}")
        
        # Sauvegarde du préprocesseur
        self.preprocessor.save_artifacts(suffix=f"_fold_{self.fold}")
        
        return lstm_model, xgb_clf, xgb_reg

    def augment_data(self, df):
        """Augmente les données avec du bruit gaussien"""
        augmented = [df]
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        for _ in range(DataParams.AUGMENT_FACTOR - 1):
            augmented_df = df.copy()
            noise = np.random.normal(
                0, 
                DataParams.AUGMENT_NOISE, 
                (len(df), len(numeric_cols)))
            for i, col in enumerate(numeric_cols):
                augmented_df[col] += noise[:, i]
            augmented.append(augmented_df)
        
        return pd.concat(augmented, ignore_index=True)

    def _extract_hybrid_features(self, lstm_model, sequences, df):
        lstm_features = lstm_model.predict(sequences)
        temporal_features = []
        for i in range(DataParams.WINDOW_SIZE, len(df)):
            temporal_features.append(
                self.feature_builder.build_features(df, i)
            )
        return np.hstack([lstm_features, np.array(temporal_features)])

    def save_models(self, xgb_clf, xgb_reg, suffix=""):
        xgb_clf.save(Paths.XGB_MODEL.with_stem(Paths.XGB_MODEL.stem + suffix))
        xgb_reg.save(Paths.XGB_REGRESSOR.with_stem(Paths.XGB_REGRESSOR.stem + suffix))
        joblib.dump({
            'window_size': DataParams.WINDOW_SIZE,
            'min_confidence': TrainingParams.MIN_CONFIDENCE,
            'fold': self.fold
        }, Paths.MODEL_DIR / f'config{suffix}.joblib')
