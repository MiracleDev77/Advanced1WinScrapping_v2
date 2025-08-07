# trainer.py
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
        
        if len(window) == 0:
            return np.zeros(15)
        
        features = [
            window['Score'].mean(),
            window['Score'].std(),
            window['Score'].max(),
            window['Score'].min(),
            np.median(window['Score']),
            (window['ScoreClass'] == 0).sum(),
            (window['ScoreClass'] == 1).sum(),
            (window['ScoreClass'] == 2).sum(),
            window['Period'].mean(),
            (window['Period'] == 1).sum(),
            window['score_diff'].mean(),
            window['moyenne_diff'].mean(),
            window['type_repetition'].mean(),
            window['Ecart_Type'].mean(),
            window['MoyenneMobileDixDernier'].mean()
        ]
        return np.array(features)

class CasinoTrainer:
    def __init__(self, fold=0):
        self.preprocessor = DataPreprocessor()
        self.feature_builder = TemporalFeatureBuilder(DataParams.WINDOW_SIZE)
        self.fold = fold

    def train(self, raw_df):
        train_raw, val_raw, test_raw = self.preprocessor.split_data(raw_df)
        
        if DataParams.AUGMENT_FACTOR > 1:
            train_raw = self.augment_data(train_raw)
        
        self.preprocessor.fit(train_raw)
        train = self.preprocessor.transform(train_raw)
        val = self.preprocessor.transform(val_raw)

        X_train, y_train, train_scores = self.preprocessor.prepare_sequences(train)
        y_train_class, y_train_period = y_train
        
        X_val, y_val, val_scores = self.preprocessor.prepare_sequences(val)
        y_val_class, y_val_period = y_val

        # Entraînement LSTM
        lstm_model = LSTMModel(
            input_shape=(DataParams.WINDOW_SIZE, len(self.preprocessor.feature_columns))
        )
        lstm_model.train(
            X_train, 
            (y_train_class, y_train_period, train_scores),
            X_val, 
            (y_val_class, y_val_period, val_scores)
        )

        # Extraction de features hybrides
        X_train_feat = self._extract_hybrid_features(lstm_model.model, X_train, train)
        X_val_feat = self._extract_hybrid_features(lstm_model.model, X_val, val)

        # Entraînement XGBoost Classifier (ScoreClass)
        xgb_clf = XGBoostModel(n_classes=3)
        xgb_clf.train(
            np.vstack([X_train_feat, X_val_feat]),
            np.concatenate([y_train_class, y_val_class]),
            eval_set=[(X_val_feat, y_val_class)]
        )
        
        # Entraînement XGBoost Classifier (Period)
        xgb_period = XGBoostModel(n_classes=1, objective='binary:logistic')
        xgb_period.train(
            np.vstack([X_train_feat, X_val_feat]),
            np.concatenate([y_train_period, y_val_period]),
            eval_set=[(X_val_feat, y_val_period)]
        )
        
        # Entraînement XGBoost Regressor (Score)
        xgb_reg = XGBRegressorModel()
        xgb_reg.train(
            np.vstack([X_train_feat, X_val_feat]),
            np.concatenate([train_scores, val_scores]),
            eval_set=[(X_val_feat, val_scores)]
        )

        # Sauvegarde des modèles
        self.save_models(xgb_clf, xgb_period, xgb_reg, suffix=f"_fold_{self.fold}")
        self.preprocessor.save_artifacts(suffix=f"_fold_{self.fold}")
        
        return lstm_model, xgb_clf, xgb_period, xgb_reg

    def augment_data(self, df):
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
        class_proba, period_proba, score_pred = lstm_model.predict(sequences)
        temporal_features = []
        for i in range(DataParams.WINDOW_SIZE, len(df)):
            temporal_features.append(
                self.feature_builder.build_features(df, i)
            )
        return np.hstack([class_proba, period_proba, score_pred, np.array(temporal_features)])

    def save_models(self, xgb_clf, xgb_period, xgb_reg, suffix=""):
        xgb_clf.save(Paths.XGB_MODEL.with_stem(Paths.XGB_MODEL.stem + suffix))
        xgb_period.save(Paths.XGB_PERIOD.with_stem(Paths.XGB_PERIOD.stem + suffix))
        xgb_reg.save(Paths.XGB_REGRESSOR.with_stem(Paths.XGB_REGRESSOR.stem + suffix))
        joblib.dump({
            'window_size': DataParams.WINDOW_SIZE,
            'min_confidence': TrainingParams.MIN_CONFIDENCE,
            'fold': self.fold
        }, Paths.MODEL_DIR / f'config{suffix}.joblib')