import numpy as np
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from data.preprocessor import DataPreprocessor
from models.lstm_attention import LSTMModel
from models.xgboost import XGBoostModel, XGBRegressorModel
from config.paths import Paths
from config.params import DataParams, TrainingParams, LSTMParams
import joblib

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

    def extract_hybrid_features(self, lstm_model, sequences, df):
        """Extrait les caractéristiques hybrides pour XGBoost"""
        if len(sequences) == 0:
            return np.array([])
            
        predictions = lstm_model.predict(sequences, verbose=0)
        class_proba = predictions[0]
        period_proba = predictions[1]
        score_pred = predictions[2]
        
        n_sequences = len(sequences)
        temporal_features = []
        
        for i in range(self.window_size, self.window_size + n_sequences):
            if i < len(df):
                temporal_features.append(self.build_features(df, i))
            else:
                temporal_features.append(self.build_features(df, len(df)-1))
        
        temporal_arr = np.array(temporal_features)
        
        # Ajustement des dimensions
        if len(class_proba) != len(temporal_arr):
            min_length = min(len(class_proba), len(temporal_arr))
            class_proba = class_proba[:min_length]
            period_proba = period_proba[:min_length]
            score_pred = score_pred[:min_length]
            temporal_arr = temporal_arr[:min_length]
        
        return np.hstack([class_proba, period_proba, score_pred, temporal_arr])

class CasinoTrainer:
    def __init__(self, fold=0):
        self.preprocessor = DataPreprocessor()
        self.feature_builder = TemporalFeatureBuilder(DataParams.WINDOW_SIZE)
        self.fold = fold

    def train(self, raw_df):
        # Rééquilibrage des classes
        balanced_df = self.preprocessor.balance_classes(raw_df)
        
        # Split des données
        train_raw, val_raw, test_raw = self.preprocessor.split_data(balanced_df)
        
        # Augmentation générale
        if DataParams.AUGMENT_FACTOR > 1:
            train_raw = self.augment_data(train_raw)
        
        # Préprocessing
        self.preprocessor.fit(train_raw)
        train = self.preprocessor.transform(train_raw)
        val = self.preprocessor.transform(val_raw)

        # Préparation des séquences
        X_train, y_train, train_scores = self.preprocessor.prepare_sequences(train)
        y_train_class, y_train_period = y_train
        
        X_val, y_val, val_scores = self.preprocessor.prepare_sequences(val)
        y_val_class, y_val_period = y_val

        # Vérification des dimensions
        if len(X_train) == 0 or len(y_train_class) == 0 or len(y_train_period) == 0 or len(train_scores) == 0:
            print("⚠️ Aucune donnée d'entraînement disponible. Arrêt de l'entraînement.")
            return None, None, None, None

        # Conversion des types
        X_train = X_train.astype('float32')
        X_val = X_val.astype('float32')
        y_train_class = y_train_class.astype('int32')
        y_val_class = y_val_class.astype('int32')
        y_train_period = y_train_period.astype('int32')
        y_val_period = y_val_period.astype('int32')
        train_scores = train_scores.astype('float32')
        val_scores = val_scores.astype('float32')

        # SMOTE amélioré
        if DataParams.USE_SMOTE and len(np.unique(y_train_class)) > 1:
            n_samples, n_timesteps, n_features = X_train.shape
            X_train_flat = X_train.reshape(n_samples, n_timesteps * n_features)
            
            sm = SMOTE(sampling_strategy='auto', random_state=42)
            X_train_res, y_train_class_res = sm.fit_resample(X_train_flat, y_train_class)
            
            X_train = X_train_res.reshape(-1, n_timesteps, n_features)
            y_train_class = y_train_class_res
            
            # Réalignement des autres cibles
            y_train_period = np.tile(y_train_period, (len(X_train) // len(y_train_period) + 1))[:len(X_train)]
            train_scores = np.tile(train_scores, (len(X_train) // len(train_scores) + 1))[:len(X_train)]
        
        # Entraînement LSTM avec gestion des classes
        lstm_model = LSTMModel(
            input_shape=(DataParams.WINDOW_SIZE, len(self.preprocessor.feature_columns))
        )
        
        # Création des datasets optimisés
        train_dataset = self.create_dataset(X_train, y_train_class, y_train_period, train_scores)
        val_dataset = self.create_dataset(X_val, y_val_class, y_val_period, val_scores)
        
        # Entraînement
        history = lstm_model.train(train_dataset, val_dataset)

        # Extraction de features hybrides
        X_train_feat = self.feature_builder.extract_hybrid_features(lstm_model.model, X_train, train)
        X_val_feat = self.feature_builder.extract_hybrid_features(lstm_model.model, X_val, val)

        # Entraînement XGBoost avec paramètres optimisés
        xgb_clf = XGBoostModel(n_classes=3)
        # CORRECTION: Utilisation uniquement des données d'entraînement
        xgb_clf.train(
            X_train_feat,
            y_train_class,
            eval_set=[(X_val_feat, y_val_class)]
        )
        
        # Entraînement XGBoost Classifier (Period)
        y_train_period_int = np.where(y_train_period > 0, 1, 0)
        y_val_period_int = np.where(y_val_period > 0, 1, 0)
        
        xgb_period = XGBoostModel(n_classes=1, objective='binary:logistic')
        xgb_period.train(
            X_train_feat,
            y_train_period_int,
            eval_set=[(X_val_feat, y_val_period_int)]
        )
        
        # Entraînement XGBoost Regressor (Score)
        xgb_reg = XGBRegressorModel()
        xgb_reg.train(
            X_train_feat,
            train_scores,
            eval_set=[(X_val_feat, val_scores)]
        )

        # Sauvegarde des modèles
        self.save_models(xgb_clf, xgb_period, xgb_reg, suffix=f"_fold_{self.fold}")
        self.preprocessor.save_artifacts(suffix=f"_fold_{self.fold}")
        
        return lstm_model, xgb_clf, xgb_period, xgb_reg

    def augment_data(self, df):
        """Augmentation ciblée des classes minoritaires"""
        # Calculer la distribution des classes
        class_counts = df['ScoreClass'].value_counts()
        majority_count = class_counts.max()
        
        augmented_dfs = [df]
        
        for cls, count in class_counts.items():
            if count < majority_count:
                # Facteur d'augmentation spécifique
                factor = DataParams.AUGMENT_MINORITY_FACTOR
                cls_df = df[df['ScoreClass'] == cls]
                
                for _ in range(factor - 1):
                    aug_df = cls_df.copy()
                    numeric_cols = aug_df.select_dtypes(include=np.number).columns
                    noise = np.random.normal(0, DataParams.AUGMENT_NOISE, (len(cls_df), len(numeric_cols)))
                    aug_df[numeric_cols] += noise
                    augmented_dfs.append(aug_df)
        
        return pd.concat(augmented_dfs, ignore_index=True)
    
    def create_dataset(self, X, y_class, y_period, scores):
        """Crée un dataset optimisé pour TensorFlow"""
        dataset = tf.data.Dataset.from_tensor_slices((
            X,
            {
                'score_class': y_class,
                'period': y_period,
                'score_reg': scores
            }
        ))
        return dataset.batch(LSTMParams.BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

    def save_models(self, xgb_clf, xgb_period, xgb_reg, suffix=""):
        xgb_clf.save(str(Paths.XGB_MODEL.with_stem(Paths.XGB_MODEL.stem + suffix)))
        xgb_period.save(str(Paths.XGB_PERIOD.with_stem(Paths.XGB_PERIOD.stem + suffix)))
        xgb_reg.save(str(Paths.XGB_REGRESSOR.with_stem(Paths.XGB_REGRESSOR.stem + suffix)))
        joblib.dump({
            'window_size': DataParams.WINDOW_SIZE,
            'min_confidence': TrainingParams.MIN_CONFIDENCE,
            'fold': self.fold
        }, str(Paths.MODEL_DIR / f'config{suffix}.joblib'))