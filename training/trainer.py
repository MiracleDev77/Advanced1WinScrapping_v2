import numpy as np
from data.preprocessor import DataPreprocessor
from models.lstm_attention import LSTMModel
from models.xgboost import XGBoostModel, XGBRegressorModel
from config.paths import Paths
from config.params import DataParams
import joblib

class TemporalFeatureBuilder:
    def __init__(self, window_size):
        self.window_size = window_size

    def build_features(self, df, start_index):
        end_index = start_index - 1
        start_index = max(0, end_index - self.window_size + 1)
        window = df.iloc[start_index:end_index]
        if len(window) > 0:
            return np.array([
                window['Score'].mean(),
                window['Score'].std(),
                (window['Type_encoded'] == 0).sum()
            ])
        else:
            return np.zeros(3)

class CasinoTrainer:
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.feature_builder = TemporalFeatureBuilder(DataParams.WINDOW_SIZE)

    def train(self, raw_df):
        train_raw, val_raw, test_raw = self.preprocessor.split_data(raw_df)
        print(f"📊 train_raw: {len(train_raw)}, val_raw: {len(val_raw)}, test_raw: {len(test_raw)}")

        self.preprocessor.fit(train_raw)
        train = self.preprocessor.transform(train_raw)
        val = self.preprocessor.transform(val_raw)

        X_train, y_train, train_scores = self.preprocessor.prepare_sequences(train)
        X_val, y_val, val_scores = self.preprocessor.prepare_sequences(val)

        n_classes = len(np.unique(y_train))
        lstm_model = LSTMModel(
            input_shape=(DataParams.WINDOW_SIZE, len(self.preprocessor.feature_columns)),
            n_classes=n_classes
        )
        lstm_model.train(X_train, y_train, X_val, y_val)

        X_train_feat = self._extract_hybrid_features(lstm_model.model, X_train, train)
        X_val_feat   = self._extract_hybrid_features(lstm_model.model, X_val, val)

        xgb_clf = XGBoostModel(n_classes=n_classes)
        xgb_clf.train(
            np.vstack([X_train_feat, X_val_feat]),
            np.concatenate([y_train, y_val])
        )
        xgb_clf.save(Paths.XGB_MODEL)

        xgb_reg = XGBRegressorModel()
        xgb_reg.train(
            np.vstack([X_train_feat, X_val_feat]),
            np.concatenate([train_scores, val_scores])
        )
        xgb_reg.save(Paths.XGB_REGRESSOR)

        self.preprocessor.save_artifacts()
        joblib.dump({
            'window_size': DataParams.WINDOW_SIZE,
            'min_confidence': 0.99
        }, Paths.MODEL_DIR / 'config.joblib')

        return lstm_model, xgb_clf, xgb_reg

    def _extract_hybrid_features(self, lstm_model, sequences, df):
        lstm_features = lstm_model.predict(sequences)
        temporal_features = []
        for i in range(DataParams.WINDOW_SIZE, len(df)):
            temporal_features.append(
                self.feature_builder.build_features(df, i)
            )
        return np.hstack([lstm_features, np.array(temporal_features)])

