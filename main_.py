import argparse
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from data.database import CasinoDatabase
from data.preprocessor import DataPreprocessor
from training.trainer import CasinoTrainer
from training.evaluator import CasinoEvaluator 
from config.paths import Paths
from models.lstm_attention import LSTMModel
from models.xgboost import XGBoostModel
from models.xgboost import XGBRegressorModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    Paths.MODEL_DIR.mkdir(exist_ok=True)
    Paths.REPORT_DIR.mkdir(exist_ok=True)

    db = CasinoDatabase()
    raw_data = db.fetch_data()
    
    if args.train:
        trainer = CasinoTrainer()
        trainer.train(raw_data)

    if args.evaluate:
        preprocessor = joblib.load(Paths.PREPROCESSOR)
        _, _, test_raw = preprocessor.split_data(raw_data)
        test = preprocessor.transform(test_raw)

        # Chargement LSTM avec n_classes
        n_classes = len(preprocessor.encoder.classes_)
        lstm = LSTMModel(
            input_shape=(preprocessor.window_size, len(preprocessor.feature_columns)),
            n_classes=n_classes
        ).load()

        xgb_clf = XGBoostModel.load(Paths.XGB_MODEL, len(preprocessor.encoder.classes_))
        xgb_reg = XGBRegressorModel.load(Paths.XGB_REGRESSOR)

        X_test, y_test, test_scores = preprocessor.prepare_sequences(test)

        trainer = CasinoTrainer()
        X_test_features = trainer._extract_hybrid_features(lstm.model, X_test, test)

        evaluator = CasinoEvaluator(preprocessor.encoder)
        y_pred = xgb_clf.model.predict(X_test_features)
        y_proba = xgb_clf.model.predict_proba(X_test_features)
        score_pred = xgb_reg.model.predict(X_test_features)

        report = evaluator.evaluate(
            y_test, test_scores,
            y_pred, score_pred,
            y_proba
        )
        print(f"✅ Evaluation ROC AUC: {report['roc_auc']:.3f}")
        print(f"✅ Score MAE: {report['score_mae']:.2f}")

if __name__ == '__main__':
    main()
