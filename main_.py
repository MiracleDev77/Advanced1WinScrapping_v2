# main.py
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
from models.xgboost import XGBoostModel, XGBRegressorModel
from config.params import TrainingParams
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--fold', type=int, default=0, help='Fold number for cross-validation')
    return parser.parse_args()

def main():
    args = parse_args()
    Paths.MODEL_DIR.mkdir(exist_ok=True)
    Paths.REPORT_DIR.mkdir(exist_ok=True)

    db = CasinoDatabase()
    raw_data = db.fetch_data()

    if args.train:
        for fold in range(TrainingParams.NUM_FOLDS):
            print(f"⏳ Training fold {fold+1}/{TrainingParams.NUM_FOLDS}")
            trainer = CasinoTrainer(fold=fold)
            trainer.train(raw_data)

    if args.evaluate:
        fold = args.fold if args.fold < TrainingParams.NUM_FOLDS else 0
        
        preprocessor = joblib.load(Paths.PREPROCESSOR.with_stem(f"{Paths.PREPROCESSOR.stem}_fold_{fold}"))
        _, _, test_raw = preprocessor.split_data(raw_data)
        test = preprocessor.transform(test_raw)

        X_test, y_test, test_scores = preprocessor.prepare_sequences(test)
        y_test_class, y_test_period = y_test

        lstm = LSTMModel(
            input_shape=(preprocessor.window_size, len(preprocessor.feature_columns))
        ).load()

        xgb_clf = XGBoostModel.load(
            Paths.XGB_MODEL.with_stem(f"{Paths.XGB_MODEL.stem}_fold_{fold}"), 
            n_classes=3
        )
        xgb_period = XGBoostModel.load(
            Paths.XGB_PERIOD.with_stem(f"{Paths.XGB_PERIOD.stem}_fold_{fold}"), 
            n_classes=1,
            objective='binary:logistic'
        )
        xgb_reg = XGBRegressorModel.load(
            Paths.XGB_REGRESSOR.with_stem(f"{Paths.XGB_REGRESSOR.stem}_fold_{fold}")
        )

        trainer = CasinoTrainer(fold=fold)
        X_test_features = trainer._extract_hybrid_features(lstm.model, X_test, test)

        evaluator = CasinoEvaluator()
        evaluator.fold = fold
        
        # Prédictions
        y_pred_class = xgb_clf.model.predict(X_test_features)
        y_proba_class = xgb_clf.model.predict_proba(X_test_features)
        y_pred_period = xgb_period.model.predict(X_test_features)
        score_pred = xgb_reg.model.predict(X_test_features)

        report = evaluator.evaluate(
            y_test_class, y_test_period, test_scores,
            y_pred_class, y_pred_period, score_pred,
            y_proba_class
        )
        
        print(f"\n📊 FOLD {fold} RESULTS:")
        print("✅ Score Classification:")
        print(f"   Accuracy: {report['score_class']['accuracy']:.3f}")
        print(f"   F1 Score: {report['score_class']['f1_score']:.3f}")
        print(f"   ROC AUC: {report['score_class']['roc_auc']:.3f}")
        
        print("\n✅ Period Prediction:")
        print(f"   Accuracy: {report['period']['accuracy']:.3f}")
        print(f"   F1 Score: {report['period']['f1_score']:.3f}")
        print(f"   ROC AUC: {report['period']['roc_auc']:.3f}")
        
        print("\n✅ Score Regression:")
        print(f"   MAE: {report['score_reg']['mae']:.2f}")
        print(f"   RMSE: {report['score_reg']['rmse']:.2f}")

if __name__ == '__main__':
    main()