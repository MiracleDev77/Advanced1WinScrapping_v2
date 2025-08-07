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

    if 'ScoreType' in raw_data.columns:
        raw_data.rename(columns={'ScoreType': 'Type'}, inplace=True)
    elif 'Type' not in raw_data.columns:
        # Handle cases where the column might be missing entirely
        raise KeyError("Expected 'ScoreType' or 'Type' column in the database data, but neither was found.")

    
    if args.train:
        # Validation croisée
        for fold in range(TrainingParams.NUM_FOLDS):
            print(f"⏳ Training fold {fold+1}/{TrainingParams.NUM_FOLDS}")
            trainer = CasinoTrainer(fold=fold)
            trainer.train(raw_data)

    if args.evaluate:
        # Charger le meilleur fold (ou spécifié)
        fold = args.fold if args.fold < TrainingParams.NUM_FOLDS else 0
        
        preprocessor = joblib.load(Paths.PREPROCESSOR.with_stem(f"{Paths.PREPROCESSOR.stem}_fold_{fold}"))
        _, _, test_raw = preprocessor.split_data(raw_data)
        test = preprocessor.transform(test_raw)

        # Chargement LSTM avec n_classes
        n_classes = len(preprocessor.encoder.classes_)
        lstm = LSTMModel(
            input_shape=(preprocessor.window_size, len(preprocessor.feature_columns)),
            n_classes=n_classes
        ).load()

        # Chargement des modèles XGBoost
        xgb_clf = XGBoostModel.load(
            Paths.XGB_MODEL.with_stem(f"{Paths.XGB_MODEL.stem}_fold_{fold}"), 
            n_classes
        )
        xgb_reg = XGBRegressorModel.load(
            Paths.XGB_REGRESSOR.with_stem(f"{Paths.XGB_REGRESSOR.stem}_fold_{fold}")
        )

        # Préparation des données de test
        X_test, y_test, test_scores = preprocessor.prepare_sequences(test)

        # Extraction des features hybrides
        trainer = CasinoTrainer(fold=fold)
        X_test_features = trainer._extract_hybrid_features(lstm.model, X_test, test)

        # Évaluation
        evaluator = CasinoEvaluator(preprocessor.encoder)
        evaluator.fold = fold  # Pour le rapport
        
        y_pred = xgb_clf.model.predict(X_test_features)
        y_proba = xgb_clf.model.predict_proba(X_test_features)
        score_pred = xgb_reg.model.predict(X_test_features)

        report = evaluator.evaluate(
            y_test, test_scores,
            y_pred, score_pred,
            y_proba
        )
        
        print(f"\n📊 FOLD {fold} RESULTS:")
        print(f"✅ ROC AUC: {report['roc_auc']:.3f}")
        print(f"✅ F1 Score: {report['f1_score']:.3f}")
        print(f"✅ Score MAE: {report['score_mae']:.2f}")
        print(f"✅ Accuracy: {report['accuracy']:.2f}")

if __name__ == '__main__':
    main()