import argparse
from pathlib import Path
import pandas as pd
from data.database import CasinoDatabase
from data.preprocessor import DataPreprocessor
from training.trainer import CasinoTrainer
from training.evaluator import CasinoEvaluator
from config.paths import Paths
import joblib
from models.xgboost import XGBRegressorModel  # Nouveau

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run full training pipeline')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate on test set')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialisation
    Paths.MODEL_DIR.mkdir(exist_ok=True)
    Paths.REPORT_DIR.mkdir(exist_ok=True)
    
    # Chargement des données
    db = CasinoDatabase()
    raw_data = db.fetch_data()
    
    if args.train:
        # Entraînement complet
        trainer = CasinoTrainer()
        lstm, xgb_clf, xgb_reg = trainer.train(raw_data)
    
    if args.evaluate:
        # Évaluation rigoureuse
        preprocessor = joblib.load(Paths.PREPROCESSOR)
        _, _, test_raw = preprocessor.split_data(raw_data)
        test = preprocessor.transform(test_raw)
        
        # Chargement des modèles
        lstm = LSTMModel((preprocessor.window_size, 3)).load()
        xgb_clf = XGBoostModel.load(Paths.XGB_MODEL, len(preprocessor.encoder.classes_))
        xgb_reg = XGBRegressorModel.load(Paths.XGB_REGRESSOR)
        
        # Préparation des données
        X_test, y_test, test_scores = preprocessor.prepare_sequences(test)
        
        # Extraction des features
        trainer = CasinoTrainer()
        X_test_features = trainer._extract_hybrid_features(lstm.model, X_test, test)
        
        # Prédiction et évaluation
        evaluator = CasinoEvaluator(preprocessor.encoder)
        
        # Évaluation classificateur
        y_pred = xgb_clf.model.predict(X_test_features)
        y_proba = xgb_clf.model.predict_proba(X_test_features)
        
        # Évaluation régresseur
        score_pred = xgb_reg.model.predict(X_test_features)
        
        report = evaluator.evaluate(
            y_test, test_scores, 
            y_pred, score_pred, 
            y_proba
        )
        print(f"✅ Evaluation completed with ROC AUC: {report['roc_auc']:.3f}")
        print(f"✅ Score prediction MAE: {report['score_mae']:.2f}")

if __name__ == "__main__":
    main()