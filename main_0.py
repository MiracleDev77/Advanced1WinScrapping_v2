import argparse
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from data.database import CasinoDatabase
from data.preprocessor import DataPreprocessor
from training.trainer import CasinoTrainer, TemporalFeatureBuilder
from training.evaluator import CasinoEvaluator
from config.paths import Paths
from config.params import TrainingParams, DataParams
from models.lstm_attention import LSTMModel
from models.xgboost import XGBoostModel, XGBRegressorModel

def main():
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Système de prédiction pour données de casino")
    parser.add_argument('--train', action='store_true', help='Lancer l\'entraînement complet (LSTM + XGBoost)')
    parser.add_argument('--evaluate', action='store_true', help='Évaluer les modèles entraînés')
    parser.add_argument('--predict', nargs='?', const='last', 
                        help='Faire des prédictions (optionnel: spécifier "last" pour les dernières données ou un nombre de jours)')
    parser.add_argument('--cross-validate', action='store_true', 
                        help='Lancer une validation croisée complète')
    args = parser.parse_args()

    # Activation des optimisations TensorFlow
    tf.config.optimizer.set_jit(True)  # Activer XLA
    print(f"XLA activé: {tf.config.optimizer.get_jit()}")
    
    # Vérification de la disponibilité GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU détecté: {len(gpus)} dispositif(s)")
    else:
        print("⚠️ Aucun GPU détecté - L'entraînement sera plus lent")

    # Initialisation de la base de données
    db = CasinoDatabase()
    
    # Mode entraînement
    if args.train or args.cross_validate:
        # Charger toutes les données historiques
        raw_data = db.fetch_data()
        print(f"📊 Données chargées: {len(raw_data)} enregistrements")
        
        # Validation croisée
        if args.cross_validate:
            print(f"\n🔁 Lancement de la validation croisée ({TrainingParams.NUM_FOLDS} folds)")
            for fold in range(TrainingParams.NUM_FOLDS):
                print(f"\n⏳ Entraînement fold {fold+1}/{TrainingParams.NUM_FOLDS}")
                trainer = CasinoTrainer(fold=fold)
                models = trainer.train(raw_data)
                
                # Évaluation immédiate après entraînement
                if models and models[0] is not None:
                    evaluate_fold(trainer, models, raw_data, fold)
        
        # Entraînement simple
        elif args.train:
            print("\n⏳ Lancement de l'entraînement sur le dataset complet")
            trainer = CasinoTrainer()
            models = trainer.train(raw_data)
            
            # Évaluation immédiate après entraînement
            if models and models[0] is not None:
                evaluate_fold(trainer, models, raw_data)

    # Mode évaluation
    if args.evaluate:
        print("\n🔍 Évaluation des modèles entraînés")
        evaluate_trained_models(db)

    # Mode prédiction
    if args.predict:
        print("\n🔮 Génération de prédictions")
        predict_future(db, args.predict)


def evaluate_fold(trainer, models, raw_data, fold=None):
    """Évalue les modèles d'un fold spécifique"""
    # Charger les données de validation
    _, val_raw, _ = trainer.preprocessor.split_data(raw_data)
    val = trainer.preprocessor.transform(val_raw)
    X_val, y_val, val_scores = trainer.preprocessor.prepare_sequences(val)
    y_val_class, y_val_period = y_val
    
    # Faire des prédictions avec LSTM
    lstm_pred = models[0].model.predict(X_val, verbose=0)
    y_pred_class_lstm = np.argmax(lstm_pred[0], axis=1)
    y_pred_period_lstm = lstm_pred[1].flatten()
    pred_scores_lstm = lstm_pred[2].flatten()
    
    # Faire des prédictions avec XGBoost
    feature_builder = TemporalFeatureBuilder(DataParams.WINDOW_SIZE)
    X_val_feat = feature_builder.extract_hybrid_features(models[0].model, X_val, val)
    
    # CORRECTION: Formatage correct des prédictions
    y_pred_class_xgb = models[1].model.predict(X_val_feat)
    y_pred_class_xgb = np.array(y_pred_class_xgb).ravel()  # S'assurer que c'est un vecteur 1D
    
    # CORRECTION: Application du seuil pour les prédictions périodiques
    y_pred_period_xgb = (models[2].model.predict(X_val_feat) > 0.5).astype(int)
    y_pred_period_xgb = np.array(y_pred_period_xgb).ravel()
    
    pred_scores_xgb = models[3].model.predict(X_val_feat)
    y_proba_class_xgb = models[1].model.predict_proba(X_val_feat)
    
    # CORRECTION: Troncature des prédictions à la taille de la vérité terrain
    min_length = min(len(y_val_class), len(y_pred_class_xgb))
    y_val_class = y_val_class[:min_length]
    y_val_period = y_val_period[:min_length]
    val_scores = val_scores[:min_length]
    y_pred_class_xgb = y_pred_class_xgb[:min_length]
    y_pred_period_xgb = y_pred_period_xgb[:min_length]
    pred_scores_xgb = pred_scores_xgb[:min_length]
    y_proba_class_xgb = y_proba_class_xgb[:min_length]
    
    # Créer l'évaluateur
    evaluator = CasinoEvaluator(fold=fold)
    
    # Évaluation LSTM
    print("\n📈 Évaluation LSTM:")
    report_lstm = evaluator.evaluate(
        y_val_class, y_val_period, val_scores,
        y_pred_class_lstm, y_pred_period_lstm, pred_scores_lstm, lstm_pred[0]
    )
    
    # Évaluation XGBoost
    print("\n📈 Évaluation XGBoost:")
    report_xgb = evaluator.evaluate(
        y_val_class, y_val_period, val_scores,
        y_pred_class_xgb, y_pred_period_xgb, pred_scores_xgb, 
        y_proba_class_xgb
    )
    
    # Évaluation de l'ensemble (moyenne)
    print("\n📈 Évaluation de l'ensemble (moyenne):")
    y_pred_class_ens = (y_pred_class_lstm + y_pred_class_xgb) // 2
    y_pred_period_ens = ((y_pred_period_lstm + y_pred_period_xgb) / 2 > 0.5).astype(int)
    pred_scores_ens = (pred_scores_lstm + pred_scores_xgb) / 2
    y_proba_class_ens = (lstm_pred[0] + y_proba_class_xgb) / 2
    
    # Troncature pour l'ensemble
    min_length_ens = min(len(y_val_class), len(y_pred_class_ens))
    y_val_class_ens = y_val_class[:min_length_ens]
    y_val_period_ens = y_val_period[:min_length_ens]
    val_scores_ens = val_scores[:min_length_ens]
    y_pred_class_ens = y_pred_class_ens[:min_length_ens]
    y_pred_period_ens = y_pred_period_ens[:min_length_ens]
    pred_scores_ens = pred_scores_ens[:min_length_ens]
    y_proba_class_ens = y_proba_class_ens[:min_length_ens]
    
    report_ens = evaluator.evaluate(
        y_val_class_ens, y_val_period_ens, val_scores_ens,
        y_pred_class_ens, y_pred_period_ens, pred_scores_ens, 
        y_proba_class_ens
    )


def evaluate_trained_models(db):
    """Évalue les modèles pré-entraînés sur les données récentes"""
    try:
        # Charger le préprocesseur
        preprocessor = joblib.load(Paths.PREPROCESSOR)
        
        # Charger les dernières données
        test_data = db.fetch_data(last_n=365)  # Dernière année
        test_data = preprocessor.transform(test_data)
        
        # Préparer les séquences
        X_test, y_test, test_scores = preprocessor.prepare_sequences(test_data)
        if len(X_test) == 0:
            print("⚠️ Aucune donnée de test disponible")
            return
            
        y_test_class, y_test_period = y_test
        
        # Charger les modèles
        lstm_model = LSTMModel((DataParams.WINDOW_SIZE, len(preprocessor.feature_columns))).load()
        xgb_clf = XGBoostModel.load(str(Paths.XGB_MODEL), n_classes=3)
        xgb_period = XGBoostModel.load(str(Paths.XGB_PERIOD), n_classes=1, objective='binary:logistic')
        xgb_reg = XGBRegressorModel.load(str(Paths.XGB_REGRESSOR))
        
        # Créer les features hybrides
        feature_builder = TemporalFeatureBuilder(DataParams.WINDOW_SIZE)
        X_test_feat = feature_builder.extract_hybrid_features(lstm_model.model, X_test, test_data)
        
        # Faire des prédictions
        lstm_pred = lstm_model.model.predict(X_test, verbose=0)
        y_pred_class_lstm = np.argmax(lstm_pred[0], axis=1)
        y_pred_period_lstm = lstm_pred[1].flatten()
        pred_scores_lstm = lstm_pred[2].flatten()
        
        # CORRECTION: Formatage correct des prédictions
        y_pred_class_xgb = xgb_clf.model.predict(X_test_feat)
        y_pred_class_xgb = np.array(y_pred_class_xgb).ravel()  # S'assurer que c'est un vecteur 1D
        
        y_pred_period_xgb = xgb_period.model.predict(X_test_feat)
        y_pred_period_xgb = np.array(y_pred_period_xgb).ravel()  # S'assurer que c'est un vecteur 1D
        
        pred_scores_xgb = xgb_reg.model.predict(X_test_feat)
        y_proba_class_xgb = xgb_clf.model.predict_proba(X_test_feat)
        
        # Évaluation
        evaluator = CasinoEvaluator()
        
        print("\n🔍 Évaluation LSTM sur données récentes:")
        evaluator.evaluate(
            y_test_class, y_test_period, test_scores,
            y_pred_class_lstm, y_pred_period_lstm, pred_scores_lstm, lstm_pred[0]
        )
        
        print("\n🔍 Évaluation XGBoost sur données récentes:")
        evaluator.evaluate(
            y_test_class, y_test_period, test_scores,
            y_pred_class_xgb, y_pred_period_xgb, pred_scores_xgb, 
            y_proba_class_xgb
        )
    
    except Exception as e:
        print(f"❌ Erreur lors de l'évaluation: {e}")
        import traceback
        traceback.print_exc()
        print("Assurez-vous d'avoir d'abord entraîné les modèles avec --train")

def predict_future(db, period="last"):
    """Génère des prédictions pour les données futures"""
    try:
        # Charger les artefacts
        preprocessor = joblib.load(Paths.PREPROCESSOR)
        lstm_model = LSTMModel((DataParams.WINDOW_SIZE, len(preprocessor.feature_columns))).load()
        xgb_clf = XGBoostModel.load(str(Paths.XGB_MODEL), n_classes=3)
        xgb_period = XGBoostModel.load(str(Paths.XGB_PERIOD), n_classes=1, objective='binary:logistic')
        xgb_reg = XGBRegressorModel.load(str(Paths.XGB_REGRESSOR))
        
        # Déterminer les données à prédire
        if period == "last":
            # Dernières données disponibles
            prediction_data = db.fetch_data(last_n=DataParams.WINDOW_SIZE + 1)
            print(f"🔮 Prédiction sur les {DataParams.WINDOW_SIZE + 1} derniers enregistrements")
        else:
            try:
                days = int(period)
                prediction_data = db.fetch_data(last_n=days)
                print(f"🔮 Prédiction sur les {days} derniers jours")
            except:
                prediction_data = db.fetch_data()
                print("🔮 Prédiction sur toutes les données disponibles")
        
        if len(prediction_data) < DataParams.WINDOW_SIZE + 1:
            print(f"⚠️ Données insuffisantes pour la prédiction (min: {DataParams.WINDOW_SIZE + 1} enregistrements)")
            return
        
        # Transformation des données
        prediction_data = preprocessor.transform(prediction_data)
        
        # Préparation des séquences
        X_pred, _, _ = preprocessor.prepare_sequences(prediction_data)
        
        if len(X_pred) == 0:
            print("⚠️ Aucune séquence valide pour la prédiction")
            return
        
        # Créer les features hybrides
        feature_builder = TemporalFeatureBuilder(DataParams.WINDOW_SIZE)
        X_pred_feat = feature_builder.extract_hybrid_features(lstm_model.model, X_pred, prediction_data)
        
        # Prédictions LSTM
        lstm_pred = lstm_model.model.predict(X_pred, verbose=0)
        class_proba_lstm = lstm_pred[0]
        class_pred_lstm = np.argmax(class_proba_lstm, axis=1)
        period_pred_lstm = (lstm_pred[1].flatten() > 0.5).astype(int)
        score_pred_lstm = lstm_pred[2].flatten()
        
        # Prédictions XGBoost
        class_pred_xgb = xgb_clf.model.predict(X_pred_feat)
        class_pred_xgb = np.array(class_pred_xgb).ravel()  # Format 1D
        
        class_proba_xgb = xgb_clf.model.predict_proba(X_pred_feat)
        period_pred_xgb = (xgb_period.model.predict(X_pred_feat) > 0.5).astype(int)
        score_pred_xgb = xgb_reg.model.predict(X_pred_feat)
        
        # Prédictions d'ensemble (moyenne)
        class_pred_ens = (class_pred_lstm + class_pred_xgb) // 2
        period_pred_ens = ((period_pred_lstm + period_pred_xgb) > 1).astype(int)
        score_pred_ens = (score_pred_lstm + score_pred_xgb) / 2
        
        # Création du DataFrame de résultats
        results = prediction_data.iloc[DataParams.WINDOW_SIZE:].copy()
        results['Pred_Class_LSTM'] = class_pred_lstm
        results['Pred_Class_XGB'] = class_pred_xgb
        results['Pred_Class_Ens'] = class_pred_ens
        results['Pred_Period_LSTM'] = period_pred_lstm
        results['Pred_Period_XGB'] = period_pred_xgb
        results['Pred_Period_Ens'] = period_pred_ens
        results['Pred_Score_LSTM'] = score_pred_lstm
        results['Pred_Score_XGB'] = score_pred_xgb
        results['Pred_Score_Ens'] = score_pred_ens
        
        # Sauvegarde des résultats
        prediction_file = Paths.PREDICTIONS / f'predictions_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results.to_csv(prediction_file, index=False)
        print(f"✅ Prédictions sauvegardées dans: {prediction_file}")
        
        # Affichage des dernières prédictions
        print("\n🔔 Dernières prédictions:")
        print(results[['Date', 'Heure', 'Score', 'Pred_Score_Ens', 
                      'ScoreClass', 'Pred_Class_Ens', 
                      'Period', 'Pred_Period_Ens']].tail())
    
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction: {e}")
        import traceback
        traceback.print_exc()
        print("Assurez-vous d'avoir d'abord entraîné les modèles avec --train")

if __name__ == "__main__":
    main()