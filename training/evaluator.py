import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, f1_score, classification_report,
    balanced_accuracy_score, cohen_kappa_score, mean_squared_error
)
import json
from pathlib import Path
from config.paths import Paths
import tensorflow as tf

class CasinoEvaluator:
    def __init__(self, fold=0):
        self.period_threshold = 0.5
        self.fold = fold

    def evaluate(self, y_true_class, y_true_period, true_scores,
                 y_pred_class, y_pred_period, pred_scores, y_proba_class):
        # Vérification de la cohérence des tailles
        assert len(y_true_class) == len(y_pred_class), \
            f"Taille incohérente: y_true_class={len(y_true_class)} vs y_pred_class={len(y_pred_class)}"
        
        # Classification metrics (ScoreClass)
        accuracy_class = accuracy_score(y_true_class, y_pred_class)
        balanced_acc_class = balanced_accuracy_score(y_true_class, y_pred_class)
        f1_class = f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
        kappa_class = cohen_kappa_score(y_true_class, y_pred_class)
        
        # ROC AUC avec gestion des classes manquantes
        unique_classes = np.unique(y_true_class)
        if len(unique_classes) < 2:
            roc_auc_class = np.nan
            cm_class = np.array([[len(y_true_class)]])
            clf_report_class = {'warning': f'Only one class present: {unique_classes[0]}'}
        else:
            y_true_bin = tf.keras.utils.to_categorical(y_true_class, num_classes=3)
            roc_auc_class = roc_auc_score(
                y_true_bin, 
                y_proba_class, 
                multi_class='ovr',
                average='weighted'
            )
            cm_class = confusion_matrix(y_true_class, y_pred_class)
            clf_report_class = classification_report(
                y_true_class, y_pred_class,
                output_dict=True,
                zero_division=0
            )
        
        # Classification metrics (Period)
        y_pred_period_bin = (y_pred_period > self.period_threshold).astype(int)
        accuracy_period = accuracy_score(y_true_period, y_pred_period_bin)
        f1_period = f1_score(y_true_period, y_pred_period_bin, zero_division=0)
        
        # ROC AUC pour période avec gestion des classes manquantes
        unique_period = np.unique(y_true_period)
        if len(unique_period) < 2:
            roc_auc_period = np.nan
            cm_period = np.array([[len(y_true_period)]])
            clf_report_period = {'warning': f'Only one class present: {unique_period[0]}'}
        else:
            roc_auc_period = roc_auc_score(y_true_period, y_pred_period)
            cm_period = confusion_matrix(y_true_period, y_pred_period_bin)
            clf_report_period = classification_report(
                y_true_period, y_pred_period_bin,
                output_dict=True,
                zero_division=0
            )
        
        # Regression metrics
        mae_score = mean_absolute_error(true_scores, pred_scores)
        rmse_score = np.sqrt(mean_squared_error(true_scores, pred_scores))
        
        report = {
            'score_class': {
                'accuracy': accuracy_class,
                'balanced_accuracy': balanced_acc_class,
                'f1_score': f1_class,
                'kappa': kappa_class,
                'roc_auc': roc_auc_class,
                'confusion_matrix': cm_class.tolist(),
                'classification_report': clf_report_class
            },
            'period': {
                'accuracy': accuracy_period,
                'f1_score': f1_period,
                'roc_auc': roc_auc_period,
                'confusion_matrix': cm_period.tolist(),
                'classification_report': clf_report_period
            },
            'score_reg': {
                'mae': mae_score,
                'rmse': rmse_score
            }
        }

        Paths.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = Paths.REPORT_DIR / f'evaluation_report_fold_{self.fold}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # Affichage synthétique
        print(f"\n📊 FOLD {self.fold} RESULTS:")
        print(f"✅ Score Classification:")
        print(f"   Accuracy: {accuracy_class:.3f}")
        print(f"   Balanced Accuracy: {balanced_acc_class:.3f}")
        print(f"   F1 Score: {f1_class:.3f}")
        print(f"   ROC AUC: {roc_auc_class:.3f}")
        print(f"✅ Period Prediction:")
        print(f"   Accuracy: {accuracy_period:.3f}")
        print(f"   F1 Score: {f1_period:.3f}")
        print(f"   ROC AUC: {roc_auc_period:.3f}")
        print(f"✅ Score Regression:")
        print(f"   MAE: {mae_score:.2f}")
        print(f"   RMSE: {rmse_score:.2f}")

        return report