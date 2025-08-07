# evaluator.py
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, f1_score, precision_score,
    recall_score, classification_report, mean_squared_error
)
from sklearn.preprocessing import label_binarize
import json
from pathlib import Path
from config.paths import Paths

class CasinoEvaluator:
    def __init__(self):
        self.total_classes = 3
        self.period_threshold = 0.5

    def evaluate(self, y_true_class, y_true_period, true_scores,
                 y_pred_class, y_pred_period, pred_scores, y_proba_class):
        # Classification metrics (ScoreClass)
        accuracy_class = accuracy_score(y_true_class, y_pred_class)
        f1_class = f1_score(y_true_class, y_pred_class, average='weighted', zero_division=0)
        roc_auc_class = roc_auc_score(
            label_binarize(y_true_class, classes=[0,1,2]), 
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
                'f1_score': f1_class,
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
        report_path = Paths.REPORT_DIR / f'evaluation_report_fold_{getattr(self, "fold", 0)}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        return report