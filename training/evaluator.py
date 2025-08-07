import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    mean_absolute_error, f1_score, precision_score,
    recall_score, classification_report
)
from sklearn.preprocessing import LabelBinarizer
import json
from pathlib import Path
from config.paths import Paths

class CasinoEvaluator:
    def __init__(self, encoder):
        self.encoder = encoder
        self.total_classes = len(encoder.classes_)
        self.lb = LabelBinarizer()
        self.lb.fit(range(self.total_classes))

    def evaluate(self, y_true, true_scores, y_pred, pred_scores, y_proba):
        # Classification metrics with zero_division handling
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        # ROC AUC
        y_true_bin = self.lb.transform(y_true)
        try:
            roc_auc = roc_auc_score(y_true_bin, y_proba,
                                     multi_class='ovr', average='weighted')
        except ValueError:
            roc_auc = float('nan')

        # Regression metric
        mae = mean_absolute_error(true_scores, pred_scores)

        # Confusion matrix
        labels = list(range(self.total_classes))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_normalized = cm.astype(float) / cm.sum(axis=1)[:, None]
        class_names = list(self.encoder.classes_)

        # Classification report with explicit labels
        clf_report = classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0
        )

        report = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
            'score_mae': mae,
            'confusion_matrix': {
                'labels': class_names,
                'matrix': cm.tolist(),
                'normalized': cm_normalized.tolist()
            },
            'classification_report': clf_report
        }

        Paths.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = Paths.REPORT_DIR / f'evaluation_report_fold_{getattr(self, "fold", 0)}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        return report
