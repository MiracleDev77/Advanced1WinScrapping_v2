import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    confusion_matrix, mean_absolute_error
)
import json
from pathlib import Path
from config.paths import Paths

class CasinoEvaluator:
    def __init__(self, encoder):
        self.encoder = encoder
        self.total_classes = len(encoder.classes_)

    def evaluate(self, y_true, true_scores, y_pred, pred_scores, y_proba):
        # Évaluation classification
        accuracy = accuracy_score(y_true, y_pred)

        # Définir les classes présentes dans y_true
        present_classes = np.unique(y_true)

        # Calculer le ROC AUC sur toutes les probabilités,
        # en restreignant l'évaluation aux classes présentes
        roc_auc = roc_auc_score(
            y_true,
            y_proba,
            multi_class='ovo',
            average='weighted',
            labels=present_classes
        )

        # Évaluation prédiction de score
        mae = mean_absolute_error(true_scores, pred_scores)

        # Matrice de confusion sur l'ensemble des classes
        cm = confusion_matrix(y_true, y_pred, labels=range(self.total_classes))
        class_names = self.encoder.classes_

        report = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'score_mae': mae,
            'confusion_matrix': {
                'labels': class_names.tolist(),
                'matrix': cm.tolist()
            }
        }

        Paths.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        with open(Paths.REPORT_DIR / 'evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        return report
