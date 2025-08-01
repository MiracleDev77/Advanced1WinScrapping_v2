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
        
    def evaluate(self, y_true, true_scores, y_pred, pred_scores, y_proba):
        # Évaluation classification
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(
            y_true, y_proba, 
            multi_class='ovo',
            average='weighted'
        )
        
        # Évaluation prédiction de score
        mae = mean_absolute_error(true_scores, pred_scores)
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        class_names = self.encoder.classes_
        
        # Sauvegarde du rapport
        report = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'score_mae': mae,
            'confusion_matrix': {
                'labels': class_names.tolist(),
                'matrix': cm.tolist()
            }
        }
        
        # Sauvegarde dans un fichier
        with open(Paths.REPORT_DIR / 'evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        return report