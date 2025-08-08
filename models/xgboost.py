import xgboost as xgb
from config.paths import Paths
import numpy as np
import joblib
from config.params import XGBoostParams

class XGBoostModel:
    """Modèle XGBoost pour la classification"""
    def __init__(self, n_classes, objective=None):
        params = XGBoostParams.BASE_PARAMS.copy()
        
        # Gestion spéciale pour les problèmes binaires
        if n_classes == 1:
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'
        elif n_classes > 1:
            params['objective'] = objective or 'multi:softprob'
            params['num_class'] = n_classes
        
        params['early_stopping_rounds'] = XGBoostParams.EARLY_STOPPING
        self.model = xgb.XGBClassifier(**params)

    def train(self, X, y, eval_set=None):
        # Conversion des étiquettes en entiers si nécessaire
        if y.dtype not in [np.int32, np.int64]:
            y = y.astype(np.int32)
        
        # Pour les problèmes binaires, vérifier que les labels sont 0 ou 1
        if self.model.objective == 'binary:logistic':
            unique_labels = np.unique(y)
            if not set(unique_labels).issubset({0, 1}):
                # Convertir en binaire si nécessaire
                y = np.where(y > 0, 1, 0)
        
        # Conversion des étiquettes de validation
        eval_set_processed = None
        if eval_set:
            X_val, y_val = eval_set[0]
            if y_val.dtype not in [np.int32, np.int64]:
                y_val = y_val.astype(np.int32)
            
            if self.model.objective == 'binary:logistic':
                y_val = np.where(y_val > 0, 1, 0)
            
            eval_set_processed = [(X_val, y_val)]
        
        try:
            self.model.fit(
                X, y,
                eval_set=eval_set_processed,
                verbose=10
            )
        except xgb.core.XGBoostError:
            # Fallback au CPU si GPU échoue
            self.model.set_params(tree_method='hist')
            self.model.fit(
                X, y,
                eval_set=eval_set_processed,
                verbose=10
            )

    def save(self, path):
        self.model.save_model(path)

    @classmethod
    def load(cls, path, n_classes, objective=None):
        model = cls(n_classes, objective)
        model.model.load_model(path)
        return model

class XGBRegressorModel:
    """Modèle XGBoost pour la régression"""
    def __init__(self):
        params = XGBoostParams.REGRESSOR_PARAMS.copy()
        params['early_stopping_rounds'] = XGBoostParams.EARLY_STOPPING
        self.model = xgb.XGBRegressor(**params)

    def train(self, X, y, eval_set=None):
        eval_set_processed = None
        if eval_set:
            X_val, y_val = eval_set[0]
            eval_set_processed = [(X_val, y_val)]
        
        try:
            self.model.fit(
                X, y,
                eval_set=eval_set_processed,
                verbose=10
            )
        except xgb.core.XGBoostError:
            self.model.set_params(tree_method='hist')
            self.model.fit(
                X, y,
                eval_set=eval_set_processed,
                verbose=10
            )

    def save(self, path):
        self.model.save_model(path)

    @classmethod
    def load(cls, path):
        model = cls()
        model.model.load_model(path)
        return model