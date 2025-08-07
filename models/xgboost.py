# xgboost.py
import xgboost as xgb
from config.paths import Paths
import numpy as np
import joblib
from config.params import XGBoostParams

class XGBoostModel:
    """Modèle XGBoost pour la classification"""
    def __init__(self, n_classes, objective=None):
        params = XGBoostParams.BASE_PARAMS.copy()
        if n_classes > 1:
            params['objective'] = objective or 'multi:softprob'
            params['num_class'] = n_classes
        else:
            params['objective'] = objective or 'binary:logistic'
        
        params['early_stopping_rounds'] = XGBoostParams.EARLY_STOPPING
        params.setdefault('tree_method', 'gpu_hist')
        self.model = xgb.XGBClassifier(**params)

    def train(self, X, y, eval_set=None):
        try:
            if eval_set:
                self.model.fit(
                    X, y,
                    eval_set=eval_set,
                    verbose=10
                )
            else:
                self.model.fit(X, y)
        except xgb.core.XGBoostError:
            self.model.set_params(tree_method='hist')
            if eval_set:
                self.model.fit(
                    X, y,
                    eval_set=eval_set,
                    verbose=10
                )
            else:
                self.model.fit(X, y)

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
        params.setdefault('tree_method', 'gpu_hist')
        self.model = xgb.XGBRegressor(**params)

    def train(self, X, y, eval_set=None):
        try:
            if eval_set:
                self.model.fit(
                    X, y,
                    eval_set=eval_set,
                    verbose=10
                )
            else:
                self.model.fit(X, y)
        except xgb.core.XGBoostError:
            self.model.set_params(tree_method='hist')
            if eval_set:
                self.model.fit(
                    X, y,
                    eval_set=eval_set,
                    verbose=10
                )
            else:
                self.model.fit(X, y)

    def save(self, path):
        self.model.save_model(path)

    @classmethod
    def load(cls, path):
        model = cls()
        model.model.load_model(path)
        return model