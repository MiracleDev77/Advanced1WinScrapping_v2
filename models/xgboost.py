import xgboost as xgb
from config.paths import Paths
import numpy as np
import joblib

class XGBoostModel:
    """Modèle XGBoost pour la classification"""
    def __init__(self, n_classes):
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=n_classes,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def save(self, path):
        self.model.save_model(path)
        
    @classmethod
    def load(cls, path, n_classes):
        model = cls(n_classes)
        model.model.load_model(path)
        return model

class XGBRegressorModel:
    """Modèle XGBoost pour la régression (prédiction de score)"""
    def __init__(self):
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse'
        )
    
    def train(self, X, y):
        self.model.fit(X, y)
    
    def save(self, path):
        self.model.save_model(path)
        
    @classmethod
    def load(cls, path):
        model = cls()
        model.model.load_model(path)
        return model