import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from config.paths import Paths
from config.params import DataParams

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.feature_columns = ['Score',
            'MoyenneMobileDixDernier',
            'Ecart_Type',
            'prev_score',
            'prev_moyenne',
            'score_diff',
            'moyenne_diff',
            'type_repetition']  # Exemple de features
        self.window_size = DataParams.WINDOW_SIZE
        self.fold = 0  # Pour la validation croisée

    def split_data(self, df):
        """Split temporel des données"""
        total_size = len(df)
        train_size = int(total_size * (1 - DataParams.VAL_SIZE - DataParams.TEST_SIZE))
        val_size = int(total_size * DataParams.VAL_SIZE)
        
        train = df.iloc[:train_size]
        val = df.iloc[train_size:train_size+val_size]
        test = df.iloc[train_size+val_size:]
        
        return train, val, test

    def fit(self, df):
        # Encodage des labels
        self.encoder.fit(df['Type'])
        
        # Identifier les colonnes numériques parmi les feature_columns
        numeric_cols = [col for col in self.feature_columns 
                        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        # Normalisation des features numériques
        if numeric_cols:
            self.scaler.fit(df[numeric_cols])

    def transform(self, df):
        df = df.copy()
        # Encodage de la cible
        df['Type_encoded'] = self.encoder.transform(df['Type'])
        
        # Identifier les colonnes numériques parmi les feature_columns
        numeric_cols = [col for col in self.feature_columns 
                        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        
        # Normalisation des features numériques
        if numeric_cols:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        
        return df

    def prepare_sequences(self, df):
        """Prépare les séquences pour le LSTM"""
        X, y, scores = [], [], []
        
        for i in range(self.window_size, len(df)):
            # Séquence des features
            sequence = df[self.feature_columns].iloc[i-self.window_size:i].values
            # Cible: type de jeu
            target = df['Type_encoded'].iloc[i]
            # Cible: score
            score = df['Score'].iloc[i]
            
            X.append(sequence)
            y.append(target)
            scores.append(score)
            
        return np.array(X), np.array(y), np.array(scores)

    def save_artifacts(self, suffix=""):
        """Sauvegarde les objets de prétraitement"""
        joblib.dump(self.scaler, Paths.SCALER.with_stem(Paths.SCALER.stem + suffix))
        joblib.dump(self.encoder, Paths.ENCODER.with_stem(Paths.ENCODER.stem + suffix))
        joblib.dump(self, Paths.PREPROCESSOR.with_stem(Paths.PREPROCESSOR.stem + suffix))