import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from config.paths import Paths
from config.params import DataParams
import joblib

class DataPreprocessor:
    def __init__(self, window_size=DataParams.WINDOW_SIZE):
        self.window_size = window_size
        self.scaler = RobustScaler()
        self.encoder = LabelEncoder()
        self.is_fitted = False
        # Définir les colonnes features par défaut
        self.feature_columns = ['Score', 'MoyenneMobileDixDernier', 'Ecart_Type']

    def fit(self, train_df):
        """Ajuste le préprocesseur sur les données d'entraînement uniquement avec gestion des cas limites"""
        if train_df.empty:
            raise ValueError("Le DataFrame d'entraînement est vide. Impossible d'ajuster le préprocesseur.")
        
        # Encodage des labels sur ScoreType
        self.encoder.fit(train_df['ScoreType'])
        
        # Vérifier la présence des colonnes nécessaires
        missing_cols = [col for col in self.feature_columns if col not in train_df.columns]
        if missing_cols:
            raise KeyError(f"Colonnes manquantes dans les données d'entraînement: {missing_cols}")
        
        # Gestion des petits datasets
        if len(train_df) < 2:
            print("Avertissement: Données d'entraînement insuffisantes. Utilisation de valeurs par défaut pour le scaler.")
            # Configuration manuelle du scaler
            self.scaler.center_ = np.zeros(len(self.feature_columns))
            self.scaler.scale_ = np.ones(len(self.feature_columns))
        else:
            self.scaler.fit(train_df[self.feature_columns])
        
        self.is_fitted = True
        
    def transform(self, df):
        """Transforme les données avec le préprocesseur ajusté"""
        if not self.is_fitted:
            raise RuntimeError("Le préprocesseur n'a pas été ajusté. Appelez d'abord 'fit'.")
        
        df = df.copy()
        
        # Encodage
        df['Type_encoded'] = self.encoder.transform(df['ScoreType'])
        
        # Création des features temporelles (si non présentes)
        if 'Datetime' in df.columns and 'hour_sin' not in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['Datetime'].dt.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['Datetime'].dt.hour / 24)
        
        # Normalisation des features numériques
        df[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        
        return df

    def split_data(self, df):
        """Split temporel strict avec garantie de taille minimale des sets"""
        if df.empty:
            return df.copy(), df.copy(), df.copy()
        
        # Valider les paramètres de split
        val_size = max(0.0, min(DataParams.VAL_SIZE, 0.3))  # Limiter à 30% max
        test_size = max(0.0, min(DataParams.TEST_SIZE, 0.3))  # Limiter à 30% max
        train_size = 1.0 - val_size - test_size
        
        if train_size <= 0:
            raise ValueError("La taille du set d'entraînement est <= 0. Ajustez VAL_SIZE et TEST_SIZE.")
        
        # Trier par date
        df = df.sort_values('Datetime').reset_index(drop=True)
        n = len(df)
        
        # Calculer les indices de split
        train_end_idx = max(1, int(n * train_size))  # Au moins 1 échantillon
        val_end_idx = train_end_idx + max(0, int(n * val_size))
        
        # Découpage des données
        train_df = df.iloc[:train_end_idx]
        val_df = df.iloc[train_end_idx:val_end_idx]
        test_df = df.iloc[val_end_idx:]
        
        # Vérification des tailles minimales
        min_samples = max(5, int(0.05 * n))  # Au moins 5 échantillons ou 5% des données
        if len(train_df) < min_samples:
            raise ValueError(f"Set d'entraînement trop petit ({len(train_df)} échantillons). Besoin d'au moins {min_samples} échantillons.")
        
        return train_df, val_df, test_df

    def prepare_sequences(self, df):
        """Crée des séquences temporelles avec fenêtre glissante"""
        sequences = []
        targets = []
        scores = []
        
        # Vérifier si suffisamment de données
        if len(df) < self.window_size:
            print(f"Avertissement: Taille des données ({len(df)}) < fenêtre ({self.window_size}). Aucune séquence générée.")
            return np.array([]), np.array([]), np.array([])
        
        # Création des séquences
        for i in range(self.window_size, len(df)):
            seq = df.iloc[i-self.window_size:i][self.feature_columns].values
            target = df.iloc[i]['Type_encoded']
            score = df.iloc[i]['Score']
            
            sequences.append(seq)
            targets.append(target)
            scores.append(score)
        
        return np.array(sequences), np.array(targets), np.array(scores)
    
    def save_artifacts(self):
        """Sauvegarde les préprocesseurs"""
        Paths.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, Paths.SCALER)
        joblib.dump(self.encoder, Paths.ENCODER)
        joblib.dump(self, Paths.PREPROCESSOR)