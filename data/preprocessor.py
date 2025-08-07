# preprocessor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from config.paths import Paths
from config.params import DataParams

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'Score', 'MoyenneMobileDixDernier', 'Ecart_Type',
            'prev_score', 'prev_moyenne', 'score_diff', 
            'moyenne_diff', 'type_repetition'
        ]
        self.target_columns = ['ScoreClass', 'Period', 'Score']
        self.window_size = DataParams.WINDOW_SIZE
        self.fold = 0

    def split_data(self, df):
        total_size = len(df)
        train_size = int(total_size * (1 - DataParams.VAL_SIZE - DataParams.TEST_SIZE))
        val_size = int(total_size * DataParams.VAL_SIZE)
        
        train = df.iloc[:train_size]
        val = df.iloc[train_size:train_size+val_size]
        test = df.iloc[train_size+val_size:]
        
        return train, val, test

    def fit(self, df):
        numeric_cols = [col for col in self.feature_columns 
                        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if numeric_cols:
            self.scaler.fit(df[numeric_cols])

    def transform(self, df):
        df = df.copy()
        numeric_cols = [col for col in self.feature_columns 
                        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if numeric_cols:
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])
        return df

    def prepare_sequences(self, df):
        X, y_class, y_period, scores = [], [], [], []
        
        for i in range(self.window_size, len(df)):
            sequence = df[self.feature_columns].iloc[i-self.window_size:i].values
            target_class = df['ScoreClass'].iloc[i]
            target_period = df['Period'].iloc[i]
            score = df['Score'].iloc[i]
            
            X.append(sequence)
            y_class.append(target_class)
            y_period.append(target_period)
            scores.append(score)
            
        return np.array(X), (np.array(y_class), np.array(y_period)), np.array(scores)

    def save_artifacts(self, suffix=""):
        joblib.dump(self.scaler, Paths.SCALER.with_stem(Paths.SCALER.stem + suffix))
        joblib.dump(self, Paths.PREPROCESSOR.with_stem(Paths.PREPROCESSOR.stem + suffix))