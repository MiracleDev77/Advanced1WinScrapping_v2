import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from config.paths import Paths
from config.params import DataParams
from imblearn.over_sampling import SMOTE

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
        if DataParams.STRATIFY_SPLIT:
            try:
                # Créer une colonne de stratification combinée
                df['stratify_col'] = df['ScoreClass'].astype(str)
                
                # Split stratifié avec garantie de représentation des classes minoritaires
                sss = StratifiedShuffleSplit(
                    n_splits=1, 
                    test_size=DataParams.VAL_SIZE + DataParams.TEST_SIZE,
                    random_state=42
                )
                
                for train_index, temp_index in sss.split(df, df['stratify_col']):
                    train = df.iloc[train_index]
                    temp = df.iloc[temp_index]
                
                # Second split stratifié
                sss2 = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=DataParams.TEST_SIZE/(DataParams.VAL_SIZE + DataParams.TEST_SIZE),
                    random_state=42
                )
                
                for val_index, test_index in sss2.split(temp, temp['stratify_col']):
                    val = temp.iloc[val_index]
                    test = temp.iloc[test_index]
                
                return train, val, test
            except Exception as e:
                print(f"Erreur de stratification: {e}")
                return self.split_without_stratification(df)
        else:
            return self.split_without_stratification(df)
    
    def split_without_stratification(self, df):
        total_size = len(df)
        min_size = self.window_size + 1
        
        # Calcul des tailles initiales
        train_size = int(total_size * (1 - DataParams.VAL_SIZE - DataParams.TEST_SIZE))
        val_size = int(total_size * DataParams.VAL_SIZE)
        test_size = total_size - train_size - val_size

        # Ajustement si les ensembles sont trop petits
        if val_size < min_size or test_size < min_size:
            if total_size < 3 * min_size:
                # Si trop peu de données, utiliser tout pour l'entraînement
                print("⚠️ Très peu de données - Utilisation de tout le dataset pour l'entraînement")
                train = df
                val = pd.DataFrame(columns=df.columns)
                test = pd.DataFrame(columns=df.columns)
                return train, val, test
            
            # Réallocation avec tailles minimales garanties
            val_size = min_size
            test_size = min_size
            train_size = total_size - val_size - test_size

        train = df.iloc[:train_size]
        val = df.iloc[train_size:train_size+val_size]
        test = df.iloc[train_size+val_size:]
        
        return train, val, test

    def fit(self, df):
        if len(df) == 0:
            print("⚠️ Aucune donnée pour le fitting du scaler")
            return
            
        numeric_cols = [col for col in self.feature_columns 
                        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if numeric_cols:
            self.scaler.fit(df[numeric_cols])

    def transform(self, df):
        if len(df) == 0:
            return df
            
        df = df.copy()
        numeric_cols = [col for col in self.feature_columns 
                        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
        if numeric_cols:
            # Vérifier si le scaler a été entraîné
            if hasattr(self.scaler, 'mean_'):
                df[numeric_cols] = self.scaler.transform(df[numeric_cols])
            else:
                print("⚠️ Scaler non entraîné - Transformation ignorée")
        return df

    def prepare_sequences(self, df):
        if len(df) == 0:
            return np.array([]), (np.array([]), np.array([])), np.array([])
            
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
            
        # Retourner des tableaux vides si aucune séquence n'a été créée
        if not X:
            return np.array([]), (np.array([]), np.array([])), np.array([])
            
        return np.array(X), (np.array(y_class), np.array(y_period)), np.array(scores)

    def save_artifacts(self, suffix=""):
        joblib.dump(self.scaler, Paths.SCALER.with_stem(Paths.SCALER.stem + suffix))
        joblib.dump(self, Paths.PREPROCESSOR.with_stem(Paths.PREPROCESSOR.stem + suffix))

    def balance_classes(self, df):
        """Rééquilibre les classes en augmentant les classes minoritaires"""
        if not DataParams.BALANCE_CLASSES:
            return df
            
        # Calculer la classe majoritaire
        class_counts = df['ScoreClass'].value_counts()
        majority_class = class_counts.idxmax()
        max_count = class_counts.max()
        
        balanced_dfs = []
        
        for cls in class_counts.index:
            cls_df = df[df['ScoreClass'] == cls]
            n_samples = len(cls_df)
            
            if n_samples < max_count:
                # Facteur d'augmentation spécifique pour les classes minoritaires
                factor = DataParams.AUGMENT_MINORITY_FACTOR if cls != majority_class else 1
                augmented = [cls_df]
                
                for _ in range(factor - 1):
                    aug_df = cls_df.copy()
                    numeric_cols = aug_df.select_dtypes(include=np.number).columns
                    noise = np.random.normal(0, DataParams.AUGMENT_NOISE, (len(aug_df), len(numeric_cols)))
                    aug_df[numeric_cols] += noise
                    augmented.append(aug_df)
                
                cls_df = pd.concat(augmented, ignore_index=True)
            
            balanced_dfs.append(cls_df)
        
        return pd.concat(balanced_dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)