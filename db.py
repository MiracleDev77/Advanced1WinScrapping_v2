import sqlite3
import numpy as np
import os

class DatabaseManager:
    def __init__(self, db_path='dataset4.db'):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialise la base de données et la connexion"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
        # Création de la table si elle n'existe pas
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS dataset (
            Id INTEGER PRIMARY KEY AUTOINCREMENT,
            Date TEXT NOT NULL,
            Heure TEXT NOT NULL,
            Score REAL NOT NULL,
            Type TEXT DEFAULT 'NO',
            MoyenneMobileDixDernier REAL DEFAULT 0,
            Ecart_Type REAL DEFAULT 0,
            Datetime TEXT,
            Type_encoded INTEGER,
            hour_sin REAL,
            hour_cos REAL,
            prev_score REAL,
            prev_moyenne REAL,
            prev_type TEXT,
            score_diff REAL,
            moyenne_diff REAL,
            type_repetition INTEGER
        );
        ''')
        self.conn.commit()
    
    def add_last_score(self, data):
        """Ajoute un nouveau score avec calcul des métadonnées"""
        try:
            date_str, heure_str, score, score_type, moyenne, ecart = data
            
            # Conversion de l'heure
            hour = int(heure_str.split(':')[0])
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            datetime_str = f"{date_str} {heure_str}"
            
            # Récupération du dernier score
            last_row = self.get_last_score(full_record=True)
            
            # Calcul des différences
            prev_score = last_row['Score'] if last_row else 0.0
            prev_moyenne = last_row['MoyenneMobileDixDernier'] if last_row else 0.0
            prev_type = last_row['Type'] if last_row else 'ND'
            
            # Préparation des données
            new_data = (
                date_str, heure_str, score, score_type, 
                moyenne, ecart, datetime_str, 
                0,  # Type_encoded (temporaire)
                hour_sin, hour_cos,
                prev_score, prev_moyenne, prev_type,
                score - prev_score,
                moyenne - prev_moyenne,
                1 if score_type == prev_type else 0
            )
            
            # Insertion dans la base
            self.cursor.execute('''
            INSERT INTO dataset(
                Date, Heure, Score, Type, MoyenneMobileDixDernier, Ecart_Type,
                Datetime, Type_encoded, hour_sin, hour_cos,
                prev_score, prev_moyenne, prev_type,
                score_diff, moyenne_diff, type_repetition
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', new_data)
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Erreur add_last_score: {str(e)}")
            return False

    def get_last_score(self, full_record=False):
        """Récupère le dernier score"""
        try:
            if full_record:
                self.cursor.execute('''
                SELECT * FROM dataset 
                ORDER BY Date DESC, Heure DESC 
                LIMIT 1
                ''')
                row = self.cursor.fetchone()
                if row:
                    columns = [col[0] for col in self.cursor.description]
                    return dict(zip(columns, row))
                return None
            
            self.cursor.execute('''
            SELECT Score FROM dataset 
            ORDER BY Date DESC, Heure DESC 
            LIMIT 1
            ''')
            row = self.cursor.fetchone()
            return row[0] if row else 0.0
        except Exception as e:
            print(f"Erreur get_last_score: {str(e)}")
            return 0.0

    def get_ten_last_scores(self):
        """Récupère les 10 derniers scores sous forme de liste"""
        try:
            self.cursor.execute('''
            SELECT Score FROM dataset 
            ORDER BY Date DESC, Heure DESC 
            LIMIT 10
            ''')
            rows = self.cursor.fetchall()
            return [row[0] for row in rows] if rows else []
        except Exception as e:
            print(f"Erreur get_ten_last_scores: {str(e)}")
            return []

    def close(self):
        """Ferme la connexion à la base de données"""
        self.conn.close()