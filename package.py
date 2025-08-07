import sqlite3
import numpy as np
from datetime import datetime

# Initialisation de la connexion
Connexion = sqlite3.connect('data/dataset.db')
Cursor = Connexion.cursor()

def init_database():
    """Crée la table si elle n'existe pas"""
    Cursor.execute('''
    CREATE TABLE IF NOT EXISTS dataset (
        Id INTEGER PRIMARY KEY AUTOINCREMENT,
        Date TEXT NOT NULL,
        Heure TEXT NOT NULL,
        Score REAL NOT NULL,
        ScoreClass INTEGER,
        Period INTEGER,
        MoyenneMobileDixDernier REAL DEFAULT 0,
        Ecart_Type REAL DEFAULT 0,
        Datetime TEXT,
        hour_sin REAL,
        hour_cos REAL,
        prev_score REAL,
        prev_moyenne REAL,
        prev_ScoreClass INTEGER,
        score_diff REAL,
        moyenne_diff REAL,
        type_repetition INTEGER
    );
    ''')
    Connexion.commit()

def addLastScore(Data):
    """Version mise à jour avec gestion robuste"""
    date, heure, score, moyenne_mobile, ecart = Data
    
    # Calculer ScoreClass et Period
    if score < 3:
        score_class = 0  # Perdant
    elif 3 <= score <= 10:
        score_class = 1  # Gagnant
    else:
        score_class = 2  # Bonus
    
    period = 1 if moyenne_mobile >= 3 else 0  # 1=Période favorable, 0=Période défavorable
    
    # Récupérer dernière entrée avec gestion des erreurs
    Cursor.execute("SELECT Score, MoyenneMobileDixDernier, ScoreClass FROM dataset ORDER BY Id DESC LIMIT 1")
    last_row = Cursor.fetchone()
    
    # Valeurs par défaut si première entrée
    prev_score = last_row[0] if last_row else 0
    prev_moyenne = last_row[1] if last_row else 0
    prev_score_class = last_row[2] if last_row else -1
    
    # Calculer les features temporelles
    datetime_str = f"{date} {heure}"
    hour = int(heure.split(':')[0])
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # Requête d'insertion
    Req = '''
    INSERT INTO dataset(
        Date, Heure, Score, ScoreClass, Period,
        MoyenneMobileDixDernier, Ecart_Type,
        Datetime, hour_sin, hour_cos,
        prev_score, prev_moyenne, prev_ScoreClass,
        score_diff, moyenne_diff, type_repetition
    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    '''
    
    new_data = (
        date, heure, score, score_class, period,
        moyenne_mobile, ecart,
        datetime_str, hour_sin, hour_cos,
        prev_score, prev_moyenne, prev_score_class,
        score - prev_score,
        moyenne_mobile - prev_moyenne,
        1 if score_class == prev_score_class else 0
    )
    
    try:
        Cursor.execute(Req, new_data)
        Connexion.commit()
        return 0
    except Exception as e:
        print(f"Erreur SQL: {str(e)}")
        return 1

def getLastScore():
    """Récupère le dernier score avec gestion des erreurs"""
    try:
        Cursor.execute("SELECT Score FROM dataset ORDER BY Id DESC LIMIT 1")
        result = Cursor.fetchone()
        return result[0] if result else 0
    except:
        return 0

def getTenLastScoreInArray():
    """Récupère les 10 derniers scores avec gestion des erreurs"""
    try:
        Cursor.execute("SELECT Score FROM dataset ORDER BY Id DESC LIMIT 10")
        return [row[0] for row in Cursor.fetchall()]
    except:
        return []