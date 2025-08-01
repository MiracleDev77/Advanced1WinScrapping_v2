from pathlib import Path
#from database import CasinoDatabase
from config.paths import Paths
import sqlite3

import random
import datetime
import math

def afficher_structure(path: Path, prefix=""):
    contenu = sorted(path.iterdir())
    for i, item in enumerate(contenu):
        is_last = i == len(contenu) - 1
        branche = "└── " if is_last else "├── "
        print(prefix + branche + item.name)
        if item.is_dir():
            extension = "    " if is_last else "│   "
            afficher_structure(item, prefix + extension)

# Utilisation
#afficher_structure(Path("."))  # racine du projet


#db = CasinoDatabase();

Connexion = sqlite3.connect('dataset.db')
Cursor = Connexion.cursor()

def check_and_add_columns():
    """Ajoute les nouvelles colonnes si elles n'existent pas"""
    new_columns = [
        ('Datetime', 'TEXT'),
        ('Type_encoded', 'INTEGER'),
        ('hour_sin', 'REAL'),
        ('hour_cos', 'REAL'),
        ('prev_score', 'REAL'),
        ('prev_moyenne', 'REAL'),
        ('prev_type', 'TEXT'),
        ('score_diff', 'REAL'),
        ('moyenne_diff', 'REAL'),
        ('type_repetition', 'INTEGER')
    ]
    
    for col_name, col_type in new_columns:
        try:
            # Vérifier si la colonne existe déjà
            Cursor.execute(f"SELECT {col_name} FROM dataset LIMIT 1")
        except sqlite3.OperationalError:
            # Ajouter la colonne si elle n'existe pas
            Cursor.execute(f"ALTER TABLE dataset ADD COLUMN {col_name} {col_type}")
    
    Connexion.commit()

def create_score_type_view():
    """Crée une vue SQL pour calculer les types de score dynamiquement"""
    view_query = """
    CREATE VIEW IF NOT EXISTS dataset_with_types AS
    SELECT *,
        CASE
            WHEN Score < 2 THEN 'Faible'
            WHEN Score BETWEEN 2 AND 4.59 THEN 'Moyenne'
            WHEN Score BETWEEN 5 AND 9.9 THEN 'Bonne'
            WHEN Score BETWEEN 10 AND 49.9 THEN 'Bonne-49'
            WHEN Score BETWEEN 50 AND 99.9 THEN 'Bonne-99'
            WHEN Score >= 100 THEN 'Jackpot'
            ELSE 'Inconnu'
        END AS ScoreType
    FROM dataset
    """
    Cursor.execute(view_query)
    Connexion.commit()

# Dans package.py ou database.py
def initialize_database():
    
    
    Cursor.execute("""
    CREATE TABLE IF NOT EXISTS dataset_with_types (
        Id INTEGER PRIMARY KEY,
        Date TEXT NOT NULL,
        Heure TEXT NOT NULL,
        Score REAL NOT NULL,
        ScoreType TEXT NOT NULL,
        MoyenneMobileDixDernier REAL,
        Ecart_Type REAL,
        Datetime TEXT,
        Type_encoded INTEGER,
        hour_sin REAL,
        hour_cos REAL
    )
    """)
    
    Connexion.commit()
    Connexion.close()






# Données manuellement extraites de la capture
donnees = [
    (859, "2025-04-06", "01:09:28.606034", 2.89, "Moyenne", 1, 0),
    (860, "2025-04-06", "01:09:51.261135", 10.07, "Bonne-49", 2.89, 6.48),
    (861, "2025-04-06", "01:10:07.780340", 1.69, "Faible", 6.48, 5.07702668891941),
    (862, "2025-04-06", "01:10:24.705201", 1.68, "Faible", 4.883333333333333, 4.53168107144946),
    (863, "2025-04-06", "01:11:10.104364", 22.54, "Bonne-49", 4.0825, 4.0188438160951),
    (864, "2025-04-06", "01:11:50.692622", 8.79, "Bonne", 8.96258005057307, 7.774),
    (865, "2025-04-06", "01:12:19.609681", 6.52, "Bonne", 7.943333333333333, 8.02710320501464),
    (866, "2025-04-06", "01:12:36.619583", 1.71, "Faible", 7.74, 7.3474038982564),
    (867, "2025-04-06", "01:13:57.449038", 369.74, "Jackpot", 6.98625, 7.1286533001259),
    (868, "2025-04-06", "01:14:15.129333", 1.48, "Faible", 47.2922222222222, 121.101643648814),
    (869, "2025-04-06", "01:14:32.709581", 1.68, "Faible", 42.711, 115.091146724865),
    (870, "2025-04-06", "01:14:44.288973", 1.01, "Faible", 42.59, 115.138920821348),
    (871, "2025-04-06", "01:15:56.349065", 1.12, "Faible", 41.684, 115.457818483154),
    (872, "2025-04-06", "01:15:09.378961", 1.18, "Faible", 41.627, 115.479895417149),
    (873, "2025-04-06", "01:15:30.525269", 3.84, "Moyenne", 40.59, 115.499219951355),
    (874, "2025-04-06", "01:15:57.248114", 3.18, "Moyenne", 39.657, 116.009302972963),
    (875, "2025-04-06", "01:16:24.483634", 3.07, "Moyenne", 39.096, 116.188581213665),
    (876, "2025-04-06", "01:17:00.008535", 9.48, "Bonne", 38.751, 116.301124619574),
    (877, "2025-04-06", "01:17:15.838392", 1.85, "Faible", 2.61, 116.051847297225),
    (878, "2025-04-06", "01:17:32.611372", 1.15, "Faible", 2.3, 2.561689765320464),
    (879, "2025-04-06", "01:17:47.638340", 1.15, "Faible", 2.691, 2.59025438476181),
    (880, "2025-04-06", "01:17:48.748328", 1.0, "Faible", 2.638, 2.59025438476181)
]

def calculs_heure(heure_str):
    t = datetime.datetime.strptime(heure_str, "%H:%M:%S.%f")
    seconds = t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
    angle = 2 * math.pi * (seconds / 86400)
    return math.sin(angle), math.cos(angle)

# Insérer les données dans la base avec des valeurs aléatoires pour les champs NULL
for row in donnees:
    Id, Date, Heure, Score, Type, Moyenne, Ecart = row
    dt = f"{Date} {Heure}"
    hour_sin, hour_cos = calculs_heure(Heure)
    
    # Génération aléatoire simulée pour les données NULL
    Type_encoded = random.randint(0, 3)
    prev_score = round(random.uniform(0.5, 15), 2)
    prev_moyenne = round(random.uniform(0.5, 40), 2)
    prev_type = random.choice(["Faible", "Bonne", "Bonne-49", "Moyenne", "Jackpot"])
    score_diff = Score - prev_score
    moyenne_diff = Moyenne - prev_moyenne
    type_repetition = random.randint(0, 5)

    Cursor.execute("""
        INSERT INTO dataset (
            Id, Date, Heure, Score, Type, MoyenneMobileDixDernier, Ecart_Type, Datetime,
            Type_encoded, hour_sin, hour_cos, prev_score, prev_moyenne, prev_type,
            score_diff, moyenne_diff, type_repetition
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        Id, Date, Heure, Score, Type, Moyenne, Ecart, dt,
        Type_encoded, hour_sin, hour_cos, prev_score, prev_moyenne, prev_type,
        score_diff, moyenne_diff, type_repetition
    ))

# Commit et fermeture
Connexion.commit()
Connexion.close()

