import sqlite3
import random
from datetime import datetime, timedelta
import math

# Connexion à la base de données
conn = sqlite3.connect('data/dataset.db')  # remplace par ton nom de fichier
cursor = conn.cursor()

period_counts = {0: 0, 1: 0}

def get_balanced_period():
    if period_counts[0] < period_counts[1]:
        period_counts[0] += 1
        return 0
    elif period_counts[1] < period_counts[0]:
        period_counts[1] += 1
        return 1
    else:
        p = random.randint(0, 1)
        period_counts[p] += 1
        return p

# Récupère les dernières données pour maintenir la cohérence
cursor.execute("SELECT * FROM dataset ORDER BY Id DESC LIMIT 10")
last_rows = cursor.fetchall()
previous_row = last_rows[0] if last_rows else None

# Générer les données à insérer
num_new_rows = 2000  # nombre de nouvelles lignes à insérer
start_time = datetime.strptime(previous_row[8], "%Y-%m-%d %H:%M:%S") if previous_row else datetime.now()

history = [row[3] for row in last_rows[::-1]] if last_rows else []

for _ in range(num_new_rows):
    start_time += timedelta(seconds=random.randint(10, 60))
    date_str = start_time.date().isoformat()
    time_str = start_time.time().strftime("%H:%M:%S")
    datetime_str = start_time.strftime("%Y-%m-%d %H:%M:%S")

    raw = random.expovariate(1.5)  # Plus la valeur est grande, plus le score est petit
    score = round(min(raw + 1.0, 12.0), 2)
    ScoreClass = 0 if score < 3 else 1 if score < 8 else 2
    Period = get_balanced_period()

    history.append(score)
    if len(history) > 10:
        history.pop(0)

    moyenne = round(sum(history) / len(history), 3)
    ecart_type = round((sum((x - moyenne) ** 2 for x in history) / len(history)) ** 0.5, 5)

    hour_rad = (start_time.hour * 3600 + start_time.minute * 60 + start_time.second) / 86400 * 2 * math.pi
    hour_sin = math.sin(hour_rad)
    hour_cos = math.cos(hour_rad)

    prev_score = previous_row[3] if previous_row else 0
    prev_moyenne = previous_row[6] if previous_row else 0
    prev_ScoreClass = previous_row[4] if previous_row else -1

    score_diff = round(score - prev_score, 2)
    moyenne_diff = round(moyenne - prev_moyenne, 3)
    type_repetition = 1 if ScoreClass == prev_ScoreClass else 0

    # Insertion
    cursor.execute('''
        INSERT INTO dataset (
            Date, Heure, Score, ScoreClass, Period,
            MoyenneMobileDixDernier, Ecart_Type, Datetime,
            hour_sin, hour_cos, prev_score, prev_moyenne,
            prev_ScoreClass, score_diff, moyenne_diff, type_repetition
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        date_str, time_str, score, ScoreClass, Period,
        moyenne, ecart_type, datetime_str,
        hour_sin, hour_cos, prev_score, prev_moyenne,
        prev_ScoreClass, score_diff, moyenne_diff, type_repetition
    ))

    conn.commit()

    # Met à jour la ligne précédente
    previous_row = (
        None, date_str, time_str, score, ScoreClass, Period,
        moyenne, ecart_type, datetime_str,
        hour_sin, hour_cos, prev_score, prev_moyenne,
        prev_ScoreClass, score_diff, moyenne_diff, type_repetition
    )

print(f"{num_new_rows} lignes insérées avec succès.")
conn.close()
