import sqlite3
import numpy as np 

Connexion = sqlite3.connect('dataset.db')
Cursor = Connexion.cursor()



def addLastScore(Data):
    # Convertir les données de base
    date, heure, score, type_, moyenne, ecart = Data
    
    # Calculer les nouvelles valeurs
    datetime_str = f"{date} {heure}"
    hour = int(heure.split(':')[0])
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # Récupérer la dernière entrée
    Cursor.execute("SELECT * FROM dataset ORDER BY Id DESC LIMIT 1")
    last_row = Cursor.fetchone()
    
    # Calculer les différences
    prev_score = last_row[3] if last_row else 0
    prev_moyenne = last_row[5] if last_row else 0
    prev_type = last_row[4] if last_row else 'ND'
    
    # Requête d'insertion mise à jour
    Req = '''
    INSERT INTO dataset(
        Date, Heure, Score, Type, 
        MoyenneMobileDixDernier, Ecart_Type,
        Datetime, Type_encoded, hour_sin, hour_cos,
        prev_score, prev_moyenne, prev_type,
        score_diff, moyenne_diff, type_repetition
    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    '''
    
    # Type encodé temporaire (sera recalculé lors du préprocessing)
    type_encoded = 0 if type_ == 'Gagnant' else 1 if type_ == 'Perdant' else 2
    
    new_data = Data + [
        datetime_str,
        type_encoded,
        hour_sin,
        hour_cos,
        prev_score,
        prev_moyenne,
        prev_type,
        score - prev_score,
        moyenne - prev_moyenne,
        1 if type_ == prev_type else 0
    ]
    
    Cursor.execute(Req, new_data)
    Connexion.commit()
    return 0

def getLastScore():
    Req = 'SELECT Score FROM dataset ORDER BY Date DESC, Heure DESC LIMIT 1'
    Cursor.execute(Req)
    Data = Cursor.fetchone()
    if Data:
    	return Data[0]
    else:
    	return 0

def getTenLastScore():
    Req = 'SELECT Score FROM dataset ORDER BY Date DESC, Heure DESC LIMIT 10'
    Cursor.execute(Req)
    Data = Cursor.fetchall()
    if Data:
    	return Data
    else:
    	return 0

def getTenLastScoreInArray():
	Req = 'SELECT Score FROM dataset ORDER BY Date DESC, Heure DESC LIMIT 10'
	Cursor.execute(Req)
	Data = Cursor.fetchall()
	if Data:
		lastData = []
		for last in Data:
			lastData.append(last[0])

		return lastData

	else:
		return 0



def getScore():
    Req = 'SELECT Date,Heure,Score FROM dataset'
    Cursor.execute(Req)
    return Cursor.fetchall()

def find_sequence(lst, value):
	sequences = []
	current_sequence = []

	for i, item in enumerate(lst):
		if item == value:
			current_sequence.append(i)
		else:
			if len(current_sequence) > 1:
				sequences.append(current_sequence)
				current_sequence = []
				if len(current_sequence) > 1:
					sequences.append(current_sequence)

					return sequences