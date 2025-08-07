# database.py
import sqlite3
import pandas as pd
from config.paths import Paths
from typing import Optional

class CasinoDatabase:
    def __init__(self):
        Paths.DATA_DIR.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(Paths.DATABASE)


    def fetch_data(self, last_n: Optional[int] = None) -> pd.DataFrame:
        query = """
        WITH computed_data AS (
            SELECT *,
                CASE
                    WHEN Score < 3 THEN 0
                    WHEN Score BETWEEN 3 AND 10 THEN 1
                    WHEN Score > 10 THEN 2
                END AS ScoreClass,
                CASE
                    WHEN MoyenneMobileDixDernier >= 3 THEN 1
                    ELSE 0
                END AS Period
            FROM dataset
        ),
        lagged AS (
            SELECT *,
                COALESCE(LAG(Score, 1) OVER (ORDER BY Date, Heure), 0) AS prev_score,
                COALESCE(LAG(MoyenneMobileDixDernier, 1) OVER (ORDER BY Date, Heure), 0) AS prev_moyenne,
                COALESCE(LAG(ScoreClass, 1) OVER (ORDER BY Date, Heure), -1) AS prev_type
            FROM computed_data
        )
        SELECT 
            Id, Date, Heure, Score, ScoreClass, Period,
            MoyenneMobileDixDernier, Ecart_Type, prev_score, prev_moyenne,
            prev_type AS prev_ScoreClass,
            COALESCE(Score - prev_score, 0) AS score_diff,
            COALESCE(MoyenneMobileDixDernier - prev_moyenne, 0) AS moyenne_diff,
            CASE WHEN ScoreClass = prev_ScoreClass THEN 1 ELSE 0 END AS type_repetition,
            Datetime, hour_sin, hour_cos
        FROM lagged
        """
        
        if last_n:
            query = f"{query} ORDER BY Date DESC, Heure DESC LIMIT {last_n}"
        else:
            query += " ORDER BY Date, Heure"
        
        df = pd.read_sql(query, self.conn)
        
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
        else:
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Heure'])
        
        return df.sort_values('Datetime')
    
    def __del__(self):
        self.conn.close()