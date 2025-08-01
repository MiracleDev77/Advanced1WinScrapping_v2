import sqlite3
import pandas as pd
from config.paths import Paths
from typing import Optional

class CasinoDatabase:
    def __init__(self):
        Paths.DATA_DIR.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(Paths.DATABASE)
        

    def fetch_data(self, last_n: Optional[int] = None) -> pd.DataFrame:
        """
        Charge les données avec features temporelles avancées
        
        Args:
            last_n: Nombre de derniers enregistrements à récupérer (optionnel)
        
        Returns:
            DataFrame avec les données enrichies
        """
        base_query = """
        WITH lagged AS (
            SELECT *,
                   LAG(Score, 1) OVER (ORDER BY Date, Heure) as prev_score,
                   LAG(MoyenneMobileDixDernier, 1) OVER (ORDER BY Date, Heure) as prev_moyenne,
                   LAG(ScoreType, 1) OVER (ORDER BY Date, Heure) as prev_type
            FROM dataset_with_types
        )
        SELECT 
            Id, Date, Heure,
            Score, ScoreType, MoyenneMobileDixDernier, Ecart_Type,
            prev_score, prev_moyenne, prev_type,
            Score - prev_score as score_diff,
            MoyenneMobileDixDernier - prev_moyenne as moyenne_diff,
            CASE WHEN ScoreType = prev_type THEN 1 ELSE 0 END as type_repetition,
            Datetime, Type_encoded, hour_sin, hour_cos
        FROM lagged
        """
        
        if last_n:
            query = f"""
            SELECT * FROM ({base_query})
            ORDER BY Date DESC, Heure DESC
            LIMIT {last_n}
            """
        else:
            query = base_query + " ORDER BY Date, Heure"
        
        df = pd.read_sql(query, self.conn)
        
        # Conversion des dates/heures si nécessaire
        if 'Datetime' in df.columns:
            df['Datetime'] = pd.to_datetime(df['Datetime'])
        else:
            df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Heure'])
        
        return df.sort_values('Datetime')
    
    def __del__(self):
        self.conn.close()