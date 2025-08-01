from evidently.report import Report
from evidently.metrics import (
    DataDriftTable,
    DatasetDriftMetric,
    ClassificationQualityMetric
)
from config.paths import Paths
import pandas as pd

class CasinoMonitor:
    def __init__(self, reference_data: pd.DataFrame):
        self.reference = reference_data
        
    def check_drift(self, current_data: pd.DataFrame) -> dict:
        """Analyse la dérive des données et des performances"""
        report = Report(metrics=[
            DataDriftTable(),
            DatasetDriftMetric(),
            ClassificationQualityMetric()
        ])
        
        report.run(
            reference_data=self.reference,
            current_data=current_data,
            column_mapping=self._get_column_mapping()
        )
        
        report.save_html(Paths.DRIFT_REPORT)
        return report.as_dict()
    
    def _get_column_mapping(self) -> dict:
        return {
            'target': 'Type_encoded',
            'numerical_features': ['Score', 'MoyenneMobileDixDernier', 'Ecart_Type'],
            'datetime': 'Datetime'
        }