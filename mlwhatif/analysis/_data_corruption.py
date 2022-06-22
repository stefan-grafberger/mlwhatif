"""
The Interface for the What-If Analyses
"""
from typing import Iterable, Dict, Callable

import networkx
import pandas

from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis


class DataCorruption(WhatIfAnalysis):
    """
    The Data Corruption What-If Analysis
    """

    def __init__(self,
                 column_to_corruption: Dict[str, Callable[[pandas.DataFrame], pandas.DataFrame]],
                 corruption_percentages: Iterable[float] or None = None):
        self.column_to_corruption = list(column_to_corruption.values())
        if corruption_percentages is None:
            self.corruption_percentages = [0.2, 0.5, 0.9]
        self._analysis_id = (*self.column_to_corruption, *self.corruption_percentages)

    @property
    def analysis_id(self):
        return self._analysis_id

    def generate_plans_to_try(self, dag: networkx.DiGraph)\
            -> Iterable[networkx.DiGraph]:
        return [dag]  # TODO

    def generate_final_report(self, extracted_plan_results: Dict[str, any]) -> any:
        return "TODO"  # TODO
