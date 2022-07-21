"""
The Operator Fairness What-If Analysis
"""
from typing import Iterable, Dict, List, Callable

import networkx

from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis


class OperatorFairness(WhatIfAnalysis):
    """
    The Operator Fairness What-If Analysis
    """

    def __init__(self, sensitive_columns: List[str], metric: Callable):
        self._sensitive_columns = sensitive_columns
        self._metric = metric
        self._analysis_id = tuple(*self._sensitive_columns)

    @property
    def analysis_id(self):
        return self._analysis_id

    def generate_plans_to_try(self, dag: networkx.DiGraph) \
            -> Iterable[networkx.DiGraph]:
        return [dag]

    def generate_final_report(self, extracted_plan_results: Dict[str, any]) -> any:
        return None
