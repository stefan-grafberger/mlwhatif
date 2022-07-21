"""
The Operator Fairness What-If Analysis
"""
from typing import Iterable, Dict

import networkx

from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis


class OperatorFairness(WhatIfAnalysis):
    """
    The Operator Fairness What-If Analysis
    """

    def __init__(self):
        self._analysis_id = None

    @property
    def analysis_id(self):
        return self._analysis_id

    def generate_plans_to_try(self, dag: networkx.DiGraph) \
            -> Iterable[networkx.DiGraph]:
        return [dag]

    def generate_final_report(self, extracted_plan_results: Dict[str, any]) -> any:
        return None
