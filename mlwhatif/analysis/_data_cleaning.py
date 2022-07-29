"""
The Data Cleaning What-If Analysis
"""
from typing import Iterable, Dict

import networkx

from mlwhatif import OperatorType
from mlwhatif.analysis._analysis_utils import find_nodes_by_type
from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis


class DataCleaning(WhatIfAnalysis):
    """
    The Data Cleaning What-If Analysis
    """

    def __init__(self, ):
        self._analysis_id = ()

    @property
    def analysis_id(self):
        return self._analysis_id

    def generate_plans_to_try(self, dag: networkx.DiGraph) \
            -> Iterable[networkx.DiGraph]:
        predict_operators = find_nodes_by_type(dag, OperatorType.PREDICT)
        if len(predict_operators) != 1:
            raise Exception("Currently, DataCorruption only supports pipelines with exactly one predict call which "
                            "must be on the test set!")
        score_operators = find_nodes_by_type(dag, OperatorType.SCORE)
        self._score_nodes_and_linenos = [(node, node.code_location.lineno) for node in score_operators]
        if len(self._score_nodes_and_linenos) != len(set(self._score_nodes_and_linenos)):
            raise Exception("Currently, DataCorruption only supports pipelines where different score operations can "
                            "be uniquely identified by the line number in the code!")
        return [dag]

    def generate_final_report(self, extracted_plan_results: Dict[str, any]) -> any:
        return None
