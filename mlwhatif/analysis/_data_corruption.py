"""
The Interface for the What-If Analyses
"""
from types import FunctionType
from typing import Iterable, Dict

import networkx

from mlwhatif import OperatorType
from mlwhatif.analysis._analysis_utils import add_intermediate_extraction_after_node, find_nodes_by_type, \
    find_first_op_modifying_a_column
from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis


class DataCorruption(WhatIfAnalysis):
    """
    The Data Corruption What-If Analysis
    """

    def __init__(self,
                 column_to_corruption: Dict[str, FunctionType],
                 corruption_percentages: Iterable[float] or None = None,
                 also_corrupt_train: bool = False):
        self.column_to_corruption = list(column_to_corruption.items())
        self.also_corrupt_train = also_corrupt_train
        if corruption_percentages is None:
            self.corruption_percentages = [0.2, 0.5, 0.9]
        self._analysis_id = (*self.column_to_corruption, *self.corruption_percentages, self.also_corrupt_train)

    @property
    def analysis_id(self):
        return self._analysis_id

    def generate_plans_to_try(self, dag: networkx.DiGraph)\
            -> Iterable[networkx.DiGraph]:
        new_dag = dag.copy()
        operator_type = OperatorType.SCORE
        score_operators = find_nodes_by_type(new_dag, operator_type)
        # TODO: Deduplication for transformers that process multiple columns at once, if necessary for corruption
        if len(score_operators) != 1:
            raise Exception("Currently, DataCorruption only supports pipelines with exactly one score call!")
        final_result_value = score_operators[0]
        add_intermediate_extraction_after_node(new_dag, final_result_value, "data-corruption-test")
        corruption_dags = []
        for (column, corruption) in self.column_to_corruption:
            corruption_dag = new_dag.copy()
            first_op_requiring_corruption = find_first_op_modifying_a_column(corruption_dag, column, True)
            corruption_dags.append(corruption)
        return corruption_dags  # TODO

    def generate_final_report(self, extracted_plan_results: Dict[str, any]) -> any:
        return "TODO"  # TODO
