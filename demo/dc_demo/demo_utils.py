# pylint: disable-all
from typing import Dict, Iterable

import networkx

from mlwhatif.analysis._data_cleaning import DataCleaning, ErrorType
from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis
from mlwhatif.execution._patches import PipelinePatch


class TrainingSetDebugging(WhatIfAnalysis):
    """
    The Data Cleaning What-If Analysis
    """

    def __init__(self):
        self._analysis = DataCleaning({None: ErrorType.MISLABEL})
        self._score_nodes_and_linenos = []
        self._analysis_id = (None,)

    @property
    def analysis_id(self):
        return self._analysis_id

    def generate_plans_to_try(self, dag: networkx.DiGraph) -> Iterable[Iterable[PipelinePatch]]:
        return self._analysis.generate_plans_to_try(dag)

    def generate_final_report(self, extracted_plan_results: Dict[str, any]) -> any:
        return self._analysis.generate_final_report(extracted_plan_results)
