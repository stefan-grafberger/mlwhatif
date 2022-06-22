"""
Data class used as result of the PipelineExecutor
"""
import dataclasses
from typing import Dict

import networkx

from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis


@dataclasses.dataclass
class AnalysisResults:
    """
    The class the PipelineExecutor returns
    """
    dag: networkx.DiGraph
    analysis_to_result_reports: Dict[WhatIfAnalysis, any]
