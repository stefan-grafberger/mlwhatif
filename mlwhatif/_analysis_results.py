"""
Data class used as result of the PipelineExecutor
"""
import dataclasses
from typing import Dict, List

import networkx

from mlwhatif.execution._patches import PipelinePatch
from mlwhatif.visualisation import save_fig_to_path


@dataclasses.dataclass
class AnalysisResults:
    """
    The class the PipelineExecutor returns
    """
    analysis_to_result_reports: Dict[any, any]
    original_dag: networkx.DiGraph
    what_if_dags: List[tuple[List[PipelinePatch], networkx.DiGraph]]
    combined_optimized_dag: networkx.DiGraph

    def save_original_dag_to_path(self, prefix_original_dag: str):
        """
        Save the extracted original DAG to a file
        """
        save_fig_to_path(self.original_dag, f"{prefix_original_dag}.png")

    def save_what_if_dags_to_path(self, prefix_analysis_dags: str):
        """
        Save the generated What-If DAGs to a file
        """
        what_if_dags = [dag for _, dag in self.what_if_dags]
        for dag_index, what_if_dag in enumerate(what_if_dags):
            if prefix_analysis_dags is not None:
                save_fig_to_path(what_if_dag, f"{prefix_analysis_dags}-{dag_index}.png")

    def save_optimised_what_if_dags_to_path(self, prefix_optimised_analysis_dag: str):
        """
        Save the extracted original DAG to a file
        """
        save_fig_to_path(self.combined_optimized_dag, f"{prefix_optimised_analysis_dag}.png")
