"""
Data class used as result of the PipelineExecutor
"""
import dataclasses
from typing import Dict, List, Tuple

import networkx

from mlwhatif.execution._patches import PipelinePatch
from mlwhatif.visualisation import save_fig_to_path
from mlwhatif.visualisation._visualisation import get_original_simple_dag, get_colored_simple_dags, \
    get_final_optimized_combined_colored_simple_dag, save_simple_fig_to_path


@dataclasses.dataclass
class RuntimeInfo:
    """
    The info that also gets logged during execution
    """
    # pylint: disable=too-many-instance-attributes
    original_pipeline_importing_and_monkeypatching: int
    original_pipeline_without_importing_and_monkeypatching: int
    original_pipeline_estimated: int
    original_model_training: int
    original_pipeline_train_data_shape: Tuple[int, int] or None
    original_pipeline_test_data_shape: Tuple[int, int] or None
    what_if_plan_generation: int
    what_if_unoptimized_estimated: int
    what_if_optimized_estimated: int
    what_if_optimization_saving_estimated: int
    what_if_query_optimization_duration: int
    what_if_execution: int
    what_if_execution_combined_model_training: int


@dataclasses.dataclass
class DagExtractionInfo:
    """All info required to reuse a previously extracted DAG for different what-if analyses"""
    original_dag: networkx.DiGraph
    original_pipeline_labels_to_extracted_plan_results: Dict[str, any]
    next_op_id: int
    next_patch_id: int
    next_missing_op_id: int


@dataclasses.dataclass
class AnalysisResults:
    """
    The class the PipelineExecutor returns
    """
    # pylint: disable=too-many-instance-attributes
    analysis_to_result_reports: Dict[any, any]
    original_dag: networkx.DiGraph
    what_if_dags: List[tuple[List[PipelinePatch], networkx.DiGraph]]
    combined_optimized_dag: networkx.DiGraph
    runtime_info: RuntimeInfo
    dag_extraction_info: DagExtractionInfo
    pipeline_executor: any
    intermediate_stages: Dict[str, List[networkx.DiGraph]]

    def save_original_dag_to_path(self, prefix_original_dag: str):
        """
        Save the extracted original DAG to a file
        """
        save_simple_fig_to_path(get_original_simple_dag(self.original_dag), f"{prefix_original_dag}.png")

    def save_what_if_dags_to_path(self, prefix_analysis_dags: str):
        """
        Save the generated What-If DAGs to a file
        """
        colored_simple_dags = get_colored_simple_dags(self.intermediate_stages["unoptimized_variants"],
                                                      with_reuse_coloring=False)
        for dag_index, what_if_dag in enumerate(colored_simple_dags):
            if prefix_analysis_dags is not None:
                save_simple_fig_to_path(what_if_dag, f"{prefix_analysis_dags}-{dag_index}.png")

    def save_all_what_if_dag_stages_to_path(self, prefix_analysis_dags: str):
        """
        Save the generated What-If DAGs to a file
        """
        for stage_name in self.intermediate_stages.keys():
            if not stage_name == "unoptimized_variants":
                colored_simple_dags = get_colored_simple_dags(self.intermediate_stages[stage_name],
                                                              with_reuse_coloring=True)
                for dag_index, what_if_dag in enumerate(colored_simple_dags):
                    if prefix_analysis_dags is not None:
                        save_simple_fig_to_path(what_if_dag, f"{prefix_analysis_dags}-{stage_name}-{dag_index}.png")
            else:
                colored_simple_dags = get_colored_simple_dags(self.intermediate_stages[stage_name],
                                                              with_reuse_coloring=False)
                for dag_index, what_if_dag in enumerate(colored_simple_dags):
                    if prefix_analysis_dags is not None:
                        save_simple_fig_to_path(what_if_dag, f"{prefix_analysis_dags}-{stage_name}-{dag_index}.png")
                colored_simple_dags = get_colored_simple_dags(self.intermediate_stages[stage_name],
                                                              with_reuse_coloring=True)
                for dag_index, what_if_dag in enumerate(colored_simple_dags):
                    if prefix_analysis_dags is not None:
                        save_simple_fig_to_path(what_if_dag, f"{prefix_analysis_dags}-CommonSubexpressionElimination0"
                                                             f"-{dag_index}.png")

    def save_optimised_what_if_dags_to_path(self, prefix_optimised_analysis_dag: str):
        """
        Save the extracted original DAG to a file
        """
        colored_combined_dag = get_final_optimized_combined_colored_simple_dag(
            self.intermediate_stages["optimize_patches_3_UdfSplitAndReuse"])
        if prefix_optimised_analysis_dag is not None:
            save_simple_fig_to_path(colored_combined_dag, f"{prefix_optimised_analysis_dag}-0.png")


@dataclasses.dataclass
class EstimationResults:
    """
    The class the PipelineExecutor returns when doing runtime estimation only
    """
    original_dag: networkx.DiGraph
    what_if_dags: List[tuple[List[PipelinePatch], networkx.DiGraph]]
    combined_optimized_dag: networkx.DiGraph
    runtime_info: RuntimeInfo
    dag_extraction_info: DagExtractionInfo
    pipeline_executor: any

    def print_estimate(self):
        """
        Print the estimated runtime
        """
        print(f"Estimated runtime is {self.runtime_info.what_if_optimized_estimated}ms.")


    def save_original_dag_to_path(self, prefix_original_dag: str):
        """
        Save the extracted original DAG to a file
        """
        save_fig_to_path(self.original_dag, f"{prefix_original_dag}.png")

    def save_what_if_dags_to_path(self, prefix_analysis_dags: str):
        """
        Save the generated What-If DAGs to a file
        """
        # FIXME: These functions should just redirect to the AnalysisResults one
        what_if_dags = [dag for patches, dag in self.what_if_dags]
        for dag_index, what_if_dag in enumerate(what_if_dags):
            if prefix_analysis_dags is not None:
                save_fig_to_path(what_if_dag, f"{prefix_analysis_dags}-{dag_index}.png")

    def save_optimised_what_if_dags_to_path(self, prefix_optimised_analysis_dag: str):
        """
        Save the extracted original DAG to a file
        """
        save_fig_to_path(self.combined_optimized_dag, f"{prefix_optimised_analysis_dag}.png")

    # def visualisation_phases_to_dags(self):
