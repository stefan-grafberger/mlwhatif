"""
The place where the Multi-Query optimization happens
"""
import logging
import time
from typing import Iterable

import networkx

from mlwhatif._analysis_results import AnalysisResults
from mlwhatif.optimization._operator_deletion_filter_push_up import OperatorDeletionFilterPushUp
from mlwhatif.optimization._simple_filter_addition_push_up import SimpleFilterAdditionPushUp
from mlwhatif.optimization._simple_projection_push_up import SimpleProjectionPushUp
from mlwhatif.optimization._udf_split_and_reuse import UdfSplitAndReuse
from mlwhatif.instrumentation._dag_node import DagNode

logger = logging.getLogger(__name__)


class MultiQueryOptimizer:
    """ Combines multiple DAGs and optimizes the joint plan """
    # pylint: disable=too-few-public-methods

    def __init__(self, pipeline_executor):
        self.pipeline_executor = pipeline_executor
        self.all_optimization_rules = [SimpleProjectionPushUp(pipeline_executor),
                                       SimpleFilterAdditionPushUp(pipeline_executor),
                                       OperatorDeletionFilterPushUp(pipeline_executor),
                                       UdfSplitAndReuse(pipeline_executor)]

    def create_optimized_plan(self, analysis_results: AnalysisResults, skip_optimizer=False) -> \
            AnalysisResults:
        """ Optimize and combine multiple given input DAGs """
        # pylint: disable=too-many-arguments
        estimate_original_runtime = self._estimate_runtime_of_dag(analysis_results.original_dag)
        logger.info(f"Estimated runtime of original DAG is {estimate_original_runtime}ms")
        analysis_results.runtime_info.original_pipeline_estimated = estimate_original_runtime

        multi_query_optimization_start = time.time()
        analysis_results = self._optimize_and_combine_dags(analysis_results, skip_optimizer)
        multi_query_optimization_duration = time.time() - multi_query_optimization_start
        logger.info(f'---RUNTIME: Multi-Query Optimization took {multi_query_optimization_duration * 1000} ms')
        analysis_results.runtime_info.what_if_query_optimization_duration = multi_query_optimization_duration * 1000

        return analysis_results

    def _optimize_and_combine_dags(self, analysis_results: AnalysisResults, skip_optimizer):
        """Here, the multi query optimization happens"""
        if skip_optimizer is False:
            logger.info(f"Performing Multi-Query Optimization")

            patches = [patches for (patches, _) in analysis_results.what_if_dags]
            big_execution_dag, what_if_dags = self._optimize_and_combine_dags_with_optimization(
                analysis_results.original_dag, patches)

            combined_estimated_runtimes = sum([self._estimate_runtime_of_dag(dag) for dag in what_if_dags])
            logger.info(f"Estimated unoptimized what-if runtime is {combined_estimated_runtimes}ms")
            analysis_results.runtime_info.what_if_unoptimized_estimated = combined_estimated_runtimes

            estimate_optimised_runtime = self._estimate_runtime_of_dag(big_execution_dag)
            logger.info(f"Estimated optimised what-if runtime is {estimate_optimised_runtime}ms")
            analysis_results.runtime_info.what_if_optimized_estimated = estimate_optimised_runtime

            estimated_saving = combined_estimated_runtimes - estimate_optimised_runtime
            logger.info(f"Estimated optimisation runtime saving is {estimated_saving}ms")
            analysis_results.runtime_info.what_if_optimization_saving_estimated = estimated_saving
        else:
            patches = [patches for patches, _ in analysis_results.what_if_dags]
            logger.warning("Skipping Multi-Query Optimization (instead, only combine execution DAGs)")
            big_execution_dag, what_if_dags = self._optimize_and_combine_dags_without_optimization(
                analysis_results.original_dag, patches)
            estimate_combined_runtime = self._estimate_runtime_of_dag(big_execution_dag)
            logger.info(f"Estimated unoptimised what-if runtime is {estimate_combined_runtime}ms")
            analysis_results.runtime_info.what_if_unoptimized_estimated = estimate_combined_runtime

        analysis_results.combined_optimized_dag = big_execution_dag
        analysis_results.what_if_dags = list(zip(patches, what_if_dags))
        return analysis_results

    def _optimize_and_combine_dags_without_optimization(self, original_dag, patches):
        """The baseline of not performing any optimizations"""
        what_if_dags = []
        for patch_set in patches:
            what_if_dag = original_dag.copy()
            for patch in patch_set:
                patch.apply(what_if_dag)
            what_if_dags.append(what_if_dag)
        self._make_all_nodes_unique(what_if_dags)
        big_execution_dag = networkx.compose_all(what_if_dags)
        return big_execution_dag, what_if_dags

    def _optimize_and_combine_dags_with_optimization(self, original_dag, patches):
        """Here, the actual optimization happens"""
        for rule in self.all_optimization_rules:
            original_dag, patches = rule.optimize_dag(original_dag, patches)
        for rule in self.all_optimization_rules:
            patches = rule.optimize_patches(original_dag, patches)
        what_if_dags = []
        for patch_set in patches:
            what_if_dag = original_dag.copy()
            all_nodes_needing_recomputation = set()
            for patch in patch_set:
                patch.apply(what_if_dag, self.pipeline_executor)
            for patch in patch_set:
                all_nodes_needing_recomputation.update(patch.get_nodes_needing_recomputation(original_dag,
                                                                                             what_if_dag))
            self._generate_unique_ids_for_selected_nodes(what_if_dag, all_nodes_needing_recomputation)
            what_if_dags.append(what_if_dag)
        if len(what_if_dags) != 0:
            big_execution_dag = networkx.compose_all(what_if_dags)
        else:
            big_execution_dag = networkx.DiGraph()
        return big_execution_dag, what_if_dags

    def _make_all_nodes_unique(self, what_if_dags):
        """We need to give all nodes new ids to combine DAGs without reusing results."""
        for dag in what_if_dags:
            self._generate_unique_ids_for_selected_nodes(dag, list(dag.nodes))

    def _generate_unique_ids_for_selected_nodes(self, dag: networkx.DiGraph, nodes_to_recompute: Iterable[DagNode]):
        """ This gives new node_ids to all reachable nodes given some input node """
        what_if_node_set = set(dag.nodes)
        for node_to_recompute in nodes_to_recompute:
            if node_to_recompute in what_if_node_set:  # condition required because node may be removed
                replacement_node = DagNode(self.pipeline_executor.get_next_op_id(),
                                           node_to_recompute.code_location,
                                           node_to_recompute.operator_info,
                                           node_to_recompute.details,
                                           node_to_recompute.optional_code_info,
                                           node_to_recompute.processing_func,
                                           node_to_recompute.make_classifier_func)
                dag.add_node(replacement_node)
                for parent_node in dag.predecessors(node_to_recompute):
                    edge_data = dag.get_edge_data(parent_node, node_to_recompute)
                    dag.add_edge(parent_node, replacement_node, **edge_data)
                for child_node in dag.successors(node_to_recompute):
                    edge_data = dag.get_edge_data(node_to_recompute, child_node)
                    dag.add_edge(replacement_node, child_node, **edge_data)
                dag.remove_node(node_to_recompute)

    @staticmethod
    def _estimate_runtime_of_dag(dag: networkx.DiGraph):
        # TODO: Currently, we treat inserted nodes as runtime 0. Especially string data corruptions can be expensive.
        #  We will need to update the what-if analyses accordingly at some point to provide estimates.
        runtimes = [node.details.optimizer_info.runtime for node in dag.nodes if node.details.optimizer_info and
                    node.details.optimizer_info.runtime]
        return sum(runtimes)
