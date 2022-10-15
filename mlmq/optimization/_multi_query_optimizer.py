"""
The place where the Multi-Query optimization happens
"""
import logging
import time
from typing import Iterable, List

import networkx

from mlmq._analysis_results import AnalysisResults
from mlmq.optimization._operator_deletion_filter_push_up import OperatorDeletionFilterPushUp
from mlmq.optimization._query_optimization_rules import QueryOptimizationRule
from mlmq.optimization._simple_filter_addition_push_up import SimpleFilterAdditionPushUp
from mlmq.optimization._simple_projection_push_up import SimpleProjectionPushUp
from mlmq.optimization._udf_split_and_reuse import UdfSplitAndReuse
from mlmq.instrumentation._dag_node import DagNode

logger = logging.getLogger(__name__)


class MultiQueryOptimizer:
    """ Combines multiple DAGs and optimizes the joint plan """

    # pylint: disable=too-few-public-methods

    def __init__(self, pipeline_executor, force_optimization_rules: List[QueryOptimizationRule] or None):
        self.pipeline_executor = pipeline_executor
        if force_optimization_rules is None:
            self.all_optimization_rules = [SimpleProjectionPushUp(pipeline_executor),
                                           SimpleFilterAdditionPushUp(pipeline_executor),
                                           OperatorDeletionFilterPushUp(pipeline_executor),
                                           UdfSplitAndReuse(pipeline_executor)]
        else:
            self.all_optimization_rules = force_optimization_rules

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
                analysis_results.original_dag.copy(), patches)

            # TODO: This comparison may be a bit unfair in case of expensive UDFs because the UDF split reuse
            #  DAG rewriting is applied before measuring the runtime of the DAGs without reuse. For now,
            #  the corruption and index selection functions are considered to have a runtime of zero,
            #  so it does not matter at the moment, but it might in the future once this changes.
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
            logger.warning("Skipping Multi-Query Optimization (instead, only applying patches)")
            what_if_dags = self._optimize_and_combine_dags_without_optimization(
                analysis_results.original_dag, patches)
            estimate_combined_runtime = sum(self._estimate_runtime_of_dag(what_if_dag) for what_if_dag in what_if_dags)
            big_execution_dag = None
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
                patch.apply(what_if_dag, self.pipeline_executor)
            what_if_dags.append(what_if_dag)
        return what_if_dags

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
            self._generate_unique_ids_for_selected_nodes(dag, set(dag.nodes))

    def _generate_unique_ids_for_selected_nodes(self, dag: networkx.DiGraph, nodes_to_recompute: Iterable[DagNode]):
        """ This gives new node_ids to all reachable nodes given some input node """
        what_if_node_set = {node.node_id for node in list(dag.nodes)}
        nodes_to_recompute = {node.node_id for node in nodes_to_recompute}
        # condition required because node may be removed
        nodes_requiring_new_id = what_if_node_set.intersection(nodes_to_recompute)

        def generate_new_node_ids_if_required(dag_node: DagNode) -> DagNode:
            if dag_node.node_id in nodes_requiring_new_id:
                result = DagNode(self.pipeline_executor.get_next_op_id(),
                                 dag_node.code_location,
                                 dag_node.operator_info,
                                 dag_node.details,
                                 dag_node.optional_code_info,
                                 dag_node.processing_func,
                                 dag_node.make_classifier_func)

            else:
                result = dag_node
            return result

        # noinspection PyTypeChecker
        networkx.relabel_nodes(dag, generate_new_node_ids_if_required, copy=False)

    @staticmethod
    def _estimate_runtime_of_dag(dag: networkx.DiGraph):
        # TODO: Currently, we treat inserted nodes as runtime 0. Especially string data corruptions can be expensive.
        #  We will need to update the what-if analyses accordingly at some point to provide estimates.
        runtimes = [node.details.optimizer_info.runtime for node in dag.nodes if node.details.optimizer_info and
                    node.details.optimizer_info.runtime]
        return sum(runtimes)
