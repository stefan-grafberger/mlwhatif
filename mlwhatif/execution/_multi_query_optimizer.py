"""
The place where the Multi-Query optimization happens
"""
import logging
import time
from typing import Iterable

import networkx

from mlwhatif.execution._patches import Patch
from mlwhatif.instrumentation._dag_node import DagNode
from mlwhatif.visualisation import save_fig_to_path

logger = logging.getLogger(__name__)


class MultiQueryOptimizer:
    """ Combines multiple DAGs and optimizes the joint plan """
    # pylint: disable=too-few-public-methods

    def __init__(self, pipeline_executor):
        self.pipeline_executor = pipeline_executor

    def create_optimized_plan(self, original_dag: networkx.DiGraph, patches: Iterable[Iterable[Patch]],
                              prefix_analysis_dags: str or None = None,
                              prefix_optimised_analysis_dag: str or None = None,
                              skip_optimizer=False) -> \
            networkx.DiGraph:
        """ Optimize and combine multiple given input DAGs """
        # pylint: disable=too-many-arguments
        estimate_original_runtime = self._estimate_runtime_of_dag(original_dag)
        logger.info(f"Estimated runtime of original DAG is {estimate_original_runtime}ms")

        multi_query_optimization_start = time.time()
        big_execution_dag, image_generation_duration = self._optimize_and_combine_dags(original_dag, skip_optimizer,
                                                                                       patches, prefix_analysis_dags)
        multi_query_optimization_duration = time.time() - multi_query_optimization_start - image_generation_duration
        logger.info(f'---RUNTIME: Multi-Query Optimization took {multi_query_optimization_duration * 1000} ms')

        image_generation_start = time.time()
        if prefix_optimised_analysis_dag is not None:
            save_fig_to_path(big_execution_dag, f"{prefix_optimised_analysis_dag}.png")
        image_generation_duration += time.time() - image_generation_start

        return big_execution_dag

    def _optimize_and_combine_dags(self, original_dag, skip_optimizer, patches, prefix_analysis_dags: str or None):
        """Here, the multi query optimization happens"""
        if skip_optimizer is False:
            logger.info(f"Performing Multi-Query Optimization")
            big_execution_dag, what_if_dags = self._optimize_and_combine_dags_with_optimization(original_dag, patches)

            combined_estimated_runtimes = sum([self._estimate_runtime_of_dag(dag) for dag in what_if_dags])
            logger.info(f"Estimated unoptimized what-if runtime is {combined_estimated_runtimes}ms")

            estimate_optimised_runtime = self._estimate_runtime_of_dag(big_execution_dag)
            logger.info(f"Estimated optimised what-if runtime is {estimate_optimised_runtime}ms")

            estimated_saving = combined_estimated_runtimes - estimate_optimised_runtime
            logger.info(f"Estimated optimisation runtime saving is {estimated_saving}ms")
        else:
            logger.warning("Skipping Multi-Query Optimization (instead, only combine execution DAGs)")
            big_execution_dag, what_if_dags = self._optimize_and_combine_dags_without_optimization(original_dag,
                                                                                                   patches)
        image_generation_duration = self._save_what_if_dags_as_figs_to_path(prefix_analysis_dags, what_if_dags)
        return big_execution_dag, image_generation_duration

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
        what_if_dags = []
        for patch_set in patches:
            what_if_dag = original_dag.copy()
            all_nodes_needing_recomputation = set()
            for patch in patch_set:
                patch.apply(what_if_dag)
            for patch in patch_set:
                all_nodes_needing_recomputation.update(patch.get_nodes_needing_recomputation(original_dag,
                                                                                             what_if_dag))
            self._generate_unique_ids_for_selected_nodes(what_if_dag, all_nodes_needing_recomputation)
            what_if_dags.append(what_if_dag)
        big_execution_dag = networkx.compose_all(what_if_dags)
        return big_execution_dag, what_if_dags

    @staticmethod
    def _save_what_if_dags_as_figs_to_path(prefix_analysis_dags, what_if_dags):
        """Store the what-if DAGs as figs."""
        image_generation_start = time.time()
        for dag_index, what_if_dag in enumerate(what_if_dags):
            if prefix_analysis_dags is not None:
                save_fig_to_path(what_if_dag, f"{prefix_analysis_dags}-{dag_index}.png")
        image_generation_duration = time.time() - image_generation_start
        return image_generation_duration

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
                                           node_to_recompute.processing_func)
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
