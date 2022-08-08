"""
The place where the Multi-Query optimization happens
"""
import logging
import time
from typing import Iterable

import networkx

from mlwhatif.instrumentation._operator_types import OperatorType
from mlwhatif.analysis._analysis_utils import find_dag_location_for_data_patch, add_new_node_after_node, \
    find_nodes_by_type, replace_node
from mlwhatif.execution._patches import Patch, DataPatch, ModelPatch, PipelinePatch, DataFiltering, DataTransformer, \
    AppendNodeAfterOperator
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
        # TODO: In the future, we will need to update this once we have more sophisticated optimisations
        estimate_original_runtime = self._estimate_runtime_of_dag(original_dag)
        logger.info(f"Estimated runtime of original DAG is {estimate_original_runtime}ms")

        what_if_dags = []
        for patch_set in patches:
            what_if_dag = original_dag.copy()
            for patch in patch_set:
                if isinstance(patch, DataPatch):
                    if isinstance(patch, DataFiltering):
                        location = find_dag_location_for_data_patch(patch.only_reads_column, what_if_dag,
                                                                    patch.train_not_test)
                        add_new_node_after_node(what_if_dag, patch.filter_operator, location)
                    elif isinstance(patch, DataTransformer):
                        train_location = find_dag_location_for_data_patch([patch.modifies_column], what_if_dag, True)
                        test_location = find_dag_location_for_data_patch([patch.modifies_column], what_if_dag, True)
                        add_new_node_after_node(what_if_dag, patch.fit_transform_operator, train_location)
                        if train_location != test_location:
                            add_new_node_after_node(what_if_dag, patch.transform_operator, test_location)
                            what_if_dag.add_edge(patch.fit_transform_operator, patch.transform_operator, arg_index=0)
                    else:
                        raise Exception(f"Unknown DataPatch type: {type(patch).__name__}!")
                elif isinstance(patch, ModelPatch):
                    estimator_nodes = find_nodes_by_type(what_if_dag, OperatorType.ESTIMATOR)
                    if len(estimator_nodes) != 1:
                        raise Exception(
                            "Currently, DataCorruption only supports pipelines with exactly one estimator!")
                    estimator_node = estimator_nodes[0]
                    replace_node(what_if_dag, estimator_node, patch.replace_with_node)
                elif isinstance(patch, PipelinePatch):
                    if isinstance(patch, AppendNodeAfterOperator):
                        add_new_node_after_node(what_if_dag, patch.operator_to_add_node_after, patch.node_to_insert)
                    else:
                        raise Exception(f"Unknown PipelinePatch type: {type(patch).__name__}!")
                else:
                    raise Exception(f"Unknown patch type: {type(patch).__name__}!")
            what_if_dags.append(what_if_dag)

        self.make_nodes_depending_on_changed_nodes_unique(original_dag, what_if_dags)

        combined_estimated_runtimes = sum([self._estimate_runtime_of_dag(dag) for dag in what_if_dags])
        logger.info(f"Estimated unoptimized what-if runtime is {combined_estimated_runtimes}ms")

        if skip_optimizer is False:
            logger.info(f"Performing Multi-Query Optimization")
            multi_query_optimization_start = time.time()

            # Actual code
            big_execution_dag = networkx.compose_all(what_if_dags)

            estimate_optimised_runtime = self._estimate_runtime_of_dag(big_execution_dag)
            logger.info(f"Estimated optimised what-if runtime is {estimate_optimised_runtime}ms")

            estimated_saving = combined_estimated_runtimes - estimate_optimised_runtime
            logger.info(f"Estimated optimisation runtime saving is {estimated_saving}ms")

            multi_query_optimization_duration = time.time() - multi_query_optimization_start
            logger.info(f'---RUNTIME: Multi-Query Optimization took {multi_query_optimization_duration * 1000} ms')
            logger.info(f"Executing generated plan")
            if prefix_optimised_analysis_dag is not None:
                save_fig_to_path(big_execution_dag, f"{prefix_optimised_analysis_dag}.png")
        else:
            logger.warning("Skipping Multi-Query Optimization")
            big_execution_dag = networkx.disjoint_union_all(what_if_dags)  # TODO: Does this work?

        # TODO: More optimizations, maybe create some optimization rule interface that what-if analyses can use
        #  to specify analysis-specific optimizations
        return big_execution_dag

    def make_nodes_depending_on_changed_nodes_unique(self, original_dag, what_if_dags):
        """We need to give new ids to all nodes that require recomputation because some parent node changed."""
        original_ids = set(node.node_id for node in list(original_dag.nodes))
        for dag in what_if_dags:
            all_nodes_needing_recomputation = set()
            # added nodes
            added_nodes = [node for node in list(dag.nodes) if node.node_id not in original_ids]
            for added_node in added_nodes:
                local_nodes_needing_recomputation = set(networkx.descendants(dag, added_node))
                all_nodes_needing_recomputation.update(local_nodes_needing_recomputation)
            # removed_nodes
            ids_in_modified_dag = set(node.node_id for node in list(dag.nodes))
            removed_nodes = [node for node in list(original_dag.nodes) if node.node_id not in ids_in_modified_dag]
            for removed_node in removed_nodes:
                original_nodes_needing_recomputation = set(networkx.descendants(original_dag, removed_node))
                all_nodes_needing_recomputation.update(original_nodes_needing_recomputation)

            self.generate_unique_ids_for_selected_nodes(dag, all_nodes_needing_recomputation)

    def generate_unique_ids_for_selected_nodes(self, dag: networkx.DiGraph, nodes_to_recompute: Iterable[DagNode]):
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
