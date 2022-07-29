"""
The place where the Multi-Query optimization happens
"""
import logging
from typing import List, Iterable
import networkx
from mlwhatif.instrumentation._dag_node import DagNode

logger = logging.getLogger(__name__)


class MultiQueryOptimizer:
    """ Combines multiple DAGs and optimizes the joint plan """
    # pylint: disable=too-few-public-methods

    def __init__(self, pipeline_executor):
        self.pipeline_executor = pipeline_executor

    def create_optimized_plan(self, original_dag: networkx.DiGraph, what_if_dags: List[networkx.DiGraph]) -> networkx.DiGraph:
        """ Optimize and combine multiple given input DAGs """
        # TODO: In the future, we will need to update this once we have more sophisticated optimisations
        estimate_original_runtime = self._estimate_runtime_of_dag(original_dag)
        logger.info(f"Estimated runtime of original DAG is {estimate_original_runtime}ms")
        self.make_nodes_depending_on_changed_nodes_unique(original_dag, what_if_dags)

        combined_estimated_runtimes = sum([self._estimate_runtime_of_dag(dag) for dag in what_if_dags])
        logger.info(f"Estimated unoptimized what-if runtime is {combined_estimated_runtimes}ms")

        big_execution_dag = networkx.compose_all(what_if_dags)

        estimate_optimised_runtime = self._estimate_runtime_of_dag(big_execution_dag)
        logger.info(f"Estimated optimised what-if runtime is {estimate_optimised_runtime}ms")

        estimated_saving = combined_estimated_runtimes - estimate_optimised_runtime
        logger.info(f"Estimated optimisation runtime saving is {estimated_saving}ms")

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
