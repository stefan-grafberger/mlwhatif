"""
The place where the Multi-Query optimization happens
"""
from typing import List, Iterable
import networkx
from mlwhatif.instrumentation._dag_node import DagNode


class MultiQueryOptimizer:
    """ Combines multiple DAGs and optimizes the joint plan """
    # pylint: disable=too-few-public-methods

    def __init__(self, pipeline_executor):
        self.pipeline_executor = pipeline_executor

    def create_optimized_plan(self, original_dag: networkx.DiGraph, what_if_dags: List[networkx.DiGraph]) -> networkx.DiGraph:
        """ Optimize and combine multiple given input DAGs """
        # TODO: In the future, we will need to update this once we have more sophisticated optimisations
        self.make_nodes_depending_on_changed_nodes_unique(original_dag, what_if_dags)

        big_execution_dag = networkx.compose_all(what_if_dags)
        # TODO: More optimizations, maybe create some optimization rule interface that what-if analyses can use
        #  to specify analysis-specific optimizations
        return big_execution_dag

    def make_nodes_depending_on_changed_nodes_unique(self, original_dag, what_if_dags):
        """We need to give new ids to all nodes that require recomputation because some parent node changed."""
        original_ids = set(node.node_id for node in list(original_dag.nodes))
        for dag in what_if_dags:
            added_nodes = [node for node in list(dag.nodes) if node.node_id not in original_ids]
            all_nodes_needing_recomputation = set()
            for added_node in added_nodes:
                local_nodes_needing_recomputation = set(networkx.descendants(dag, added_node))
                all_nodes_needing_recomputation.update(local_nodes_needing_recomputation)
            self.generate_unique_ids_for_selected_nodes(dag, all_nodes_needing_recomputation)

    def generate_unique_ids_for_selected_nodes(self, dag: networkx.DiGraph, nodes_to_recompute: Iterable[DagNode]):
        """ This gives new node_ids to all reachable nodes given some input node """
        for node_to_recompute in nodes_to_recompute:
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
