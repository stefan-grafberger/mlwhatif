"""
Util functions to make writing What-If Analyses easier
"""
import networkx

from mlwhatif import OperatorContext, OperatorType
from mlwhatif.instrumentation._dag_node import DagNode, DagNodeDetails
from mlwhatif.instrumentation._pipeline_executor import singleton


def find_nodes_by_type(new_dag: networkx.DiGraph, operator_type: OperatorType):
    """Find DagNodes in the DAG by OperatorType"""
    return [node for node in new_dag.nodes if node.operator_info.operator == operator_type]


def add_intermediate_extraction_after_node(dag: networkx.DiGraph, dag_node: DagNode, label: str):
    """Add a new node behind some given node to extract the intermediate result of that given node"""

    def extract_intermediate(intermediate_value):
        singleton.labels_to_extracted_plan_results[label] = intermediate_value
        return intermediate_value

    new_extraction_node = DagNode(singleton.get_next_op_id(),
                                  dag_node.code_location,
                                  OperatorContext(OperatorType.EXTRACT_RESULT, None),
                                  DagNodeDetails(None, dag_node.details.columns),
                                  None,
                                  extract_intermediate)
    add_new_node_after_node(dag, new_extraction_node, dag_node)


def add_new_node_after_node(dag: networkx.DiGraph, new_node: DagNode, dag_node: DagNode):
    """Add a new node behind some given node to extract the intermediate result of that given node"""
    dag.add_node(new_node)
    children_before_modifications = list(dag.successors(dag_node))
    for child_node in children_before_modifications:
        edge_data = dag.get_edge_data(dag_node, child_node)
        dag.add_edge(new_node, child_node, **edge_data)
        dag.remove_edge(dag_node, child_node)
    dag.add_edge(dag_node, new_node)


def find_first_op_modifying_a_column(dag: networkx.DiGraph, column_name: str, test_not_train: bool):
    """Find DagNodes in the DAG by OperatorType"""
    # TODO: Performance optimizations if necessary, testing
    project_modify_matches = [node for node in dag.nodes
                              if node.operator_info.operator == OperatorType.PROJECTION_MODIFY
                              and node.details.description == f"modifies ['{column_name}']"]
    # TODO: This is not very robust and clean, also needs testing
    if len(project_modify_matches) != 0:
        sorted_matches = sorted(project_modify_matches, key=lambda dag_node: dag_node.node_id)
        return sorted_matches[0]  # Test this, is it 0 or -1?
    if test_not_train is True:
        transformer_matches = [node for node in dag.nodes
                               if node.operator_info.operator == OperatorType.TRANSFORMER
                               and ": transform" in node.details.description
                               and column_name in list(dag.predecessors(node))[1].details.columns]
    else:
        transformer_matches = [node for node in dag.nodes
                               if node.operator_info.operator == OperatorType.TRANSFORMER
                               and ": fit_transform" in node.details.description
                               and column_name in list(dag.predecessors(node))[0].details.columns]
    assert len(transformer_matches) >= 1
    # Can be two for example in the COMPAS pipeline when there is a SimpleImputer first
    sorted_transformer_matches = sorted(transformer_matches, key=lambda dag_node: dag_node.node_id)
    return sorted_transformer_matches[0]


def mark_nodes_to_recompute_after_changed_node(dag: networkx.DiGraph, changed_dag_node: DagNode):
    """ This gives new node_ids to all reachable nodes given some input node """
    nodes_needing_recomputation = list(networkx.descendants(dag, changed_dag_node))
    for node_to_recompute in nodes_needing_recomputation:
        replacement_node = DagNode(singleton.get_next_op_id(),
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
