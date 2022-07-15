"""
Util functions to make writing What-If Analyses easier
"""
from typing import Tuple, Iterable

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


def add_new_node_between_nodes(dag: networkx.DiGraph, new_node: DagNode, dag_location: Tuple[DagNode, DagNode]):
    """Add a new node between two chosen nodes"""
    parent, child = dag_location
    edge_data = dag.get_edge_data(parent, child)
    dag.remove_edge(parent, child)
    dag.add_edge(parent, new_node, arg_index=0)
    dag.add_edge(new_node, child, **edge_data)


def filter_estimator_transformer_edges(parent, child):
    """Filter edges that are not relevant to the actual data flow but only to estimator/transformer state"""
    is_transformer_edge = ((parent.operator_info.operator == OperatorType.TRANSFORMER
                           and ": fit_transform" in parent.details.description
                           and child.operator_info.operator == OperatorType.TRANSFORMER
                           and ": transform" in child.details.description) or
                           (parent.operator_info.operator == OperatorType.ESTIMATOR
                           and child.operator_info.operator == OperatorType.SCORE))
    return not is_transformer_edge


def find_first_op_modifying_a_column(dag, search_start_node: DagNode, column_name: str, test_not_train: bool):
    """Find DagNodes in the DAG by OperatorType"""
    if test_not_train is True:
        dag_to_consider = networkx.subgraph_view(dag, filter_edge=filter_estimator_transformer_edges)
    else:
        dag_to_consider = dag
    nodes_to_search = list(networkx.ancestors(dag_to_consider, search_start_node))
    project_modify_matches = [node for node in nodes_to_search
                              if node.operator_info.operator == OperatorType.PROJECTION_MODIFY
                              and node.details.description == f"modifies ['{column_name}']"]
    # TODO: Using the description string is not very clean
    if len(project_modify_matches) != 0:
        sorted_matches = sorted(project_modify_matches, key=lambda dag_node: dag_node.node_id)
        return sorted_matches[0]
    if test_not_train is True:
        transformer_matches = [node for node in nodes_to_search
                               if node.operator_info.operator == OperatorType.TRANSFORMER
                               and ": transform" in node.details.description
                               and column_name in list(dag.predecessors(node))[1].details.columns]
    else:
        transformer_matches = [node for node in nodes_to_search
                               if node.operator_info.operator == OperatorType.TRANSFORMER
                               and ": fit_transform" in node.details.description
                               and column_name in list(dag.predecessors(node))[0].details.columns]
    if len(transformer_matches) >= 1:
        # Can be two for example in the COMPAS pipeline when there is a SimpleImputer first
        sorted_transformer_matches = sorted(transformer_matches, key=lambda dag_node: dag_node.node_id)
        return sorted_transformer_matches[0]

    # If no node changes the column, we can apply the corruption directly before the estimator
    return search_start_node
