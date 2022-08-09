"""
Util functions to make writing What-If Analyses easier
"""
import logging
from typing import Tuple, Iterable

import networkx

from mlwhatif.instrumentation._operator_types import OperatorType
from mlwhatif.instrumentation._dag_node import DagNode

logger = logging.getLogger(__name__)


def find_nodes_by_type(new_dag: networkx.DiGraph, operator_type: OperatorType):
    """Find DagNodes in the DAG by OperatorType"""
    return [node for node in new_dag.nodes if node.operator_info.operator == operator_type]


def add_new_node_after_node(dag: networkx.DiGraph, new_node: DagNode, dag_node: DagNode, arg_index=0):
    """Add a new node behind some given node to extract the intermediate result of that given node"""
    dag.add_node(new_node)
    children_before_modifications = list(dag.successors(dag_node))
    for child_node in children_before_modifications:
        edge_data = dag.get_edge_data(dag_node, child_node)
        dag.add_edge(new_node, child_node, **edge_data)
        dag.remove_edge(dag_node, child_node)
    dag.add_edge(dag_node, new_node, arg_index=arg_index)


def get_sorted_parent_nodes(dag: networkx.DiGraph, first_op_requiring_corruption):
    """Get the parent nodes of a node sorted by arg_index"""
    operator_parent_nodes = list(dag.predecessors(first_op_requiring_corruption))
    parent_nodes_with_arg_index = [(parent_node, dag.get_edge_data(parent_node, first_op_requiring_corruption))
                                   for parent_node in operator_parent_nodes]
    parent_nodes_with_arg_index = sorted(parent_nodes_with_arg_index, key=lambda x: x[1]['arg_index'])
    operator_parent_nodes = [node for (node, _) in parent_nodes_with_arg_index]
    return operator_parent_nodes


def replace_node(dag: networkx.DiGraph, dag_node_to_be_replaced: DagNode, new_node: DagNode):
    """Replace a given DAG node with a new node"""
    dag.add_node(new_node)

    children_before_modifications = list(dag.successors(dag_node_to_be_replaced))
    for child_node in children_before_modifications:
        edge_data = dag.get_edge_data(dag_node_to_be_replaced, child_node)
        dag.add_edge(new_node, child_node, **edge_data)

    parents_before_modifications = list(dag.predecessors(dag_node_to_be_replaced))
    for parent_node in parents_before_modifications:
        edge_data = dag.get_edge_data(parent_node, dag_node_to_be_replaced)
        dag.add_edge(parent_node, new_node, **edge_data)

    dag.remove_node(dag_node_to_be_replaced)


def remove_node(dag: networkx.DiGraph, operator_to_remove: DagNode):
    """Replace a node from a given DAG"""
    # TODO: Not totally sure yet if this work as intended and if this parent is the right node always
    parent = get_sorted_parent_nodes(dag, operator_to_remove)[0]
    children_before_modifications = list(dag.successors(operator_to_remove))
    for child_node in children_before_modifications:
        edge_data = dag.get_edge_data(operator_to_remove, child_node)
        dag.add_edge(parent, child_node, **edge_data)
    dag.remove_node(operator_to_remove)


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
                            and child.operator_info.operator == OperatorType.PREDICT))
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


def find_dag_location_for_first_op_modifying_column(column, dag, test_not_train):
    """Find out between which two nodes to apply the corruption"""
    search_start_node = find_train_or_test_pipeline_part_end(dag, test_not_train)
    first_op_requiring_corruption = find_first_op_modifying_a_column(dag, search_start_node, column, test_not_train)
    operator_parent_nodes = get_sorted_parent_nodes(dag, first_op_requiring_corruption)
    first_op_requiring_corruption, operator_to_apply_corruption_after = \
        find_where_to_apply_corruption_exactly(dag, first_op_requiring_corruption, operator_parent_nodes)
    return operator_to_apply_corruption_after, first_op_requiring_corruption


def find_dag_location_for_data_patch(columns, dag, train_not_test) -> tuple[any, bool]:
    """Find out between which two nodes to apply the corruption"""
    train_search_start_node = find_train_or_test_pipeline_part_end(dag, False)
    test_search_start_node = find_train_or_test_pipeline_part_end(dag, True)
    dag_to_consider = networkx.subgraph_view(dag, filter_edge=filter_estimator_transformer_edges)

    train_nodes_to_search = set(networkx.ancestors(dag_to_consider, train_search_start_node))
    test_nodes_to_search = set(networkx.ancestors(dag_to_consider, test_search_start_node))

    if train_not_test is True:
        nodes_to_search = train_nodes_to_search.difference(test_nodes_to_search)
    else:
        nodes_to_search = test_nodes_to_search.difference(train_nodes_to_search)

    columns = set(columns)
    matches = [node for node in nodes_to_search if columns.issubset(set(node.details.columns)) and
               node.operator_info.operator not in {OperatorType.PROJECTION, OperatorType.SUBSCRIPT}]

    is_before_split = False
    if len(matches) == 0:
        is_before_split = True

        nodes_to_search = train_nodes_to_search.intersection(test_nodes_to_search)
        matches = [node for node in nodes_to_search
                   if node.details.columns and columns.issubset(set(node.details.columns))
                   and node.operator_info.operator not in {OperatorType.PROJECTION, OperatorType.SUBSCRIPT}]

        if len(matches) == 0:
            raise Exception(f"Columns {columns} not present in DAG!")

    sorted_matches = sorted(matches, key=lambda dag_node: dag_node.node_id)
    first_op_requiring_corruption = sorted_matches[0]
    return first_op_requiring_corruption, is_before_split


def find_train_or_test_pipeline_part_end(dag, test_not_train):
    """We want to start at the end of the pipeline to find the relevant train or test operations"""
    if test_not_train is True:
        search_start_nodes = find_nodes_by_type(dag, OperatorType.PREDICT)
        if len(search_start_nodes) != 1:
            raise Exception("Currently, DataCorruption only supports pipelines with exactly one predict call "
                            "for the test set!")

        search_start_node = search_start_nodes[0]
    else:
        search_start_nodes = find_nodes_by_type(dag, OperatorType.ESTIMATOR)
        if len(search_start_nodes) != 1:
            raise Exception("Currently, DataCorruption only supports pipelines with exactly one estimator!")
        search_start_node = search_start_nodes[0]
    return search_start_node


def get_sorted_children_nodes(dag: networkx.DiGraph, first_op_requiring_corruption):
    """Get the parent nodes of a node sorted by arg_index"""
    operator_child_nodes = list(dag.successors(first_op_requiring_corruption))
    sorted_operator_child_nodes = sorted(operator_child_nodes, key=lambda x: x.node_id)
    return sorted_operator_child_nodes


def find_where_to_apply_corruption_exactly(dag, first_op_requiring_corruption, operator_parent_nodes):
    """
    We know which operator requires the corruption to be present already; now we need to decide between which
    parent node and the current node we need to insert the corruption node.
    """
    if first_op_requiring_corruption.operator_info.operator == OperatorType.TRANSFORMER:
        operator_to_apply_corruption_after = operator_parent_nodes[-1]
    elif first_op_requiring_corruption.operator_info.operator == OperatorType.PREDICT:
        operator_to_apply_corruption_after = operator_parent_nodes[1]
    elif first_op_requiring_corruption.operator_info.operator == OperatorType.ESTIMATOR:
        operator_to_apply_corruption_after = operator_parent_nodes[0]
    elif first_op_requiring_corruption.operator_info.operator == OperatorType.PROJECTION_MODIFY:
        project_modify_parent_a = operator_parent_nodes[0]
        project_modify_parent_b = operator_parent_nodes[-1]
        # We want to introduce the change before all subscript behavior
        operator_to_apply_corruption_after = networkx.lowest_common_ancestor(dag, project_modify_parent_a,
                                                                             project_modify_parent_b)
        sorted_successors = get_sorted_children_nodes(dag, operator_to_apply_corruption_after)
        first_op_requiring_corruption = sorted_successors[0]
    else:
        raise Exception("Either a column was changed by a transformer or project_modify or we can apply"
                        "the corruption right before the estimator operation!")
    return first_op_requiring_corruption, operator_to_apply_corruption_after
