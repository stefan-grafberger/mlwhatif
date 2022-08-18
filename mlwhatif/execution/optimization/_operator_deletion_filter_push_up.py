"""
Simple Projection push-up optimization that ignores that data corruptions only corrupt subsets and that
it is possible to corrupt the whole set at once and then only sample from the corrupted DF.
"""
from typing import List

import networkx

from mlwhatif.instrumentation._dag_node import DagNode
from mlwhatif.instrumentation._operator_types import OperatorType
from mlwhatif.analysis._analysis_utils import find_dag_location_for_new_filter_on_column, get_sorted_parent_nodes, \
    get_sorted_children_nodes
from mlwhatif.execution._patches import Patch, OperatorRemoval
from mlwhatif.execution.optimization._query_optimization_rules import QueryOptimizationRule


class OperatorDeletionFilterPushUp(QueryOptimizationRule):
    """ Combines multiple DAGs and optimizes the joint plan """

    # pylint: disable=too-few-public-methods

    def __init__(self, pipeline_executor):
        self._pipeline_executor = pipeline_executor

    def optimize_dag(self, dag: networkx.DiGraph, patches: List[List[Patch]]) -> networkx.DiGraph:
        selectivity_and_filters_to_push_up = []

        for pipeline_variant_patches in patches:
            for patch in pipeline_variant_patches:
                if isinstance(patch, OperatorRemoval) and \
                        patch.operator_to_remove.operator_info.operator == OperatorType.SELECTION:
                    parent = list(dag.predecessors(patch.operator_to_remove))[0]
                    parent_row_count = parent.details.optimizer_info.shape[0]
                    current_row_count = patch.operator_to_remove.details.optimizer_info.shape[0]
                    selectivity = current_row_count / parent_row_count
                    selectivity_and_filters_to_push_up.append((selectivity, patch))
                    # calculate selectivities

        # Sort filters by selectivity, the ones with the highest selectivity should be moved up first
        selectivity_and_filters_to_push_up = sorted(selectivity_and_filters_to_push_up, key=lambda x: x[0])

        for _, filter_removal_patch in selectivity_and_filters_to_push_up:
            using_columns_for_filter = self._get_columns_required_for_filter_eval(dag, filter_removal_patch)

            operator_to_add_node_after_train = find_dag_location_for_new_filter_on_column(
                using_columns_for_filter, dag, True)
            operator_to_add_node_after_test = find_dag_location_for_new_filter_on_column(
                using_columns_for_filter, dag, False)

            dag = self._move_filters_and_duplicate_if_required(dag, filter_removal_patch,
                                                               operator_to_add_node_after_test,
                                                               operator_to_add_node_after_train)
        return dag

    def _move_filters_and_duplicate_if_required(self, dag, filter_removal_patch, operator_to_add_node_after_test,
                                                operator_to_add_node_after_train):
        """The part where the actual filter push-up happens"""
        if filter_removal_patch.before_train_test_split is True:
            # Modifiy DAG, duplicate deletion node not needed probably
            operator_after_which_cutoff_required_train = filter_removal_patch \
                .filter_get_operator_after_which_cutoff_required(dag, filter_removal_patch.operator_to_remove)

            dag, operator_after_which_cutoff_required_test = \
                self.duplicate_filter_nodes_for_push_up_behind_train_test_split(dag, filter_removal_patch)

            self._move_duplicated_filter_to_new_location(dag, filter_removal_patch,
                                                         operator_after_which_cutoff_required_test,
                                                         operator_after_which_cutoff_required_train,
                                                         operator_to_add_node_after_test,
                                                         operator_to_add_node_after_train)

        elif filter_removal_patch.maybe_corresponding_test_set_operator is not None:
            operator_after_which_cutoff_required_train = filter_removal_patch \
                .filter_get_operator_after_which_cutoff_required(dag, filter_removal_patch.operator_to_remove)
            operator_after_which_cutoff_required_test = filter_removal_patch \
                .filter_get_operator_after_which_cutoff_required(
                dag, filter_removal_patch.maybe_corresponding_test_set_operator)

            self._move_filter_to_new_location(dag, filter_removal_patch, operator_after_which_cutoff_required_train,
                                              operator_to_add_node_after_train)
            self._move_filter_to_new_location(dag, filter_removal_patch, operator_after_which_cutoff_required_test,
                                              operator_to_add_node_after_test)
        else:
            # Need to move train operator only
            # Modifiy DAG, duplicate deletion node not needed probably
            operator_after_which_cutoff_required_train = filter_removal_patch \
                .filter_get_operator_after_which_cutoff_required(dag, filter_removal_patch.operator_to_remove)

            self._move_filter_to_new_location(dag, filter_removal_patch, operator_after_which_cutoff_required_train,
                                              operator_to_add_node_after_train)
        return dag

    @staticmethod
    def _move_duplicated_filter_to_new_location(dag, filter_removal_patch,
                                                operator_after_which_cutoff_required_test,
                                                operator_after_which_cutoff_required_train,
                                                operator_to_add_node_after_test, operator_to_add_node_after_train):
        # pylint: disable=too-many-arguments
        """Move the duplicated filter to the new location. Assumes that the duplicated nodes are present in the DAG"""
        # Apply the filter to the new train filter location and remove the old filter application
        children_of_train_operator_after_which_cutoff_required = list(dag.successors(
            operator_after_which_cutoff_required_train))
        children_of_test_operator_after_which_cutoff_required = list(dag.successors(
            operator_after_which_cutoff_required_test))
        children_of_filter_to_be_removed = list(dag.successors(filter_removal_patch.operator_to_remove))
        children_of_operator_to_add_node_after_train = list(dag.successors(operator_to_add_node_after_train))
        children_of_operator_to_add_node_after_test = list(dag.successors(operator_to_add_node_after_test))
        # Apply the filters to the new data
        for child_node in children_of_test_operator_after_which_cutoff_required:
            edge_data = dag.get_edge_data(operator_after_which_cutoff_required_test, child_node)
            dag.add_edge(operator_to_add_node_after_test, child_node, **edge_data)
            dag.remove_edge(operator_after_which_cutoff_required_test, child_node)
        dag.remove_node(operator_after_which_cutoff_required_test)  # Is only a duplicate without parents
        for child_node in children_of_train_operator_after_which_cutoff_required:
            edge_data = dag.get_edge_data(operator_after_which_cutoff_required_train, child_node)
            dag.add_edge(operator_to_add_node_after_train, child_node, **edge_data)
            dag.remove_edge(operator_after_which_cutoff_required_train, child_node)
        # Remove the filter output from its previous location and make the previous filter results
        # use the unfiltered result
        for child_node in children_of_filter_to_be_removed:
            edge_data = dag.get_edge_data(filter_removal_patch.operator_to_remove, child_node)
            dag.add_edge(operator_after_which_cutoff_required_train, child_node, **edge_data)
            dag.remove_edge(filter_removal_patch.operator_to_remove, child_node)
        # Actually use the results from the pushed-up filters
        for child_node in children_of_operator_to_add_node_after_train:
            edge_data = dag.get_edge_data(operator_to_add_node_after_train, child_node)
            dag.add_edge(filter_removal_patch.operator_to_remove, child_node, **edge_data)
            dag.remove_edge(operator_to_add_node_after_train, child_node)
        for child_node in children_of_operator_to_add_node_after_test:
            edge_data = dag.get_edge_data(operator_to_add_node_after_test, child_node)
            dag.add_edge(filter_removal_patch.maybe_corresponding_test_set_operator, child_node, **edge_data)
            dag.remove_edge(operator_to_add_node_after_test, child_node)

    @staticmethod
    def _move_filter_to_new_location(dag, filter_removal_patch, operator_after_which_cutoff_required_train,
                                     operator_to_add_node_after_train):
        """Remove a filter from its old location and move it to the new one"""
        # Apply the filter to the new train filter location and remove the old filter application
        if operator_after_which_cutoff_required_train == operator_to_add_node_after_train:
            return
        children_of_train_operator_after_which_cutoff_required = list(dag.successors(
            operator_after_which_cutoff_required_train))
        children_of_filter_to_be_removed = list(dag.successors(filter_removal_patch.operator_to_remove))
        children_of_operator_to_add_node_after_train = list(dag.successors(operator_to_add_node_after_train))
        # Apply the filters to the new data
        for child_node in children_of_train_operator_after_which_cutoff_required:
            edge_data = dag.get_edge_data(operator_after_which_cutoff_required_train, child_node)
            dag.add_edge(operator_to_add_node_after_train, child_node, **edge_data)
            dag.remove_edge(operator_after_which_cutoff_required_train, child_node)
        # Remove the filter output from its previous location and make the previous filter results
        # use the unfiltered result
        for child_node in children_of_filter_to_be_removed:
            edge_data = dag.get_edge_data(filter_removal_patch.operator_to_remove, child_node)
            dag.add_edge(operator_after_which_cutoff_required_train, child_node, **edge_data)
            dag.remove_edge(filter_removal_patch.operator_to_remove, child_node)
        # Actually use the results from the pushed-up filters
        for child_node in children_of_operator_to_add_node_after_train:
            edge_data = dag.get_edge_data(operator_to_add_node_after_train, child_node)
            dag.add_edge(filter_removal_patch.operator_to_remove, child_node, **edge_data)
            dag.remove_edge(operator_to_add_node_after_train, child_node)

    def duplicate_filter_nodes_for_push_up_behind_train_test_split(self, dag, filter_removal_patch: OperatorRemoval):
        """Duplicate a filter so we can push it up to both the train and test side separately"""
        all_operators_associated_with_filter = filter_removal_patch.get_all_operators_associated_with_filter(
            dag, filter_removal_patch.operator_to_remove, True)
        only_filter_nodes_dag = dag.copy()
        only_filter_nodes_dag.remove_nodes_from([node for node in list(only_filter_nodes_dag.nodes)
                                                 if node not in all_operators_associated_with_filter])

        def get_new_dag_node_id_new_node_label(node: DagNode) -> DagNode:
            return DagNode(self._pipeline_executor.get_next_op_id(),
                           node.code_location,
                           node.operator_info,
                           node.details,
                           node.optional_code_info,
                           node.processing_func,
                           node.make_classifier_func)

        # noinspection PyTypeChecker
        only_filter_nodes_dag = networkx.relabel_nodes(only_filter_nodes_dag,
                                                       get_new_dag_node_id_new_node_label)
        test_side_selection_to_remove = [node for node in only_filter_nodes_dag.nodes
                                         if node.operator_info.operator == OperatorType.SELECTION][0]
        filter_removal_patch.maybe_corresponding_test_set_operator = test_side_selection_to_remove
        test_set_filter_duplicate_root_node = [node for node in only_filter_nodes_dag.nodes
                                               if len(list(only_filter_nodes_dag.predecessors(node))) == 0][0]
        dag = networkx.compose(dag, only_filter_nodes_dag)
        return dag, test_set_filter_duplicate_root_node

    @staticmethod
    def _get_columns_required_for_filter_eval(dag, filter_removal_patch):
        """Get all columns required to be in the df required for filter evaluation"""
        operator_parent_nodes = get_sorted_parent_nodes(dag, filter_removal_patch.operator_to_remove)
        selection_parent_a = operator_parent_nodes[0]
        selection_parent_b = operator_parent_nodes[-1]
        # We want to introduce the change before all subscript behavior
        operator_to_apply_corruption_after = networkx.lowest_common_ancestor(dag, selection_parent_a,
                                                                             selection_parent_b)
        sorted_successors = get_sorted_children_nodes(dag, operator_to_apply_corruption_after)
        using_columns_for_filter = set()
        for child in sorted_successors:
            if child.operator_info.operator == OperatorType.PROJECTION:
                using_columns_for_filter.update(child.details.columns)
        return using_columns_for_filter
