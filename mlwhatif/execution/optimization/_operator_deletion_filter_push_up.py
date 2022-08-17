"""
Simple Projection push-up optimization that ignores that data corruptions only corrupt subsets and that
it is possible to corrupt the whole set at once and then only sample from the corrupted DF.
"""
from typing import List

import networkx

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
            for patch_index, patch in enumerate(pipeline_variant_patches):
                if isinstance(patch, OperatorRemoval) and \
                        patch.operator_to_remove.operator_info.operator == OperatorType.SELECTION:
                    parent = list(dag.predecessors(patch.operator_to_remove))[0]
                    parent_row_count = parent.details.optimizer_info.shape[1]
                    current_row_count = patch.operator_to_remove.details.optimizer_info.shape[1]
                    selectivity = current_row_count / parent_row_count
                    selectivity_and_filters_to_push_up.append((selectivity, patch))
                    # calculate selectivities

        # Sort filters by selectivity, the ones with the highest selectivity should be moved up first
        selectivity_and_filters_to_push_up = sorted(selectivity_and_filters_to_push_up, key=lambda x: x[0])

        for _, filter_removal_patch in selectivity_and_filters_to_push_up:
            using_columns_for_filter = self._get_columns_required_for_filter_eval(dag, filter_removal_patch)

            operator_to_add_node_after_train = find_dag_location_for_new_filter_on_column(
                using_columns_for_filter, dag, True)

            # TODO: Move the filter up (on both train and test set if required, maybe even duplicate filter if before
            #  train_test_split)

        return dag

    def _get_columns_required_for_filter_eval(self, dag, filter_removal_patch):
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
