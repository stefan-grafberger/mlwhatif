"""
Simple Projection push-up optimization that ignores that data corruptions only corrupt subsets and that
it is possible to corrupt the whole set at once and then only sample from the corrupted DF.
"""
from collections import defaultdict
from functools import partial
from typing import List

import networkx

from mlwhatif.instrumentation._dag_node import DagNode, BasicCodeLocation, OperatorContext, DagNodeDetails
from mlwhatif.instrumentation._operator_types import OperatorType
from mlwhatif.execution._patches import Patch
from mlwhatif.execution.optimization._internal_optimization_patches import AppendNodeBetweenOperators, \
    UdfSplitAndReuseAppendNodeBetweenOperators
from mlwhatif.execution.optimization._query_optimization_rules import QueryOptimizationRule


class UdfSplitAndReuse(QueryOptimizationRule):
    """ Combines multiple DAGs and optimizes the joint plan """

    # pylint: disable=too-few-public-methods

    def __init__(self, pipeline_executor):
        self._pipeline_executor = pipeline_executor

    def optimize_patches(self, dag: networkx.DiGraph, patches: List[List[Patch]]) -> List[List[Patch]]:
        id_to_index_selection = dict()
        id_to_projection_func = dict()
        # If we allow multiple corruption funcs per column at some point, this would need to change
        selectivities_per_column = defaultdict(list)
        column_to_index_selection_ids = defaultdict(list)
        column_to_projection_func_ids = defaultdict(list)

        for pipeline_variant_patches in patches:
            for patch in pipeline_variant_patches:
                if isinstance(patch, AppendNodeBetweenOperators) and patch.maybe_udf_split_info is not None:
                    id_to_projection_func[patch.maybe_udf_split_info.projection_func_only_id] = \
                        patch.maybe_udf_split_info.projection_func_only
                    id_to_index_selection[patch.maybe_udf_split_info.index_selection_func_id] = \
                        patch.maybe_udf_split_info.index_selection_func
                    selectivities_per_column[patch.maybe_udf_split_info.column_name_to_corrupt].append(
                        patch.maybe_udf_split_info.maybe_selectivity_info)
                    column_to_index_selection_ids[patch.maybe_udf_split_info.column_name_to_corrupt].append(
                        patch.maybe_udf_split_info.index_selection_func_id)
                    column_to_projection_func_ids[patch.maybe_udf_split_info.column_name_to_corrupt]\
                        .append(patch.maybe_udf_split_info.projection_func_only_id)


        columns_worth_fully_corrupting = set()
        for column, selectivity_list in selectivities_per_column.items():
            total_corruption_fraction = sum(filter(None, selectivity_list))
            if total_corruption_fraction > 1.0:
                columns_worth_fully_corrupting.add(column)

        corruption_func_id_to_dag_node = dict()
        index_selection_func_id_to_dag_node = dict()
        for column in columns_worth_fully_corrupting:
            projection_func_ids = column_to_projection_func_ids[column]
            index_selection_ids = column_to_index_selection_ids[column]
            for projection_id in projection_func_ids:
                if projection_id not in corruption_func_id_to_dag_node:
                    projection_func = id_to_projection_func[projection_id]
                    corruption_func_id_to_dag_node[(True, projection_id)] = self.create_projection_func_dag_node(
                        projection_func, column)
                    corruption_func_id_to_dag_node[(False, projection_id)] = self.create_projection_func_dag_node(
                        projection_func, column)
            for index_id in index_selection_ids:
                if index_id not in index_selection_func_id_to_dag_node:
                    index_func = id_to_index_selection[index_id]
                    index_selection_func_id_to_dag_node[(True, index_id)] = \
                        self.create_index_selection_func_dag_node(index_func)
                    index_selection_func_id_to_dag_node[(False, index_id)] = \
                        self.create_index_selection_func_dag_node(index_func)

        # TODO: Determine for which projection funcs the total selectivity is greater than 1
        #  For other patches, leave them unchanged.
        #  For these patches, generate a projection node for each projection func id
        #  and an index selection func for each index selection func id
        #  Then iterate through all patches again, and get the nodes from the maps and generate a new DagNode
        updated_patches = []
        for pipeline_variant_patches in patches:
            updated_pipeline_variant_patches = []
            for patch in pipeline_variant_patches:
                if isinstance(patch, AppendNodeBetweenOperators) and patch.maybe_udf_split_info is not None and \
                        patch.maybe_udf_split_info.column_name_to_corrupt in columns_worth_fully_corrupting:
                    corruption_dag_node = corruption_func_id_to_dag_node[
                        (patch.train_not_test, patch.maybe_udf_split_info.projection_func_only_id)]
                    index_selection_node = index_selection_func_id_to_dag_node[
                        (patch.train_not_test, patch.maybe_udf_split_info.index_selection_func_id)]
                    apply_corruption_to_fraction_node = self.create_apply_corruption_to_fraction_node(patch)
                    updated_patch = UdfSplitAndReuseAppendNodeBetweenOperators(
                        patch.patch_id,
                        patch.analysis,
                        patch.changes_following_results,
                        patch.operator_to_add_node_after,
                        patch.operator_to_add_node_before,
                        corruption_dag_node,
                        index_selection_node,
                        apply_corruption_to_fraction_node,
                        patch.train_not_test
                    )
                else:
                    updated_patch = patch
                updated_pipeline_variant_patches.append(updated_patch)
            updated_patches.append(updated_pipeline_variant_patches)

        return updated_patches

    def create_projection_func_dag_node(self, projection_func, column_name) -> DagNode:
        def corrupt_full_df(pandas_df, corruption_function):
            # TODO: If we model this as 3 operations instead of one, optimization should be easy
            # TODO: Think about when we actually want to be defensive and call copy and when not
            # TODO: Think about datatypes. corruption_function currently assumes pandas DataFrames.
            #  We may want to automatically convert data formats as needed, here and in other places.
            completely_corrupted_df = pandas_df.copy()
            completely_corrupted_df = corruption_function(completely_corrupted_df)
            return completely_corrupted_df

        corrupt_df_with_proper_bindings = partial(corrupt_full_df, corruption_function=projection_func)

        description = f"Corrupt 100% of '{column_name}'"
        new_corruption_node = DagNode(self._pipeline_executor.get_next_op_id(),
                                      BasicCodeLocation("UdfSplitAndReuse", None),
                                      OperatorContext(OperatorType.PROJECTION_MODIFY, None),
                                      DagNodeDetails(description, None),
                                      None,
                                      corrupt_df_with_proper_bindings)
        return new_corruption_node

    def create_index_selection_func_dag_node(self, index_selection_func) -> DagNode:
        new_corruption_node = DagNode(self._pipeline_executor.get_next_op_id(),
                                      BasicCodeLocation("UdfSplitAndReuse", None),
                                      OperatorContext(OperatorType.SUBSCRIPT, None),
                                      DagNodeDetails("Indices to corrupt", None),
                                      None,
                                      index_selection_func)
        return new_corruption_node

    def create_apply_corruption_to_fraction_node(self, previous_patch: AppendNodeBetweenOperators) -> DagNode:
        def corrupt_df(pandas_df, completely_corrupted_df, indexes_to_corrupt, column):
            return_df = pandas_df.copy()
            return_df.loc[indexes_to_corrupt, column] = completely_corrupted_df.loc[indexes_to_corrupt, column]
            return return_df

        corrupt_df_with_proper_bindings = partial(corrupt_df,
                                                  column=previous_patch.maybe_udf_split_info.column_name_to_corrupt)
        new_corruption_node = DagNode(self._pipeline_executor.get_next_op_id(),
                                      previous_patch.node_to_insert.code_location,
                                      OperatorContext(OperatorType.PROJECTION_MODIFY, None),
                                      previous_patch.node_to_insert.details,
                                      None,
                                      corrupt_df_with_proper_bindings)
        return new_corruption_node
