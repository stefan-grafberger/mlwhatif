"""
Simple Projection push-up optimization that ignores that data corruptions only corrupt subsets and that
it is possible to corrupt the whole set at once and then only sample from the corrupted DF.
"""
from collections import defaultdict
from functools import partial
from typing import List

import networkx
import pandas

from mlwhatif.instrumentation._dag_node import DagNode, BasicCodeLocation, OperatorContext, DagNodeDetails
from mlwhatif.instrumentation._operator_types import OperatorType
from mlwhatif.execution._patches import PipelinePatch
from mlwhatif.optimization._internal_optimization_patches import AppendNodeBetweenOperators, \
    UdfSplitAndReuseAppendNodeBetweenOperators
from mlwhatif.optimization._query_optimization_rules import QueryOptimizationRule


class UdfSplitAndReuse(QueryOptimizationRule):
    """ Combines multiple DAGs and optimizes the joint plan """
    # pylint: disable=too-few-public-methods

    def __init__(self, pipeline_executor, disable_selectivity_safety=False):
        self._pipeline_executor = pipeline_executor
        self._disable_selectivity_safety = disable_selectivity_safety

    def optimize_patches(self, dag: networkx.DiGraph, patches: List[List[PipelinePatch]]) -> List[List[PipelinePatch]]:
        columns_worth_fully_corrupting = self._get_columns_worth_fully_corrupting(patches)

        corruption_func_id_to_dag_node, index_selection_func_id_to_dag_node, index_selection_func_requires_input = \
            self._get_split_and_reuse_dag_node_mapping(patches, columns_worth_fully_corrupting)

        updated_patches = []
        for pipeline_variant_patches in patches:
            updated_pipeline_variant_patches = []
            for patch in pipeline_variant_patches:
                if isinstance(patch, AppendNodeBetweenOperators) and patch.maybe_udf_split_info is not None and \
                        patch.maybe_udf_split_info.column_name_to_corrupt in columns_worth_fully_corrupting:

                    corruption_dag_node, first_index_function_occurrence, index_selection_node = \
                        self._get_udf_split_reuse_patch_arguments(patch, corruption_func_id_to_dag_node,
                                                                  index_selection_func_id_to_dag_node,
                                                                  index_selection_func_requires_input)
                    updated_patch = self._create_udf_split_reuse_patch(corruption_dag_node, index_selection_node,
                                                                       first_index_function_occurrence, patch)
                else:
                    updated_patch = patch
                updated_pipeline_variant_patches.append(updated_patch)
            updated_patches.append(updated_pipeline_variant_patches)

        return updated_patches

    @staticmethod
    def _get_udf_split_reuse_patch_arguments(patch, corruption_func_id_to_dag_node,
                                             index_selection_func_id_to_dag_node, index_selection_func_requires_input):
        """
        Lookup the corresponding DAG nodes for the current patch and if the index selection func DAG node
        still needs an edge to look at the input shape
        """
        corruption_lookup_key = (patch.train_not_test, patch.maybe_udf_split_info.projection_func_only_id)
        index_selection_lookup_key = (patch.train_not_test,
                                      patch.maybe_udf_split_info.index_selection_func_id)

        corruption_dag_node = corruption_func_id_to_dag_node[corruption_lookup_key]
        index_selection_node = index_selection_func_id_to_dag_node[index_selection_lookup_key]

        first_index_function_occurrence = index_selection_lookup_key in index_selection_func_requires_input
        index_selection_func_requires_input.discard(index_selection_lookup_key)
        return corruption_dag_node, first_index_function_occurrence, index_selection_node

    @staticmethod
    def _get_column_to_index_func_and_to_corruption_func_maps(patches):
        # If we allow multiple corruption funcs per column at some point, this would need to change
        column_to_index_selection_ids = defaultdict(list)
        column_to_projection_func_ids = defaultdict(list)
        for pipeline_variant_patches in patches:
            for patch in pipeline_variant_patches:
                if isinstance(patch, AppendNodeBetweenOperators) and patch.maybe_udf_split_info is not None:
                    column_to_index_selection_ids[patch.maybe_udf_split_info.column_name_to_corrupt].append(
                        patch.maybe_udf_split_info.index_selection_func_id)
                    column_to_projection_func_ids[patch.maybe_udf_split_info.column_name_to_corrupt] \
                        .append(patch.maybe_udf_split_info.projection_func_only_id)
        return column_to_index_selection_ids, column_to_projection_func_ids

    @staticmethod
    def get_index_id_to_func_and_corruption_id_to_func_maps(patches):
        """Create DAG nodes and get maps from index selection/coruption id to the corresponding node"""
        id_to_index_selection = dict()
        id_to_projection_func = dict()
        for pipeline_variant_patches in patches:
            for patch in pipeline_variant_patches:
                if isinstance(patch, AppendNodeBetweenOperators) and patch.maybe_udf_split_info is not None:
                    id_to_projection_func[patch.maybe_udf_split_info.projection_func_only_id] = \
                        patch.maybe_udf_split_info.projection_func_only
                    id_to_index_selection[patch.maybe_udf_split_info.index_selection_func_id] = \
                        patch.maybe_udf_split_info.index_selection_func
        return id_to_index_selection, id_to_projection_func

    def _get_split_and_reuse_dag_node_mapping(self, patches, columns_worth_fully_corrupting):
        """Create DAG nodes for corruption and index selection funcs that we want to split and reuse"""
        id_to_index_selection, id_to_projection_func = self.get_index_id_to_func_and_corruption_id_to_func_maps(patches)
        column_to_index_selection_ids, column_to_projection_func_ids = self \
            ._get_column_to_index_func_and_to_corruption_func_maps(patches)

        corruption_func_id_to_dag_node = self._create_corruption_id_to_dag_nodes_map(column_to_projection_func_ids,
                                                                                     columns_worth_fully_corrupting,
                                                                                     id_to_projection_func)
        index_selection_func_id_to_dag_node, index_selection_func_requires_input = \
            self._create_index_selection_id_to_dag_nodes_map(column_to_index_selection_ids,
                                                             columns_worth_fully_corrupting, id_to_index_selection)
        return corruption_func_id_to_dag_node, index_selection_func_id_to_dag_node, index_selection_func_requires_input

    def _create_index_selection_id_to_dag_nodes_map(self, column_to_index_selection_ids, columns_worth_fully_corrupting,
                                                    id_to_index_selection):
        """Create DAG nodes for index selection funcs that we want to split and reuse"""
        index_selection_func_id_to_dag_node = dict()
        index_selection_func_requires_input = set()
        for column in columns_worth_fully_corrupting:
            index_selection_ids = column_to_index_selection_ids[column]
            for index_id in index_selection_ids:
                if index_id not in index_selection_func_id_to_dag_node:
                    index_func = id_to_index_selection[index_id]
                    index_selection_func_id_to_dag_node[(True, index_id)] = \
                        self._create_index_selection_func_dag_node(index_func)
                    index_selection_func_id_to_dag_node[(False, index_id)] = \
                        self._create_index_selection_func_dag_node(index_func)
                    index_selection_func_requires_input.add((True, index_id))
                    index_selection_func_requires_input.add((False, index_id))
        return index_selection_func_id_to_dag_node, index_selection_func_requires_input

    def _create_corruption_id_to_dag_nodes_map(self, column_to_projection_func_ids, columns_worth_fully_corrupting,
                                               id_to_projection_func):
        """Create DAG nodes for corruption funcs that we want to split and reuse"""
        corruption_func_id_to_dag_node = dict()
        for column in columns_worth_fully_corrupting:
            projection_func_ids = column_to_projection_func_ids[column]
            for projection_id in projection_func_ids:
                if projection_id not in corruption_func_id_to_dag_node:
                    projection_func = id_to_projection_func[projection_id]
                    corruption_func_id_to_dag_node[(True, projection_id)] = self._create_projection_func_dag_node(
                        projection_func, column)
                    corruption_func_id_to_dag_node[(False, projection_id)] = self._create_projection_func_dag_node(
                        projection_func, column)
        return corruption_func_id_to_dag_node

    def _create_udf_split_reuse_patch(self, corruption_dag_node, index_selection_node, first_index_function_occurrence,
                                      patch):
        """Create the patch that applies the optimized corruption"""
        apply_corruption_to_fraction_node = self._create_apply_corruption_to_fraction_node(patch)
        updated_patch = UdfSplitAndReuseAppendNodeBetweenOperators(
            patch.patch_id,
            patch.analysis,
            patch.changes_following_results,
            patch.operator_to_add_node_after,
            patch.operator_to_add_node_before,
            corruption_dag_node,
            index_selection_node,
            apply_corruption_to_fraction_node,
            patch.train_not_test,
            first_index_function_occurrence
        )
        return updated_patch

    def _get_columns_worth_fully_corrupting(self, patches):
        """Get columns where, in total, different corruptions corrupt more than 100% of the data"""
        selectivities_per_column = defaultdict(list)

        for pipeline_variant_patches in patches:
            for patch in pipeline_variant_patches:
                if isinstance(patch, AppendNodeBetweenOperators) and patch.maybe_udf_split_info is not None:
                    selectivities_per_column[patch.maybe_udf_split_info.column_name_to_corrupt].append(
                        patch.maybe_udf_split_info.maybe_selectivity_info)

        columns_worth_fully_corrupting = set()
        for column, selectivity_list in selectivities_per_column.items():
            if self._disable_selectivity_safety is True:
                columns_worth_fully_corrupting.add(column)
            else:
                total_corruption_fraction = sum(filter(None, selectivity_list))
                if total_corruption_fraction > 1.0:
                    columns_worth_fully_corrupting.add(column)
        return columns_worth_fully_corrupting

    def _create_projection_func_dag_node(self, projection_func, column_name) -> DagNode:
        """Create the DAG node that corrupts the whole df at once"""

        def corrupt_full_df(pandas_df, corruption_function):
            if isinstance(pandas_df, pandas.Series):
                pandas_df = pandas.DataFrame(pandas_df)
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

    def _create_index_selection_func_dag_node(self, index_selection_func) -> DagNode:
        """Create the DAG node that creates the sampling index array to sample from the corrupted df"""
        new_corruption_node = DagNode(self._pipeline_executor.get_next_op_id(),
                                      BasicCodeLocation("UdfSplitAndReuse", None),
                                      OperatorContext(OperatorType.SUBSCRIPT, None),
                                      DagNodeDetails("Indices to corrupt", None),
                                      None,
                                      index_selection_func)
        return new_corruption_node

    def _create_apply_corruption_to_fraction_node(self, previous_patch: AppendNodeBetweenOperators) -> DagNode:
        """
        Create the DAG node that applies the sampling index array to sample from the corrupted data to
        corrupt certain rows
        """

        def corrupt_df(pandas_df, completely_corrupted_df, indexes_to_corrupt, column):
            if isinstance(pandas_df, pandas.Series):
                pandas_df = pandas.DataFrame(pandas_df)
                was_series = True
            else:
                was_series = False
            return_df = pandas_df.copy()
            return_df.loc[indexes_to_corrupt, column] = completely_corrupted_df.loc[indexes_to_corrupt, column]
            if was_series is True:
                return_df = return_df[column]
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
