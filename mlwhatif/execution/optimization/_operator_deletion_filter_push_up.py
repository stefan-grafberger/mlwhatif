"""
Simple Projection push-up optimization that ignores that data corruptions only corrupt subsets and that
it is possible to corrupt the whole set at once and then only sample from the corrupted DF.
"""
from typing import List

import networkx

from mlwhatif import OperatorType
from mlwhatif.execution._patches import Patch, OperatorRemoval
from mlwhatif.execution.optimization._query_optimization_rules import QueryOptimizationRule


class OperatorDeletionFilterPushUp(QueryOptimizationRule):
    """ Combines multiple DAGs and optimizes the joint plan """

    # pylint: disable=too-few-public-methods

    def __init__(self, pipeline_executor):
        self._pipeline_executor = pipeline_executor

    def optimize(self, dag: networkx.DiGraph, patches: List[List[Patch]]) -> List[List[Patch]]:
        pipeline_variants_with_selectivity = {}
        # TODO: For now, only consider set of pipeline variants that only contain filter patches

        for pipeline_variant_patches in patches:
            for patch_index, patch in enumerate(pipeline_variant_patches):
                if isinstance(patch, OperatorRemoval) and \
                        patch.operator_to_remove.operator_info.operator == OperatorType.SELECTION:
                    parent_row_count = dag.predecessors(patch.operator_to_remove)[0].details.optimizer_info.shape[1]
                    current_row_count = patch.operator_to_remove.details.optimizer_info.shape[1]
                    selectivity = current_row_count / parent_row_count
                    pipeline_variants_with_selectivity[patch_index] = selectivity

        # This assumes that selections are independent, not sure yet if that is an assumption we should make
        percentage_of_data_processed_without_filter = sum(pipeline_variants_with_selectivity.values())
        is_filter_removal_worth = percentage_of_data_processed_without_filter > 1.0


        if is_filter_removal_worth:
            # TODO
            updated_patches = patches
        else:
            updated_patches = patches
        return updated_patches
