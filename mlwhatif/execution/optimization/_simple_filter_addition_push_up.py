"""
Simple Projection push-up optimization that ignores that data corruptions only corrupt subsets and that
it is possible to corrupt the whole set at once and then only sample from the corrupted DF.
"""
from typing import List

import networkx

from mlwhatif.analysis._analysis_utils import find_dag_location_for_new_filter_on_column
from mlwhatif.execution._patches import PipelinePatch, DataFiltering, AppendNodeAfterOperator
from mlwhatif.execution.optimization._query_optimization_rules import QueryOptimizationRule


class SimpleFilterAdditionPushUp(QueryOptimizationRule):
    """ Push up a filter that is added as a DataFiltering Patch """

    # pylint: disable=too-few-public-methods

    def __init__(self, pipeline_executor):
        self._pipeline_executor = pipeline_executor

    def optimize_patches(self, dag: networkx.DiGraph, patches: List[List[PipelinePatch]]) -> List[List[PipelinePatch]]:
        updated_patches = []
        for pipeline_variant_patches in patches:
            updated_pipeline_variant_patches = []
            for patch in pipeline_variant_patches:
                if isinstance(patch, DataFiltering):
                    operator_to_add_node_after = find_dag_location_for_new_filter_on_column(
                        patch.only_reads_column, dag, patch.train_not_test)
                    updated_patch = AppendNodeAfterOperator(patch.patch_id, patch.analysis,
                                                            patch.changes_following_results,
                                                            operator_to_add_node_after,
                                                            patch.filter_operator)
                    updated_pipeline_variant_patches.append(updated_patch)
                else:
                    updated_pipeline_variant_patches.append(patch)
            updated_patches.append(updated_pipeline_variant_patches)
        return updated_patches
