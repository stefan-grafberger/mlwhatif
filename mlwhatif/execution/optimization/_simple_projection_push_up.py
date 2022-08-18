"""
Simple Projection push-up optimization that ignores that data corruptions only corrupt subsets and that
it is possible to corrupt the whole set at once and then only sample from the corrupted DF.
"""
from typing import List

import networkx

from mlwhatif.analysis._analysis_utils import find_dag_location_for_first_op_modifying_column
from mlwhatif.execution.optimization._internal_optimization_patches import AppendNodeBetweenOperators, \
    PipelineTransformerInsertion
from mlwhatif.execution._patches import Patch, DataProjection, DataTransformer
from mlwhatif.execution.optimization._query_optimization_rules import QueryOptimizationRule


class SimpleProjectionPushUp(QueryOptimizationRule):
    """ Combines multiple DAGs and optimizes the joint plan """

    # pylint: disable=too-few-public-methods

    def __init__(self, pipeline_executor):
        self._pipeline_executor = pipeline_executor

    def optimize_patches(self, dag: networkx.DiGraph, patches: List[List[Patch]]) -> List[List[Patch]]:
        updated_patches = []
        for pipeline_variant_patches in patches:
            updated_pipeline_variant_patches = []
            for patch in pipeline_variant_patches:
                if isinstance(patch, DataProjection):
                    if patch.only_reads_column == [patch.modifies_column]:
                        operator_to_add_node_after, operator_to_add_node_before = \
                            find_dag_location_for_first_op_modifying_column(patch.modifies_column, dag,
                                                                            patch.train_not_test)
                        updated_patch = AppendNodeBetweenOperators(patch.patch_id, patch.analysis,
                                                                   patch.changes_following_results,
                                                                   patch.projection_operator,
                                                                   operator_to_add_node_after,
                                                                   operator_to_add_node_before,
                                                                   patch.maybe_udf_split_info)
                        updated_pipeline_variant_patches.append(updated_patch)
                    else:
                        # TODO: This can be done too but probably a waste of time at this point to look at edge cases
                        #  too much
                        updated_pipeline_variant_patches.append(patch)
                elif isinstance(patch, DataTransformer):
                    fit_transform_operator_to_add_node_after, fit_transform_operator_to_add_node_before = \
                        find_dag_location_for_first_op_modifying_column(patch.modifies_column, dag, True)
                    transform_operator_to_add_node_after, transform_operator_to_add_node_before = \
                        find_dag_location_for_first_op_modifying_column(patch.modifies_column, dag, False)
                    updated_patch = PipelineTransformerInsertion(patch.patch_id, patch.analysis,
                                                                 patch.changes_following_results,
                                                                 patch.fit_transform_operator,
                                                                 fit_transform_operator_to_add_node_after,
                                                                 fit_transform_operator_to_add_node_before,
                                                                 patch.transform_operator,
                                                                 transform_operator_to_add_node_after,
                                                                 transform_operator_to_add_node_before)
                    updated_pipeline_variant_patches.append(updated_patch)
                else:
                    updated_pipeline_variant_patches.append(patch)
            updated_patches.append(updated_pipeline_variant_patches)
        return updated_patches
