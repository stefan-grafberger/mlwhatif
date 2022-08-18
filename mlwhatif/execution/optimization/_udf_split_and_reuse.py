"""
Simple Projection push-up optimization that ignores that data corruptions only corrupt subsets and that
it is possible to corrupt the whole set at once and then only sample from the corrupted DF.
"""
from collections import defaultdict
from typing import List

import networkx

from mlwhatif.execution._patches import Patch
from mlwhatif.execution.optimization._internal_optimization_patches import AppendNodeBetweenOperators
from mlwhatif.execution.optimization._query_optimization_rules import QueryOptimizationRule


class UdfSplitAndReuse(QueryOptimizationRule):
    """ Combines multiple DAGs and optimizes the joint plan """

    # pylint: disable=too-few-public-methods

    def __init__(self, pipeline_executor):
        self._pipeline_executor = pipeline_executor

    def optimize_patches(self, dag: networkx.DiGraph, patches: List[List[Patch]]) -> List[List[Patch]]:
        selectivity_and_filters_to_push_up = []

        id_to_index_selection = dict()
        id_to_projection_func = dict()
        splittable_patches = []
        selectivities_per_projection_func_id = defaultdict(list)

        for pipeline_variant_patches in patches:
            for patch in pipeline_variant_patches:
                if isinstance(patch, AppendNodeBetweenOperators) and patch.maybe_udf_split_info is not None:
                    splittable_patches.append(patch)
                    id_to_projection_func[patch.maybe_udf_split_info.projection_func_only_id] = \
                        patch.maybe_udf_split_info.projection_func_only
                    id_to_index_selection[patch.maybe_udf_split_info.index_selection_func_id] = \
                        patch.maybe_udf_split_info.index_selection_func
                    selectivities_per_projection_func_id[patch.maybe_udf_split_info.projection_func_only_id].append(
                        patch.maybe_udf_split_info.maybe_selectivity_info)

        # TODO: Determine for which projection funcs the total selectivity is greater than 1
        #  For other patches, leave them unchanged.
        #  For these patches, generate a projection node for each projection func id
        #  and an index selection func for each index selection func id
        #  Then iterate through all patches again, and get the nodes from the maps and generate a new DagNode
        for _, _ in selectivity_and_filters_to_push_up:
            pass
        return patches
