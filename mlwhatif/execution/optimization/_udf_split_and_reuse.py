"""
Simple Projection push-up optimization that ignores that data corruptions only corrupt subsets and that
it is possible to corrupt the whole set at once and then only sample from the corrupted DF.
"""
from collections import defaultdict
from typing import List

import networkx

from mlwhatif.execution._patches import Patch, DataProjection
from mlwhatif.execution.optimization._query_optimization_rules import QueryOptimizationRule


class UdfSplitAndReuse(QueryOptimizationRule):
    """ Combines multiple DAGs and optimizes the joint plan """

    # pylint: disable=too-few-public-methods

    def __init__(self, pipeline_executor):
        self._pipeline_executor = pipeline_executor

    def optimize_dag(self, dag: networkx.DiGraph, patches: List[List[Patch]]) -> \
            tuple[networkx.DiGraph, List[List[Patch]]]:
        selectivity_and_filters_to_push_up = []

        index_selection_ids = set()
        projection_func_ids = set()
        splittable_patches = []
        selectivities_per_projection_func_id = defaultdict(list)

        for pipeline_variant_patches in patches:
            for patch in pipeline_variant_patches:
                if isinstance(patch, DataProjection) and patch.projection_func_only_id is not None \
                        and patch.index_selection_func_id is not None:
                    splittable_patches.append(patch)
                    projection_func_ids.add(patch.projection_func_only_id)
                    index_selection_ids.add(patch.index_selection_func_id)
                    selectivities_per_projection_func_id[patch.projection_func_only_id].append(
                        patch.maybe_selectivity_info)

        # TODO
        for _, _ in selectivity_and_filters_to_push_up:
            pass
        return dag, patches
