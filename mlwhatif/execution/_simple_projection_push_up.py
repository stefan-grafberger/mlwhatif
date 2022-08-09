"""
Simple Projection push-up optimization that ignores that data corruptions only corrupt subsets and that
it is possible to corrupt the whole set at once and then only sample from the corrupted DF.
"""
from typing import List

import networkx

from mlwhatif.execution._patches import Patch
from mlwhatif.execution._query_optimization_rules import QueryOptimizationRule


class SimpleProjectionPushUp(QueryOptimizationRule):
    """ Combines multiple DAGs and optimizes the joint plan """
    # pylint: disable=too-few-public-methods

    def __init__(self, pipeline_executor):
        self._pipeline_executor = pipeline_executor

    def optimize(self, dag: networkx.DiGraph, patches: List[List[Patch]]) -> List[List[Patch]]:
        return patches
