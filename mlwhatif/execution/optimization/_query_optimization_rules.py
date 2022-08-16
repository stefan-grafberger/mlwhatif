"""
Simple Projection push-up optimization that ignores that data corruptions only corrupt subsets and that
it is possible to corrupt the whole set at once and then only sample from the corrupted DF.
"""
import abc
from typing import List

import networkx

from mlwhatif.execution._patches import Patch


class QueryOptimizationRule(metaclass=abc.ABCMeta):
    """
    The Interface for Query Optimization Rules
    """
    def optimize_patches(self, dag: networkx.DiGraph, patches: List[List[Patch]]) -> List[List[Patch]]:
        """Transform the patches into more efficient ones"""
        # pylint: disable=unused-argument,no-self-use
        return patches

    def optimize_dag(self, dag: networkx.DiGraph, patches: List[List[Patch]]) -> networkx.DiGraph:
        """Transform the patches into more efficient ones"""
        # pylint: disable=unused-argument,no-self-use
        return dag
