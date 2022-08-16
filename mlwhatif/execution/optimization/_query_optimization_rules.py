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
    # pylint: disable=too-few-public-methods
    def optimize(self, dag: networkx.DiGraph, patches: List[List[Patch]]) -> List[List[Patch]]:
        """Transform the patches into more efficient ones"""
        raise NotImplementedError
