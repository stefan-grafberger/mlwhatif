"""
The place where the Multi-Query optimization happens
"""
from typing import List

import networkx


class MultiQueryOptimizer:
    """ Combines multiple DAGs and optimizes the joint plan """
    # pylint: disable=too-few-public-methods

    @staticmethod
    def create_optimized_plan(dags: List[networkx.DiGraph]) -> networkx.DiGraph:
        """ Optimize and combine multiple given input DAGs """
        big_execution_dag = networkx.compose_all(dags)
        # TODO: More optimizations, maybe create some optimization rule interface that what-if analyses can use
        #  to specify analysis-specific optimizations
        return big_execution_dag
