"""
The place where the Multi-Query optimization happens
"""
from typing import List

import networkx


class MultiQueryOptimizer:
    """ Combines multiple DAGs and optimizes the joint plan """

    @staticmethod
    def create_optimized_plan(dags: List[networkx.DiGraph]) -> networkx.DiGraph:
        """ Optimize and combine multiple given input DAGs """
        big_execution_dag = networkx.compose_all(dags)
        return big_execution_dag
