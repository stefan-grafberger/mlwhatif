"""
The Interface for the What-If Analyses
"""
import functools
from types import FunctionType
from typing import Iterable, Dict

import networkx
import numpy

from mlwhatif import OperatorType, DagNode, OperatorContext, DagNodeDetails
from mlwhatif.analysis._analysis_utils import add_intermediate_extraction_after_node, find_nodes_by_type, \
    find_first_op_modifying_a_column, add_new_node_after_node, mark_nodes_to_recompute_after_changed_node
from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis
from mlwhatif.instrumentation._pipeline_executor import singleton


class DataCorruption(WhatIfAnalysis):
    """
    The Data Corruption What-If Analysis
    """

    def __init__(self,
                 column_to_corruption: Dict[str, FunctionType],
                 corruption_percentages: Iterable[float] or None = None,
                 also_corrupt_train: bool = False):
        self.column_to_corruption = list(column_to_corruption.items())
        self.also_corrupt_train = also_corrupt_train
        if corruption_percentages is None:
            self.corruption_percentages = [0.2, 0.5, 0.9]
        self._analysis_id = (*self.column_to_corruption, *self.corruption_percentages, self.also_corrupt_train)

    @property
    def analysis_id(self):
        return self._analysis_id

    def generate_plans_to_try(self, dag: networkx.DiGraph)\
            -> Iterable[networkx.DiGraph]:
        # pylint: disable=cell-var-from-loop
        operator_type = OperatorType.SCORE
        score_operators = find_nodes_by_type(dag, operator_type)
        final_result_value = score_operators[0]
        # TODO: Deduplication for transformers that process multiple columns at once, if necessary for corruption
        if len(score_operators) != 1:
            raise Exception("Currently, DataCorruption only supports pipelines with exactly one score call!")

        corruption_dags = []
        for corruption_percentage in self.corruption_percentages:
            for (column, corruption_function) in self.column_to_corruption:
                corruption_dag = dag.copy()
                add_intermediate_extraction_after_node(corruption_dag, final_result_value,
                                                       f"data-corruption-test-{column}-{corruption_percentage}")
                first_op_requiring_corruption = find_first_op_modifying_a_column(corruption_dag, column, True)
                operator_to_apply_corruption_after = list(dag.predecessors(first_op_requiring_corruption))[-1]

                def corrupt_df(pandas_df, corruption_percentage, corruption_function, column):
                    # TODO: If we model this as 3 operations instead of one, optimization should be easy
                    completely_corrupted_df = pandas_df.copy()
                    completely_corrupted_df = corruption_function(completely_corrupted_df)
                    corrupt_count = int(len(pandas_df) * corruption_percentage)
                    indexes_to_corrupt = numpy.random.permutation(pandas_df.index)[:corrupt_count]
                    return_df = pandas_df.copy()
                    return_df.loc[indexes_to_corrupt, column] = completely_corrupted_df.loc[indexes_to_corrupt, column]
                    return return_df

                # We need to use partial here to avoid problems with late bindings, see
                #  https://stackoverflow.com/questions/3431676/creating-functions-in-a-loop
                corrupt_df_with_proper_bindings = functools.partial(corrupt_df,
                                                                    corruption_percentage=corruption_percentage,
                                                                    corruption_function=corruption_function,
                                                                    column=column)

                new_corruption_node = DagNode(singleton.get_next_op_id(),
                                              first_op_requiring_corruption.code_location,
                                              OperatorContext(OperatorType.PROJECTION_MODIFY, None),
                                              DagNodeDetails(f"Corrupt {corruption_percentage*100}% of '{column}'",
                                                             operator_to_apply_corruption_after.details.columns),
                                              None,
                                              corrupt_df_with_proper_bindings)
                add_new_node_after_node(corruption_dag, new_corruption_node, operator_to_apply_corruption_after)
                mark_nodes_to_recompute_after_changed_node(corruption_dag, new_corruption_node)
                corruption_dags.append(corruption_dag)
        return corruption_dags  # TODO

    def generate_final_report(self, extracted_plan_results: Dict[str, any]) -> any:
        # TODO: Make this pretty
        results = dict()
        for (column, _) in self.column_to_corruption:
            for corruption_percentage in self.corruption_percentages:
                label = f"data-corruption-test-{column}-{corruption_percentage}"
                results[label] = singleton.labels_to_extracted_plan_results[label]
        return results
