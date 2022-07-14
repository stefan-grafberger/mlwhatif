"""
The Interface for the What-If Analyses
"""
import functools
from types import FunctionType
from typing import Iterable, Dict

import networkx
import numpy
import pandas

from mlwhatif import OperatorType, DagNode, OperatorContext, DagNodeDetails
from mlwhatif.analysis._analysis_utils import add_intermediate_extraction_after_node, find_nodes_by_type, \
    find_first_op_modifying_a_column, add_new_node_after_node
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
        else:
            self.corruption_percentages = corruption_percentages
        self._analysis_id = (*self.column_to_corruption, *self.corruption_percentages, self.also_corrupt_train)

    @property
    def analysis_id(self):
        return self._analysis_id

    def generate_plans_to_try(self, dag: networkx.DiGraph)\
            -> Iterable[networkx.DiGraph]:
        # pylint: disable=cell-var-from-loop
        score_operators = find_nodes_by_type(dag, OperatorType.SCORE)
        if len(score_operators) != 1:
            raise Exception("Currently, DataCorruption only supports pipelines with exactly one score call!")
        final_result_value = score_operators[0]
        # TODO: Performance optimisation: deduplication for transformers that process multiple columns at once

        corruption_dags = []
        for corruption_percentage in self.corruption_percentages:
            for (column, corruption_function) in self.column_to_corruption:
                test_corruption_result_label = f"data-corruption-test-{column}-{corruption_percentage}"
                corruption_dag = dag.copy()
                add_intermediate_extraction_after_node(corruption_dag, final_result_value, test_corruption_result_label)
                self.find_corruption_location_and_add_corruption_node(column, corruption_dag, corruption_function,
                                                                      corruption_percentage, dag, True)
                corruption_dags.append(corruption_dag)

                if self.also_corrupt_train is True:
                    test_corruption_result_label = f"data-corruption-train-{column}-{corruption_percentage}"
                    corruption_dag = dag.copy()
                    add_intermediate_extraction_after_node(corruption_dag, final_result_value,
                                                           test_corruption_result_label)

                    self.find_corruption_location_and_add_corruption_node(column, corruption_dag, corruption_function,
                                                                          corruption_percentage, dag, True)
                    self.find_corruption_location_and_add_corruption_node(column, corruption_dag, corruption_function,
                                                                          corruption_percentage, dag, False)

                    corruption_dags.append(corruption_dag)

        return corruption_dags

    def find_corruption_location_and_add_corruption_node(self, column, corruption_dag, corruption_function,
                                                         corruption_percentage, dag, test_not_train):
        first_op_requiring_corruption = find_first_op_modifying_a_column(corruption_dag, column, test_not_train)
        operator_to_apply_corruption_after = list(dag.predecessors(first_op_requiring_corruption))[-1]
        new_corruption_node = self.create_corruption_node(column, corruption_function, corruption_percentage,
                                                          first_op_requiring_corruption,
                                                          operator_to_apply_corruption_after)
        add_new_node_after_node(corruption_dag, new_corruption_node, operator_to_apply_corruption_after)

    @staticmethod
    def create_corruption_node(column, corruption_function, corruption_percentage, first_op_requiring_corruption,
                               operator_to_apply_corruption_after):
        """Create the node that applies the specified corruption"""
        def corrupt_df(pandas_df, corruption_percentage, corruption_function, column):
            # TODO: If we model this as 3 operations instead of one, optimization should be easy
            # TODO: Think about when we actually want to be defensive and call copy and when not
            # TODO: Think about datatypes. corruption_function currently assumes pandas DataFrames.
            #  We may want to automatically convert data formats as needed, here and in other places.
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
                                      DagNodeDetails(f"Corrupt {corruption_percentage * 100}% of '{column}'",
                                                     operator_to_apply_corruption_after.details.columns),
                                      None,
                                      corrupt_df_with_proper_bindings)
        return new_corruption_node

    def generate_final_report(self, extracted_plan_results: Dict[str, any]) -> any:
        result_df_columns = []
        result_df_percentages = []
        result_df_metrics_corrupt_test_only = []
        result_df_metrics_corrupt_train_and_test = []
        for (column, _) in self.column_to_corruption:
            for corruption_percentage in self.corruption_percentages:
                test_label = f"data-corruption-test-{column}-{corruption_percentage}"
                result_df_columns.append(column)
                result_df_percentages.append(corruption_percentage)
                result_df_metrics_corrupt_test_only.append(singleton.labels_to_extracted_plan_results[test_label])

                train_label = f"data-corruption-train-{column}-{corruption_percentage}"
                if train_label in singleton.labels_to_extracted_plan_results:
                    train_and_test_metric = singleton.labels_to_extracted_plan_results[train_label]
                    result_df_metrics_corrupt_train_and_test.append(train_and_test_metric)
        if self.also_corrupt_train is True:
            result_df = pandas.DataFrame({'column': result_df_columns,
                                          'corruption_percentage': result_df_percentages,
                                          'metric_corrupt_test_only': result_df_metrics_corrupt_test_only,
                                          'metric_corrupt_train_and_test:': result_df_metrics_corrupt_train_and_test})
        else:
            result_df = pandas.DataFrame({'column': result_df_columns,
                                          'corruption_percentage': result_df_percentages,
                                          'metric_corrupt_test_only': result_df_metrics_corrupt_test_only})
        return result_df
