"""
The Data Corruption What-If Analysis
"""
import functools
from types import FunctionType
from typing import Iterable, Dict

import networkx
import numpy
import pandas

from mlwhatif import OperatorType, DagNode, OperatorContext, DagNodeDetails
from mlwhatif.analysis._analysis_utils import find_nodes_by_type, \
    add_new_node_between_nodes, add_intermediate_extraction_after_node, \
    find_dag_location_for_first_op_modifying_column
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
        self._score_nodes_and_linenos = []
        self._analysis_id = (*self.column_to_corruption, *self.corruption_percentages, self.also_corrupt_train)

    @property
    def analysis_id(self):
        return self._analysis_id

    def generate_plans_to_try(self, dag: networkx.DiGraph) \
            -> Iterable[networkx.DiGraph]:
        predict_operators = find_nodes_by_type(dag, OperatorType.PREDICT)
        if len(predict_operators) != 1:
            raise Exception("Currently, DataCorruption only supports pipelines with exactly one predict call which "
                            "must be on the test set!")
        score_operators = find_nodes_by_type(dag, OperatorType.SCORE)
        self._score_nodes_and_linenos = [(node, node.code_location.lineno) for node in score_operators]
        if len(self._score_nodes_and_linenos) != len(set(self._score_nodes_and_linenos)):
            raise Exception("Currently, DataCorruption only supports pipelines where different score operations can "
                            "be uniquely identified by the line number in the code!")
        # TODO: Performance optimisation: deduplication for transformers that process multiple columns at once
        #  For project_modify, we can think about a similar deduplication: splitting operations on multiple columns and
        #  then using a concat in the end.

        corruption_dags = []
        for corruption_percentage in self.corruption_percentages:
            for (column, corruption_function) in self.column_to_corruption:
                train_corruption_location = find_dag_location_for_first_op_modifying_column(column, dag, True)
                test_corruption_location = find_dag_location_for_first_op_modifying_column(column, dag, False)

                # Test set corruption
                test_corruption_result_label = f"data-corruption-test-{column}-{corruption_percentage}"
                corruption_dag = dag.copy()
                self.add_intermediate_extraction_after_score_nodes(corruption_dag, test_corruption_result_label)

                self.add_corruption_in_location(column, corruption_dag, corruption_function, train_corruption_location,
                                                corruption_percentage)
                corruption_dags.append(corruption_dag)

                # Train and test set corruption
                if self.also_corrupt_train is True:
                    test_corruption_result_label = f"data-corruption-train-{column}-{corruption_percentage}"
                    corruption_dag = dag.copy()
                    self.add_intermediate_extraction_after_score_nodes(corruption_dag, test_corruption_result_label)
                    self.add_corruption_in_location(column, corruption_dag, corruption_function,
                                                    train_corruption_location, corruption_percentage)
                    self.add_corruption_in_location(column, corruption_dag, corruption_function,
                                                    test_corruption_location, corruption_percentage)
                    corruption_dags.append(corruption_dag)

        return corruption_dags

    def add_corruption_in_location(self, column, corruption_dag, corruption_function, corruption_location,
                                   corruption_percentage):
        # pylint: disable=too-many-arguments
        """We now know where to apply the corruption, and use this function to actually apply it."""
        new_corruption_node = self.create_corruption_node(column, corruption_function,
                                                          corruption_percentage,
                                                          corruption_location)
        add_new_node_between_nodes(corruption_dag, new_corruption_node, corruption_location)

    def add_intermediate_extraction_after_score_nodes(self, dag: networkx.DiGraph, label: str):
        """Add a new node behind some given node to extract the intermediate result of that given node"""
        node_linenos = []
        for node, lineno in self._score_nodes_and_linenos:
            node_linenos.append(lineno)
            node_label = f"{label}_L{lineno}"
            add_intermediate_extraction_after_node(dag, node, node_label)
        return node_linenos

    @staticmethod
    def create_corruption_node(column, corruption_function, corruption_percentage, corruption_location):
        """Create the node that applies the specified corruption"""
        operator_to_apply_corruption_after, first_op_requiring_corruption = corruption_location

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
        # pylint: disable=too-many-locals
        result_df_columns = []
        result_df_percentages = []
        result_df_metrics = {}
        score_linenos = [lineno for (_, lineno) in self._score_nodes_and_linenos]
        for (column, _) in self.column_to_corruption:
            for corruption_percentage in self.corruption_percentages:
                result_df_columns.append(column)
                result_df_percentages.append(corruption_percentage)
                for lineno in score_linenos:
                    test_label = f"data-corruption-test-{column}-{corruption_percentage}_L{lineno}"
                    test_result_column_name = f"metric_corrupt_test_only_L{lineno}"
                    test_column_values = result_df_metrics.get(test_result_column_name, [])
                    test_column_values.append(singleton.labels_to_extracted_plan_results[test_label])
                    result_df_metrics[test_result_column_name] = test_column_values

                for lineno in score_linenos:
                    train_label = f"data-corruption-train-{column}-{corruption_percentage}_L{lineno}"
                    if train_label in singleton.labels_to_extracted_plan_results:
                        train_and_test_metric = singleton.labels_to_extracted_plan_results[train_label]
                        train_result_column_name = f"metric_corrupt_train_and_test_L{lineno}"
                        train_column_values = result_df_metrics.get(train_result_column_name, [])
                        train_column_values.append(train_and_test_metric)
                        result_df_metrics[train_result_column_name] = train_column_values
        result_df = pandas.DataFrame({'column': result_df_columns,
                                      'corruption_percentage': result_df_percentages,
                                      **result_df_metrics})
        return result_df
