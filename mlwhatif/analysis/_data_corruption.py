"""
The Data Corruption What-If Analysis
"""
from functools import partial
from types import FunctionType
from typing import Iterable, Dict, Callable, Union

import networkx
import numpy
import pandas

from mlwhatif import OperatorType, DagNode, OperatorContext, DagNodeDetails, BasicCodeLocation
from mlwhatif.analysis._analysis_utils import find_nodes_by_type
from mlwhatif.analysis._patch_creation import get_intermediate_extraction_patch_after_score_nodes
from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis
from mlwhatif.execution._patches import DataProjection, Patch
from mlwhatif.instrumentation._pipeline_executor import singleton


class DataCorruption(WhatIfAnalysis):
    """
    The Data Corruption What-If Analysis
    """

    def __init__(self,
                 column_to_corruption: Dict[str, FunctionType],
                 corruption_percentages: Iterable[Union[float, Callable]] or None = None,
                 also_corrupt_train: bool = False):
        # pylint: disable=unsubscriptable-object
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

    def generate_plans_to_try(self, dag: networkx.DiGraph) -> Iterable[Iterable[Patch]]:
        # pylint: disable=too-many-locals
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

        corruption_patch_sets = []
        for corruption_percentage_index, corruption_percentage in enumerate(self.corruption_percentages):
            for (column_corruption_tuple_index, (column, corruption_function)) in enumerate(self.column_to_corruption):
                patches_for_variant = []

                # Test set corruption
                test_corruption_result_label = f"data-corruption-test-{column}-{corruption_percentage}"
                extraction_nodes = get_intermediate_extraction_patch_after_score_nodes(singleton, self,
                                                                                       test_corruption_result_label,
                                                                                       self._score_nodes_and_linenos)
                patches_for_variant.extend(extraction_nodes)
                corruption_node, corrupt_func, index_selection_func = self.create_corruption_node(
                    column, corruption_function, corruption_percentage)
                if isinstance(corruption_percentage, float):
                    only_reads_column = [column]
                    maybe_selectivity_info = corruption_percentage
                else:
                    only_reads_column = None  # TODO: Support this case properly in optimizations
                    maybe_selectivity_info = None
                patch = DataProjection(singleton.get_next_patch_id(), self, True, corruption_node, False, column,
                                       only_reads_column, index_selection_func, corruption_percentage_index,
                                       corrupt_func, column_corruption_tuple_index, maybe_selectivity_info)
                patches_for_variant.append(patch)
                corruption_patch_sets.append(patches_for_variant)

                # Train and test set corruption
                if self.also_corrupt_train is True:
                    patches_for_variant = []
                    test_corruption_result_label = f"data-corruption-train-{column}-{corruption_percentage}"
                    extraction_nodes = get_intermediate_extraction_patch_after_score_nodes(
                        singleton, self, test_corruption_result_label, self._score_nodes_and_linenos)
                    patches_for_variant.extend(extraction_nodes)
                    corruption_node, corrupt_func, index_selection_func = self.create_corruption_node(
                        column, corruption_function, corruption_percentage)
                    patch = DataProjection(singleton.get_next_patch_id(), self, True, corruption_node, False, column,
                                           only_reads_column, index_selection_func, corruption_percentage_index,
                                           corrupt_func, column_corruption_tuple_index, maybe_selectivity_info)
                    patches_for_variant.append(patch)
                    corruption_node, corrupt_func, index_selection_func = self.create_corruption_node(
                        column, corruption_function, corruption_percentage)
                    patch = DataProjection(singleton.get_next_patch_id(), self, True, corruption_node, True, column,
                                           only_reads_column, index_selection_func, corruption_percentage_index,
                                           corrupt_func, column_corruption_tuple_index, maybe_selectivity_info)
                    patches_for_variant.append(patch)
                    corruption_patch_sets.append(patches_for_variant)

        return corruption_patch_sets

    @staticmethod
    def create_corruption_node(column, corruption_function, corruption_percentage_or_selection_function):
        """Create the node that applies the specified corruption"""

        def corruption_index_selection(pandas_df, corruption_percentage):
            corrupt_count = int(len(pandas_df) * corruption_percentage)
            indexes_to_corrupt = numpy.random.permutation(pandas_df.index)[:corrupt_count]
            return indexes_to_corrupt

        def corrupt_df(pandas_df, corruption_index_selection_func, corruption_function, column):
            # TODO: If we model this as 3 operations instead of one, optimization should be easy
            # TODO: Think about when we actually want to be defensive and call copy and when not
            # TODO: Think about datatypes. corruption_function currently assumes pandas DataFrames.
            #  We may want to automatically convert data formats as needed, here and in other places.
            completely_corrupted_df = pandas_df.copy()
            completely_corrupted_df = corruption_function(completely_corrupted_df)
            indexes_to_corrupt = corruption_index_selection_func(pandas_df)
            return_df = pandas_df.copy()
            return_df.loc[indexes_to_corrupt, column] = completely_corrupted_df.loc[indexes_to_corrupt, column]
            return return_df

        # We need to use partial here to avoid problems with late bindings, see
        #  https://stackoverflow.com/questions/3431676/creating-functions-in-a-loop
        if isinstance(corruption_percentage_or_selection_function, float):
            index_selection_with_proper_bindings = partial(corruption_index_selection,
                                                           corruption_percentage=
                                                           corruption_percentage_or_selection_function)
            description = f"Corrupt {corruption_percentage_or_selection_function * 100}% of '{column}'"
        else:
            index_selection_with_proper_bindings = corruption_percentage_or_selection_function
            description = f"Corrupt '{column}' with custom corruption index selection"
        corrupt_df_with_proper_bindings = partial(corrupt_df,
                                                  corruption_index_selection_func=index_selection_with_proper_bindings,
                                                  corruption_function=corruption_function,
                                                  column=column)
        new_corruption_node = DagNode(singleton.get_next_op_id(),
                                      BasicCodeLocation("DataCorruption", None),
                                      OperatorContext(OperatorType.PROJECTION_MODIFY, None),
                                      DagNodeDetails(description, None),
                                      None,
                                      corrupt_df_with_proper_bindings)
        return new_corruption_node, corruption_function, index_selection_with_proper_bindings

    def generate_final_report(self, extracted_plan_results: Dict[str, any]) -> any:
        # pylint: disable=too-many-locals
        result_df_columns = []
        result_df_percentages = []
        result_df_metrics = {}
        score_description_and_linenos = [(score_node.details.description, lineno)
                                         for (score_node, lineno) in self._score_nodes_and_linenos]
        for (column, _) in self.column_to_corruption:
            for corruption_percentage in self.corruption_percentages:
                result_df_columns.append(column)

                if isinstance(corruption_percentage, float):
                    sanitized_corruption_percentage = corruption_percentage
                else:
                    sanitized_corruption_percentage = corruption_percentage.__name__  # pylint: disable=no-member
                result_df_percentages.append(sanitized_corruption_percentage)

                for (score_description, lineno) in score_description_and_linenos:
                    test_label = f"data-corruption-test-{column}-{corruption_percentage}_L{lineno}"
                    test_result_column_name = f"{score_description}_corrupt_test_only_L{lineno}"
                    test_column_values = result_df_metrics.get(test_result_column_name, [])
                    test_column_values.append(singleton.labels_to_extracted_plan_results[test_label])
                    result_df_metrics[test_result_column_name] = test_column_values

                for (score_description, lineno) in score_description_and_linenos:
                    train_label = f"data-corruption-train-{column}-{corruption_percentage}_L{lineno}"
                    if train_label in singleton.labels_to_extracted_plan_results:
                        train_and_test_metric = singleton.labels_to_extracted_plan_results[train_label]
                        train_result_column_name = f"{score_description}_corrupt_train_and_test_L{lineno}"
                        train_column_values = result_df_metrics.get(train_result_column_name, [])
                        train_column_values.append(train_and_test_metric)
                        result_df_metrics[train_result_column_name] = train_column_values
        result_df = pandas.DataFrame({'column': result_df_columns,
                                      'corruption_percentage': result_df_percentages,
                                      **result_df_metrics})
        return result_df
