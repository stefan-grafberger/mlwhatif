"""
The Data Corruption What-If Analysis
"""
from enum import Enum
from functools import partial
from types import FunctionType
from typing import Iterable, Dict, Callable, Union, Tuple, List

import networkx
import numpy
import pandas
from jenga.corruptions.generic import CategoricalShift
from jenga.corruptions.numerical import Scaling, GaussianNoise
from jenga.corruptions.text import BrokenCharacters

from mlmq import OperatorType, DagNode, OperatorContext, DagNodeDetails, BasicCodeLocation
from mlmq.analysis._analysis_utils import find_nodes_by_type
from mlmq.analysis._patch_creation import get_intermediate_extraction_patch_after_score_nodes
from mlmq.analysis._what_if_analysis import WhatIfAnalysis
from mlmq.execution._patches import DataProjection, PipelinePatch, UdfSplitInfo
from mlmq.execution._pipeline_executor import singleton


class CorruptionType(Enum):
    """
    The different corruption functions known to mlmq, but custom corruption functions can also be used
    """
    CATEGORICAL_SHIFT = "categorical shift"
    MISSING_VALUES = "missing values"
    BROKEN_CHARACTERS = "broken characters"
    GAUSSIAN_NOISE = "gaussian noise"
    SCALING = "scaling"


def corrupt_broken_characters(pandas_df, column):
    """Corrupt broken characters that may be in a pandas df, but may also be in a different format"""
    if isinstance(pandas_df, pandas.DataFrame):
        result = BrokenCharacters(column=column, fraction=1.).transform(pandas_df)
    else:
        pandas_df = pandas.DataFrame(pandas_df)
        result = BrokenCharacters(column=column, fraction=1.).transform(pandas_df)
        result = result[column]
    return result


CORRUPTION_FUNCS_FOR_CORRUPTION_TYPES = {
    CorruptionType.CATEGORICAL_SHIFT:
        lambda pandas_df, column: CategoricalShift(column=column, fraction=1.).transform(pandas_df),
    CorruptionType.MISSING_VALUES: lambda pandas_df, column: Scaling(column=column, fraction=1.).transform(pandas_df),
    CorruptionType.BROKEN_CHARACTERS: corrupt_broken_characters,
    CorruptionType.GAUSSIAN_NOISE:
        lambda pandas_df, column: GaussianNoise(column=column, fraction=1.).transform(pandas_df),
    CorruptionType.SCALING: lambda pandas_df, column: Scaling(column=column, fraction=1.).transform(pandas_df)
}


class DataCorruption(WhatIfAnalysis):
    """
    The Data Corruption What-If Analysis
    """

    def __init__(self,
                 column_to_corruption: List[Tuple[str, Union[FunctionType, CorruptionType]]],
                 corruption_percentages: Iterable[Union[float, Callable]] or None = None,
                 also_corrupt_train: bool = False):
        # pylint: disable=unsubscriptable-object
        self.also_corrupt_train = also_corrupt_train
        self.column_to_corruption = column_to_corruption
        if corruption_percentages is None:
            self.corruption_percentages = [0.2, 0.5, 0.9]
        else:
            self.corruption_percentages = corruption_percentages
        self._score_nodes_and_linenos = []
        self._analysis_id = (*self.column_to_corruption, *self.corruption_percentages, self.also_corrupt_train)

    @property
    def analysis_id(self):
        return self._analysis_id

    def generate_plans_to_try(self, dag: networkx.DiGraph) -> Iterable[Iterable[PipelinePatch]]:
        # pylint: disable=too-many-locals
        column_to_corruption_with_replaced_corruption_types = []
        for column, corruption_function in self.column_to_corruption:
            if isinstance(corruption_function, CorruptionType):
                # noinspection PyTypeChecker
                column_to_corruption_with_replaced_corruption_types.append(
                    (column, partial(CORRUPTION_FUNCS_FOR_CORRUPTION_TYPES[corruption_function], column=column)))
            else:
                column_to_corruption_with_replaced_corruption_types.append((column, corruption_function))
        self.column_to_corruption = column_to_corruption_with_replaced_corruption_types

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
                                       only_reads_column,
                                       UdfSplitInfo(corruption_percentage_index, index_selection_func,
                                                    column_corruption_tuple_index, corrupt_func, column,
                                                    maybe_selectivity_info))
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
                                           only_reads_column,
                                           UdfSplitInfo(corruption_percentage_index, index_selection_func,
                                                        column_corruption_tuple_index, corrupt_func, column,
                                                        maybe_selectivity_info))
                    patches_for_variant.append(patch)
                    corruption_node, corrupt_func, index_selection_func = self.create_corruption_node(
                        column, corruption_function, corruption_percentage)
                    patch = DataProjection(singleton.get_next_patch_id(), self, True, corruption_node, True, column,
                                           only_reads_column,
                                           UdfSplitInfo(corruption_percentage_index, index_selection_func,
                                                        column_corruption_tuple_index, corrupt_func, column,
                                                        maybe_selectivity_info))
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
            if isinstance(pandas_df, pandas.Series):
                pandas_df = pandas.DataFrame(pandas_df)
                was_series = True
            else:
                was_series = False
            indexes_to_corrupt = corruption_index_selection_func(pandas_df)
            return_df = pandas_df.copy()
            corrupted_df = return_df.loc[indexes_to_corrupt, :]
            completely_corrupted_df = corruption_function(corrupted_df)
            return_df.loc[indexes_to_corrupt, column] = completely_corrupted_df[column]
            if was_series is True:
                return_df = return_df[column]
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

        result_df_columns.append(None)
        result_df_percentages.append(None)
        for (score_description, lineno) in score_description_and_linenos:
            original_pipeline_result_label = f"original_L{lineno}"
            test_result_column_name = f"{score_description}_corrupt_test_only_L{lineno}"
            test_column_values = result_df_metrics.get(test_result_column_name, [])
            metric_value = singleton.labels_to_extracted_plan_results[original_pipeline_result_label]
            test_column_values.append(metric_value)
            result_df_metrics[test_result_column_name] = test_column_values
            if self.also_corrupt_train is True:
                train_result_column_name = f"{score_description}_corrupt_train_and_test_L{lineno}"
                train_column_values = result_df_metrics.get(train_result_column_name, [])
                train_column_values.append(metric_value)
                result_df_metrics[train_result_column_name] = train_column_values

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

                if self.also_corrupt_train is True:
                    for (score_description, lineno) in score_description_and_linenos:
                        train_label = f"data-corruption-train-{column}-{corruption_percentage}_L{lineno}"
                        train_and_test_metric = singleton.labels_to_extracted_plan_results[train_label]
                        train_result_column_name = f"{score_description}_corrupt_train_and_test_L{lineno}"
                        train_column_values = result_df_metrics.get(train_result_column_name, [])
                        train_column_values.append(train_and_test_metric)
                        result_df_metrics[train_result_column_name] = train_column_values
        result_df = pandas.DataFrame({'column': result_df_columns,
                                      'corruption_percentage': result_df_percentages,
                                      **result_df_metrics})
        return result_df
