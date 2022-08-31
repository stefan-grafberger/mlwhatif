"""
The Model Variants What-If Analysis, which we use as an example to showcase common subexpression elimination
"""
from functools import partial
from typing import Iterable, Dict, Callable

import networkx
import pandas

from mlwhatif import OperatorType, DagNode, DagNodeDetails, BasicCodeLocation, OperatorContext
from mlwhatif.analysis._analysis_utils import find_nodes_by_type
from mlwhatif.analysis._patch_creation import get_intermediate_extraction_patch_after_score_nodes
from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis
from mlwhatif.execution._patches import PipelinePatch, ModelPatch, DataFiltering
from mlwhatif.execution._pipeline_executor import singleton


class DataFilterVariants(WhatIfAnalysis):
    """
    A Data Filtering Variants What-If Analysis, currently only used for benchmarking optimizations
    """

    def __init__(self, filter_name_column_and_filter_functions: Dict[str, tuple[str, Callable]]):
        # pylint: disable=unsubscriptable-object
        self._filter_name_column_and_filter_functions = list(filter_name_column_and_filter_functions.items())
        self._score_nodes_and_linenos = []
        self._analysis_id = (*self._filter_name_column_and_filter_functions,)

    @property
    def analysis_id(self):
        return self._analysis_id

    def generate_plans_to_try(self, dag: networkx.DiGraph) -> Iterable[Iterable[PipelinePatch]]:
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

        corruption_patch_sets = []
        for filter_description, (column, filter_function) in self._filter_name_column_and_filter_functions:
            patches_for_variant = []

            test_corruption_result_label = f"model-variant-{filter_description}"
            extraction_nodes = get_intermediate_extraction_patch_after_score_nodes(singleton, self,
                                                                                   test_corruption_result_label,
                                                                                   self._score_nodes_and_linenos)
            patches_for_variant.extend(extraction_nodes)

            filter_patches = self._get_filter_patches(filter_description, column, filter_function)
            patches_for_variant.extend(filter_patches)

            corruption_patch_sets.append(patches_for_variant)

        return corruption_patch_sets

    def _get_filter_patches(self, filter_description, column, filter_function):
        filter_patches = []

        new_train_cleaning_node = DagNode(singleton.get_next_op_id(),
                                          BasicCodeLocation("Data Filtering Variants", None),
                                          OperatorContext(OperatorType.SELECTION, None),
                                          DagNodeDetails(
                                              f"Filter {column}: {filter_description}", None),
                                          None,
                                          filter_function)
        filter_patch_train = DataFiltering(singleton.get_next_patch_id(), self, True,
                                           new_train_cleaning_node, True, [column])
        filter_patches.append(filter_patch_train)

        new_test_cleaning_node = DagNode(singleton.get_next_op_id(),
                                         BasicCodeLocation("Data Filtering Variants", None),
                                         OperatorContext(OperatorType.SELECTION, None),
                                         DagNodeDetails(
                                             f"Filter {column}: {filter_description}", None),
                                         None,
                                         filter_function)
        filter_patch_test = DataFiltering(singleton.get_next_patch_id(), self, True,
                                          new_test_cleaning_node, False, [column])
        filter_patches.append(filter_patch_test)
        return filter_patches

    def generate_final_report(self, extracted_plan_results: Dict[str, any]) -> any:
        # pylint: disable=too-many-locals
        result_df_filter_variants = []
        result_df_columns = []
        result_df_metrics = {}
        score_description_and_linenos = [(score_node.details.description, lineno)
                                         for (score_node, lineno) in self._score_nodes_and_linenos]

        result_df_filter_variants.append("original")
        result_df_columns.append(None)
        for (score_description, lineno) in score_description_and_linenos:
            original_pipeline_result_label = f"original_L{lineno}"
            test_result_column_name = f"{score_description}_L{lineno}"
            test_column_values = result_df_metrics.get(test_result_column_name, [])
            metric_value = singleton.labels_to_extracted_plan_results[original_pipeline_result_label]
            test_column_values.append(metric_value)
            result_df_metrics[test_result_column_name] = test_column_values

        for filter_description, (column, _) in self._filter_name_column_and_filter_functions:
            result_df_filter_variants.append(filter_description)
            result_df_columns.append(column)

            for (score_description, lineno) in score_description_and_linenos:
                test_label = f"model-variant-{filter_description}_L{lineno}"
                test_result_column_name = f"{score_description}_L{lineno}"
                test_column_values = result_df_metrics.get(test_result_column_name, [])
                test_column_values.append(singleton.labels_to_extracted_plan_results[test_label])
                result_df_metrics[test_result_column_name] = test_column_values

        result_df = pandas.DataFrame({'filter_variant': result_df_filter_variants,
                                      'column': result_df_columns,
                                      **result_df_metrics})
        return result_df
