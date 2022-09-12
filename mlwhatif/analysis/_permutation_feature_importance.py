"""
The Permutation Feature Importance What-If Analysis
"""
from functools import partial
from types import FunctionType
from typing import Iterable, Dict, Callable, Union, List

import networkx
import numpy
import pandas

from mlwhatif import OperatorType, DagNode, OperatorContext, DagNodeDetails, BasicCodeLocation
from mlwhatif.analysis._analysis_utils import find_nodes_by_type, get_sorted_parent_nodes
from mlwhatif.analysis._data_corruption import CorruptionType, CORRUPTION_FUNCS_FOR_CORRUPTION_TYPES
from mlwhatif.analysis._patch_creation import get_intermediate_extraction_patch_after_score_nodes
from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis
from mlwhatif.execution._patches import DataProjection, PipelinePatch, UdfSplitInfo
from mlwhatif.execution._pipeline_executor import singleton


class PermutationFeatureImportance(WhatIfAnalysis):
    """
    The Permutation Feature Importance What-If Analysis
    """

    def __init__(self, restrict_to_columns: Iterable[str] or None = None):
        # pylint: disable=unsubscriptable-object
        if restrict_to_columns is None:
            self._columns_to_test = None
        else:
            self._columns_to_test = list(restrict_to_columns)
        self._score_nodes_and_linenos = []
        self._analysis_id = (*(self._columns_to_test or []),)

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

        if self._columns_to_test is None:
            self._columns_to_test = self.get_columns_used_as_feature(dag)
        # TODO: Performance optimisation: deduplication for transformers that process multiple columns at once
        #  For project_modify, we can think about a similar deduplication: splitting operations on multiple columns and
        #  then using a concat in the end. Then we can perform additional optimizations.

        permutation_patch_sets = []
        for column in self._columns_to_test:
            patches_for_variant = []

            result_label = f"permutation-feature-importance-{column}"
            extraction_nodes = get_intermediate_extraction_patch_after_score_nodes(singleton, self,
                                                                                   result_label,
                                                                                   self._score_nodes_and_linenos)
            patches_for_variant.extend(extraction_nodes)
            permutation_node = self.create_permutation_node(column)
            patch = DataProjection(singleton.get_next_patch_id(), self, True, permutation_node, False, column,
                                   [column], None)
            patches_for_variant.append(patch)
            permutation_patch_sets.append(patches_for_variant)
        return permutation_patch_sets

    @staticmethod
    def create_permutation_node(column):
        """Create the node that applies the permutation"""

        def permute_columns(pandas_df, column):
            return_df = pandas_df.copy()
            return_df[column] = return_df[column].sample(frac=1.).values
            return return_df

        # We need to use partial here to avoid problems with late bindings, see
        #  https://stackoverflow.com/questions/3431676/creating-functions-in-a-loop
        description = f"Permute '{column}' randomly"
        permute_columns_with_proper_bindings = partial(permute_columns, column=column)
        new_perutation_node = DagNode(singleton.get_next_op_id(),
                                      BasicCodeLocation("DataCorruption", None),
                                      OperatorContext(OperatorType.PROJECTION_MODIFY, None),
                                      DagNodeDetails(description, None),
                                      None,
                                      permute_columns_with_proper_bindings)
        return new_perutation_node

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
            test_result_column_name = f"{score_description}_L{lineno}"
            test_column_values = result_df_metrics.get(test_result_column_name, [])
            metric_value = singleton.labels_to_extracted_plan_results[original_pipeline_result_label]
            test_column_values.append(metric_value)
            result_df_metrics[test_result_column_name] = test_column_values

        for column in self._columns_to_test:
            result_df_columns.append(column)
            for (score_description, lineno) in score_description_and_linenos:
                test_label = f"permutation-feature-importance-{column}_L{lineno}"
                test_result_column_name = f"{score_description}_L{lineno}"
                test_column_values = result_df_metrics.get(test_result_column_name, [])
                test_column_values.append(singleton.labels_to_extracted_plan_results[test_label])
                result_df_metrics[test_result_column_name] = test_column_values
        result_df = pandas.DataFrame({'column': result_df_columns, **result_df_metrics})
        return result_df

    @staticmethod
    def get_columns_used_as_feature(dag) -> List[str]:
        test_data_operator = find_nodes_by_type(dag, OperatorType.TEST_DATA)[0]
        if test_data_operator.details.columns != ["array"]:
            feature_columns = test_data_operator.details.columns
        else:
            feature_columns = set()
            transformer_ops = find_nodes_by_type(dag, OperatorType.TRANSFORMER)
            for transformer in transformer_ops:
                transformer_parent = get_sorted_parent_nodes(dag, transformer)[-1]
                feature_columns.update(transformer_parent.details.columns)
            feature_columns.discard("array")
            feature_columns = list(feature_columns)
        return feature_columns
