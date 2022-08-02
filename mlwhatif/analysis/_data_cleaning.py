"""
The Data Cleaning What-If Analysis
"""
import dataclasses
from enum import Enum
from functools import partial
from typing import Iterable, Dict, Callable

import networkx
import pandas

from mlwhatif import OperatorType, DagNode, OperatorContext, DagNodeDetails
from mlwhatif.analysis._analysis_utils import find_nodes_by_type, add_intermediate_extraction_after_node, \
    find_dag_location_for_data_patch, \
    add_new_node_after_node, replace_node
from mlwhatif.analysis._cleaning_methods import MissingValueCleaner, DuplicateCleaner, OutlierCleaner, MislabelCleaner
from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis
from mlwhatif.instrumentation._pipeline_executor import singleton


class ErrorType(Enum):
    """
    The different error types supported by the data cleaning what-if analysis
    """
    NUM_MISSING_VALUES = "numerical missing values"
    CAT_MISSING_VALUES = "categorical missing values"
    OUTLIERS = "outliers"
    DUPLICATES = "duplicates"
    MISLABEL = "mislabel"


class PatchType(Enum):
    """
    The different patch types
    """
    DATA_FILTER_PATCH = "data filter patch"
    DATA_TRANSFORMER_PATCH = "transformer patch"
    ESTIMATOR_PATCH = "estimator patch"


@dataclasses.dataclass
class CleaningMethod:
    """
    A DAG Node
    """

    method_name: str
    patch_type: PatchType
    filter_func: Callable or None = None
    fit_or_fit_transform_func: Callable or None = None
    predict_or_fit_func: Callable or None = None
    numeric_only: bool = False
    categorical_only: bool = False

    def __hash__(self):
        return hash(self.method_name)


CLEANING_METHODS_FOR_ERROR_TYPE = {
    ErrorType.NUM_MISSING_VALUES: [
        CleaningMethod("delete", PatchType.DATA_FILTER_PATCH, filter_func=MissingValueCleaner.drop_missing),
        CleaningMethod("impute_num_median", PatchType.DATA_TRANSFORMER_PATCH,
                       fit_or_fit_transform_func=partial(MissingValueCleaner.fit_transform_all, strategy='median'),
                       predict_or_fit_func=MissingValueCleaner.transform_all),
        CleaningMethod("impute_num_mean", PatchType.DATA_TRANSFORMER_PATCH,
                       fit_or_fit_transform_func=partial(MissingValueCleaner.fit_transform_all, strategy='mean'),
                       predict_or_fit_func=MissingValueCleaner.transform_all),
        CleaningMethod("impute_num_mode", PatchType.DATA_TRANSFORMER_PATCH,
                       fit_or_fit_transform_func=partial(MissingValueCleaner.fit_transform_all, strategy='mode'),
                       predict_or_fit_func=MissingValueCleaner.transform_all)
    ],
    ErrorType.CAT_MISSING_VALUES: [
        CleaningMethod("delete", PatchType.DATA_FILTER_PATCH, filter_func=MissingValueCleaner.drop_missing),
        CleaningMethod("impute_cat_mode", PatchType.DATA_TRANSFORMER_PATCH,
                       fit_or_fit_transform_func=partial(MissingValueCleaner.fit_transform_all,
                                                         strategy='mode', cat=True),
                       predict_or_fit_func=partial(MissingValueCleaner.transform_all, cat=True)),
        CleaningMethod("impute_cat_dummy", PatchType.DATA_TRANSFORMER_PATCH,
                       fit_or_fit_transform_func=partial(MissingValueCleaner.fit_transform_all,
                                                         strategy='dummy', cat=True),
                       predict_or_fit_func=partial(MissingValueCleaner.transform_all, cat=True))],
    ErrorType.OUTLIERS: [
        CleaningMethod("clean_SD_impute_mean", PatchType.DATA_TRANSFORMER_PATCH,
                       fit_or_fit_transform_func=partial(OutlierCleaner.fit_transform_all,
                                                         detection_strategy='SD', repair_strategy='mean'),
                       predict_or_fit_func=OutlierCleaner.transform_all),
        CleaningMethod("clean_SD_impute_median", PatchType.DATA_TRANSFORMER_PATCH,
                       fit_or_fit_transform_func=partial(OutlierCleaner.fit_transform_all,
                                                         detection_strategy='SD', repair_strategy='median'),
                       predict_or_fit_func=OutlierCleaner.transform_all),
        CleaningMethod("clean_SD_impute_mode", PatchType.DATA_TRANSFORMER_PATCH,
                       fit_or_fit_transform_func=partial(OutlierCleaner.fit_transform_all,
                                                         detection_strategy='SD', repair_strategy='mode'),
                       predict_or_fit_func=OutlierCleaner.transform_all),
        CleaningMethod("clean_IQR_impute_mean", PatchType.DATA_TRANSFORMER_PATCH,
                       fit_or_fit_transform_func=partial(OutlierCleaner.fit_transform_all,
                                                         detection_strategy='IQR', repair_strategy='mean'),
                       predict_or_fit_func=OutlierCleaner.transform_all),
        CleaningMethod("clean_IQR_impute_median", PatchType.DATA_TRANSFORMER_PATCH,
                       fit_or_fit_transform_func=partial(OutlierCleaner.fit_transform_all,
                                                         detection_strategy='IQR',
                                                         repair_strategy='median'),
                       predict_or_fit_func=OutlierCleaner.transform_all),
        CleaningMethod("clean_IQR_impute_mode", PatchType.DATA_TRANSFORMER_PATCH,
                       fit_or_fit_transform_func=partial(OutlierCleaner.fit_transform_all,
                                                         detection_strategy='IQR', repair_strategy='mode'),
                       predict_or_fit_func=OutlierCleaner.transform_all),
        CleaningMethod("clean_IF_impute_mean", PatchType.DATA_TRANSFORMER_PATCH,
                       fit_or_fit_transform_func=partial(OutlierCleaner.fit_transform_all,
                                                         detection_strategy='IF', repair_strategy='mean'),
                       predict_or_fit_func=OutlierCleaner.transform_all),
        CleaningMethod("clean_IF_impute_median", PatchType.DATA_TRANSFORMER_PATCH,
                       fit_or_fit_transform_func=partial(OutlierCleaner.fit_transform_all,
                                                         detection_strategy='IF', repair_strategy='median'),
                       predict_or_fit_func=OutlierCleaner.transform_all),
        CleaningMethod("clean_IF_impute_mode", PatchType.DATA_TRANSFORMER_PATCH,
                       fit_or_fit_transform_func=partial(OutlierCleaner.fit_transform_all,
                                                         detection_strategy='IF', repair_strategy='mode'),
                       predict_or_fit_func=OutlierCleaner.transform_all)
    ],
    ErrorType.DUPLICATES: [
        CleaningMethod("delete", PatchType.DATA_FILTER_PATCH, filter_func=DuplicateCleaner.drop_duplicates)
    ],
    ErrorType.MISLABEL: [
        CleaningMethod("cleanlab", PatchType.ESTIMATOR_PATCH, fit_or_fit_transform_func=MislabelCleaner.fit_cleanlab)
    ]
}


class DataCleaning(WhatIfAnalysis):
    """
    The Data Cleaning What-If Analysis
    """

    def __init__(self, columns_with_error: Dict[str or None, ErrorType]):
        self._columns_with_error = list(columns_with_error.items())
        self._score_nodes_and_linenos = []
        self._analysis_id = (*self._columns_with_error,)

    @property
    def analysis_id(self):
        return self._analysis_id

    def generate_plans_to_try(self, dag: networkx.DiGraph) \
            -> Iterable[networkx.DiGraph]:
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
        cleaning_dags = []
        for column, error in self._columns_with_error:
            for cleaning_method in CLEANING_METHODS_FOR_ERROR_TYPE[error]:
                cleaning_result_label = f"data-cleaning-{column}-{cleaning_method.method_name}"
                cleaning_dag = dag.copy()
                self.add_intermediate_extraction_after_score_nodes(cleaning_dag, cleaning_result_label)
                if cleaning_method.patch_type == PatchType.DATA_FILTER_PATCH:
                    train_first_node_with_column = find_dag_location_for_data_patch(column, dag, True)
                    test_first_node_with_column = find_dag_location_for_data_patch(column, dag, False)

                    filter_func = partial(cleaning_method.filter_func, column=column)

                    new_train_cleaning_node = DagNode(singleton.get_next_op_id(),
                                                      train_first_node_with_column.code_location,
                                                      OperatorContext(OperatorType.SELECTION, None),
                                                      DagNodeDetails(
                                                          f"Clean {column}: {cleaning_method.method_name}",
                                                          train_first_node_with_column.details.columns),
                                                      None,
                                                      filter_func)
                    add_new_node_after_node(cleaning_dag, new_train_cleaning_node, train_first_node_with_column)
                    # Is this second step really necessary? Is modifying the test set a good idea?
                    if test_first_node_with_column != train_first_node_with_column:
                        new_test_cleaning_node = DagNode(singleton.get_next_op_id(),
                                                         test_first_node_with_column.code_location,
                                                         OperatorContext(OperatorType.SELECTION, None),
                                                         DagNodeDetails(
                                                             f"Clean {column}: {cleaning_method.method_name}",
                                                             train_first_node_with_column.details.columns),
                                                         None,
                                                         filter_func)
                        add_new_node_after_node(cleaning_dag, new_test_cleaning_node, test_first_node_with_column)
                elif cleaning_method.patch_type == PatchType.DATA_TRANSFORMER_PATCH:
                    train_first_node_with_column = find_dag_location_for_data_patch(column, dag, True)
                    test_first_node_with_column = find_dag_location_for_data_patch(column, dag, False)

                    fit_transform = partial(cleaning_method.fit_or_fit_transform_func, column=column)
                    transform = partial(cleaning_method.predict_or_fit_func, column=column)
                    new_train_cleaning_node = DagNode(singleton.get_next_op_id(),
                                                      train_first_node_with_column.code_location,
                                                      OperatorContext(OperatorType.TRANSFORMER, None),
                                                      DagNodeDetails(
                                                          f"Clean {column}: {cleaning_method.method_name} "
                                                          f"fit_transform",
                                                          train_first_node_with_column.details.columns),
                                                      None,
                                                      fit_transform)
                    add_new_node_after_node(cleaning_dag, new_train_cleaning_node, train_first_node_with_column)

                    if test_first_node_with_column != train_first_node_with_column:
                        new_test_cleaning_node = DagNode(singleton.get_next_op_id(),
                                                         test_first_node_with_column.code_location,
                                                         OperatorContext(OperatorType.TRANSFORMER, None),
                                                         DagNodeDetails(
                                                             f"Clean {column}: {cleaning_method.method_name} "
                                                             f"transform",
                                                             train_first_node_with_column.details.columns),
                                                         None,
                                                         transform)
                        add_new_node_after_node(cleaning_dag, new_test_cleaning_node, test_first_node_with_column,
                                                arg_index=1)
                        cleaning_dag.add_edge(new_train_cleaning_node, new_test_cleaning_node, arg_index=0)
                elif cleaning_method.patch_type == PatchType.ESTIMATOR_PATCH:
                    estimator_nodes = find_nodes_by_type(dag, OperatorType.ESTIMATOR)
                    if len(estimator_nodes) != 1:
                        raise Exception(
                            "Currently, DataCorruption only supports pipelines with exactly one estimator!")
                    estimator_node = estimator_nodes[0]
                    new_processing_func = partial(cleaning_method.fit_or_fit_transform_func,
                                                  make_classifier_func=estimator_node.make_classifier_func)
                    new_description = f"Cleanlab patched {estimator_node.details.description}"
                    new_estimator_node = DagNode(singleton.get_next_op_id(),
                                                 estimator_node.code_location,
                                                 estimator_node.operator_info,
                                                 DagNodeDetails(new_description, estimator_node.details.columns),
                                                 estimator_node.optional_code_info,
                                                 new_processing_func)
                    replace_node(cleaning_dag, estimator_node, new_estimator_node)
                else:
                    raise Exception(f"Unknown patch type: {cleaning_method.patch_type}!")
                cleaning_dags.append(cleaning_dag)
        return cleaning_dags

    def add_intermediate_extraction_after_score_nodes(self, dag: networkx.DiGraph, label: str):
        """Add a new node behind some given node to extract the intermediate result of that given node"""
        node_linenos = []
        for node, lineno in self._score_nodes_and_linenos:
            node_linenos.append(lineno)
            node_label = f"{label}_L{lineno}"
            add_intermediate_extraction_after_node(dag, node, node_label)
        return node_linenos

    def generate_final_report(self, extracted_plan_results: Dict[str, any]) -> any:
        # pylint: disable=too-many-locals
        result_df_columns = []
        result_df_errors = []
        result_df_cleaning_methods = []
        result_df_metrics = {}
        score_linenos = [lineno for (_, lineno) in self._score_nodes_and_linenos]
        for (column, error_type) in self._columns_with_error:
            for cleaning_method in CLEANING_METHODS_FOR_ERROR_TYPE[error_type]:
                result_df_columns.append(column)
                result_df_errors.append(error_type.value)
                result_df_cleaning_methods.append(cleaning_method.method_name)
                for lineno in score_linenos:
                    cleaning_result_label = f"data-cleaning-{column}-{cleaning_method.method_name}_L{lineno}"
                    test_result_column_name = f"metric_L{lineno}"
                    test_column_values = result_df_metrics.get(test_result_column_name, [])
                    test_column_values.append(singleton.labels_to_extracted_plan_results[cleaning_result_label])
                    result_df_metrics[test_result_column_name] = test_column_values
        result_df = pandas.DataFrame({'column': result_df_columns,
                                      'error': result_df_errors,
                                      'cleaning_method': result_df_cleaning_methods,
                                      **result_df_metrics})
        return result_df
