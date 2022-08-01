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
    add_new_node_after_node
from mlwhatif.analysis._cleaning_methods import MissingValueCleaner, DuplicateCleaner
from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis
from mlwhatif.instrumentation._pipeline_executor import singleton


class ErrorType(Enum):
    """
    The different error types supported by the data cleaning what-if analysis
    """
    MISSING_VALUES = "missing values"
    OUTLIERS = "outliers"
    DUPLICATES = "duplicates"
    MISLABEL = "mislabel"


@dataclasses.dataclass
class CleaningMethod:
    """
    A DAG Node
    """

    method_name: str
    is_filter_not_project: bool
    filter_func: Callable or None = None
    transformer_fit_transform_func: Callable or None = None
    transformer_transform_func: Callable or None = None
    numeric_only: bool = False
    categorical_only: bool = False

    def __hash__(self):
        return hash(self.method_name)


CLEANING_METHODS_FOR_ERROR_TYPE = {
    ErrorType.MISSING_VALUES: [
        CleaningMethod("delete", True, filter_func=MissingValueCleaner.drop_missing),
        CleaningMethod("impute_mean_mode", False,
                       transformer_fit_transform_func=partial(MissingValueCleaner.fit_transform_all,
                                                              num_strategy='mean',
                                                              cat_strategy='mode'),
                       transformer_transform_func=MissingValueCleaner.transform_all)
    ],
    ErrorType.OUTLIERS: [
        CleaningMethod("clean_SD_impute_mean_dummy", False,
                       transformer_fit_transform_func=None,  # TODO: MVCleaner("impute", num="mean", cat="mode")...
                       transformer_transform_func=None)
    ],
    ErrorType.DUPLICATES: [
        CleaningMethod("delete", True, filter_func=DuplicateCleaner.drop_duplicates)
    ],
    ErrorType.MISLABEL: [
        CleaningMethod("cleanlab", False,
                       transformer_fit_transform_func=None,  # TODO: MVCleaner("impute", num="mean", cat="mode")...
                       transformer_transform_func=None)
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
            train_first_node_with_column = find_dag_location_for_data_patch(column, dag, True)
            test_first_node_with_column = find_dag_location_for_data_patch(column, dag, False)
            for cleaning_method in CLEANING_METHODS_FOR_ERROR_TYPE[error]:
                cleaning_result_label = f"data-cleaning-{column}-{cleaning_method.method_name}"
                cleaning_dag = dag.copy()
                self.add_intermediate_extraction_after_score_nodes(cleaning_dag, cleaning_result_label)
                if cleaning_method.is_filter_not_project is True:
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
                else:
                    # TODO: Get first node after train test split instead
                    fit_transform = partial(cleaning_method.transformer_fit_transform_func, column=column)
                    transform = partial(cleaning_method.transformer_transform_func, column=column)
                    new_train_cleaning_node = DagNode(singleton.get_next_op_id(),
                                                      train_first_node_with_column.code_location,
                                                      OperatorContext(OperatorType.TRANSFORMER, None),
                                                      DagNodeDetails(
                                                          f"Clean {column}: {cleaning_method.method_name}",
                                                          train_first_node_with_column.details.columns),
                                                      None,
                                                      fit_transform)
                    add_new_node_after_node(cleaning_dag, new_train_cleaning_node, test_first_node_with_column)

                    if test_first_node_with_column != train_first_node_with_column:
                        new_test_cleaning_node = DagNode(singleton.get_next_op_id(),
                                                         test_first_node_with_column.code_location,
                                                         OperatorContext(OperatorType.SELECTION, None),
                                                         DagNodeDetails(
                                                             f"Clean {column}: {cleaning_method.method_name}",
                                                             train_first_node_with_column.details.columns),
                                                         None,
                                                         transform)
                        add_new_node_after_node(cleaning_dag, new_test_cleaning_node, test_first_node_with_column)
                        # TODO: It would be cleaner to increase the other arg_index numbers by 1 instead
                        #  of using -1 here to substitute 0
                        cleaning_dag.add_edge(new_train_cleaning_node, new_test_cleaning_node, arg_index=-1)
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
