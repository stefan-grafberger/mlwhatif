"""
The Data Cleaning What-If Analysis
"""
import dataclasses
from enum import Enum
from functools import partial
from typing import Iterable, Dict, Callable, List, Tuple

import networkx
import pandas

from mlwhatif import OperatorType, DagNode, OperatorContext, DagNodeDetails, BasicCodeLocation
from mlwhatif.analysis._analysis_utils import find_nodes_by_type, get_columns_used_as_feature
from mlwhatif.analysis._cleaning_methods import MissingValueCleaner, DuplicateCleaner, OutlierCleaner, MislabelCleaner
from mlwhatif.analysis._patch_creation import get_intermediate_extraction_patch_after_score_nodes
from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis
from mlwhatif.execution._patches import DataFiltering, DataTransformer, ModelPatch, PipelinePatch
from mlwhatif.instrumentation._dag_node import OptimizerInfo
from mlwhatif.execution._pipeline_executor import singleton


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
        CleaningMethod("cleanlab_delete", PatchType.ESTIMATOR_PATCH, fit_or_fit_transform_func=partial(MislabelCleaner.fit_cleanlab,
                       correct_not_drop=False)),
        CleaningMethod("cleanlab_update", PatchType.ESTIMATOR_PATCH, fit_or_fit_transform_func=partial(MislabelCleaner.fit_cleanlab,
                       correct_not_drop=True)),
        CleaningMethod("shapley_delete", PatchType.ESTIMATOR_PATCH,
                       fit_or_fit_transform_func=partial(MislabelCleaner.fit_shapley_cleaning, correct_not_drop=True)),
        CleaningMethod("shapley_update", PatchType.ESTIMATOR_PATCH,
                       fit_or_fit_transform_func=partial(MislabelCleaner.fit_shapley_cleaning, correct_not_drop=False))
    ]
}


class DataCleaning(WhatIfAnalysis):
    """
    The Data Cleaning What-If Analysis
    """

    def __init__(self, columns_with_error: dict[str or None, ErrorType] or List[Tuple[str, ErrorType]],
                 parallelism=True):
        if isinstance(columns_with_error, dict):
            self._columns_with_error = list(columns_with_error.items())
        else:
            self._columns_with_error = columns_with_error
        self._score_nodes_and_linenos = []
        self._analysis_id = (*self._columns_with_error,)
        self._parallelism = parallelism

    @property
    def analysis_id(self):
        return self._analysis_id

    def generate_plans_to_try(self, dag: networkx.DiGraph) -> Iterable[Iterable[PipelinePatch]]:
        # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        predict_operators = find_nodes_by_type(dag, OperatorType.PREDICT)
        if len(predict_operators) != 1:
            raise Exception("Currently, DataCorruption only supports pipelines with exactly one predict call which "
                            "must be on the test set!")
        score_operators = find_nodes_by_type(dag, OperatorType.SCORE)
        self._score_nodes_and_linenos = [(node, node.code_location.lineno) for node in score_operators]
        if len(self._score_nodes_and_linenos) != len(set(self._score_nodes_and_linenos)):
            raise Exception("Currently, DataCorruption only supports pipelines where different score operations can "
                            "be uniquely identified by the line number in the code!")
        cleaning_patch_sets = []
        for column, error in self._columns_with_error:
            for cleaning_method in CLEANING_METHODS_FOR_ERROR_TYPE[error]:
                cleaning_result_label = f"data-cleaning-{column}-{cleaning_method.method_name}"
                patches_for_variant = []
                extraction_nodes = get_intermediate_extraction_patch_after_score_nodes(singleton, self,
                                                                                       cleaning_result_label,
                                                                                       self._score_nodes_and_linenos)
                patches_for_variant.extend(extraction_nodes)
                if cleaning_method.patch_type == PatchType.DATA_FILTER_PATCH:
                    feature_cols = set(get_columns_used_as_feature(dag))
                    if column in feature_cols:
                        required_cols = list(feature_cols)
                    else:
                        required_cols = [column]
                    filter_func = partial(cleaning_method.filter_func, column=column)

                    new_train_cleaning_node = DagNode(singleton.get_next_op_id(),
                                                      BasicCodeLocation("Data Cleaning", None),
                                                      OperatorContext(OperatorType.SELECTION, None),
                                                      DagNodeDetails(
                                                          f"Clean {column}: {cleaning_method.method_name}", None),
                                                      None,
                                                      filter_func)
                    filter_patch_train = DataFiltering(singleton.get_next_patch_id(), self, True,
                                                       new_train_cleaning_node, True, required_cols)
                    patches_for_variant.append(filter_patch_train)

                    new_test_cleaning_node = DagNode(singleton.get_next_op_id(),
                                                     BasicCodeLocation("Data Cleaning", None),
                                                     OperatorContext(OperatorType.SELECTION, None),
                                                     DagNodeDetails(
                                                         f"Clean {column}: {cleaning_method.method_name}", None),
                                                     None,
                                                     filter_func)
                    filter_patch_test = DataFiltering(singleton.get_next_patch_id(), self, True,
                                                      new_test_cleaning_node, False, required_cols)
                    patches_for_variant.append(filter_patch_test)
                elif cleaning_method.patch_type == PatchType.DATA_TRANSFORMER_PATCH:
                    fit_transform = partial(cleaning_method.fit_or_fit_transform_func, column=column)
                    transform = partial(cleaning_method.predict_or_fit_func, column=column)
                    new_train_cleaning_node = DagNode(singleton.get_next_op_id(),
                                                      BasicCodeLocation("Data Cleaning", None),
                                                      OperatorContext(OperatorType.TRANSFORMER, None),
                                                      DagNodeDetails(
                                                          f"Clean {column}: {cleaning_method.method_name} "
                                                          f"fit_transform", None),
                                                      None,
                                                      fit_transform)
                    new_test_cleaning_node = DagNode(singleton.get_next_op_id(),
                                                     BasicCodeLocation("Data Cleaning", None),
                                                     OperatorContext(OperatorType.TRANSFORMER, None),
                                                     DagNodeDetails(
                                                         f"Clean {column}: {cleaning_method.method_name} "
                                                         f"transform", None),
                                                     None,
                                                     transform)
                    transform_patch_train = DataTransformer(singleton.get_next_patch_id(), self, True,
                                                            new_train_cleaning_node, new_test_cleaning_node, column)
                    patches_for_variant.append(transform_patch_train)
                elif cleaning_method.patch_type == PatchType.ESTIMATOR_PATCH:
                    estimator_nodes = find_nodes_by_type(dag, OperatorType.ESTIMATOR)
                    if len(estimator_nodes) != 1:
                        raise Exception(
                            "Currently, DataCorruption only supports pipelines with exactly one estimator!")
                    estimator_node = estimator_nodes[0]
                    new_processing_func = partial(cleaning_method.fit_or_fit_transform_func,
                                                  make_classifier_func=estimator_node.make_classifier_func,
                                                  parallelism=self._parallelism)
                    new_description = f"{cleaning_method.method_name} patched {estimator_node.details.description}"
                    old_optimizer_info = estimator_node.details.optimizer_info
                    if cleaning_method.method_name == "cleanlab":
                        optimizer_mult_factor = (5 + 1)  # cleanlab default cross-val setting is *5, +1 is final retrain
                    elif cleaning_method.method_name == "shapley":
                        optimizer_mult_factor = (1 + 1)  # no cross-val currently, +1 is final retrain
                    else:
                        optimizer_mult_factor = 1
                    new_optimizer_info = OptimizerInfo(old_optimizer_info.runtime * optimizer_mult_factor,
                                                       old_optimizer_info.shape, old_optimizer_info.memory)
                    new_estimator_node = DagNode(singleton.get_next_op_id(),
                                                 estimator_node.code_location,
                                                 estimator_node.operator_info,
                                                 DagNodeDetails(new_description, estimator_node.details.columns,
                                                                new_optimizer_info),
                                                 estimator_node.optional_code_info,
                                                 new_processing_func)
                    model_patch = ModelPatch(singleton.get_next_patch_id(), self, True, new_estimator_node)
                    patches_for_variant.append(model_patch)
                else:
                    raise Exception(f"Unknown patch type: {cleaning_method.patch_type}!")
                cleaning_patch_sets.append(patches_for_variant)
        return cleaning_patch_sets

    def generate_final_report(self, extracted_plan_results: Dict[str, any]) -> any:
        # pylint: disable=too-many-locals
        result_df_columns = []
        result_df_errors = []
        result_df_cleaning_methods = []
        result_df_metrics = {}
        score_description_and_linenos = [(score_node.details.description, lineno)
                                         for (score_node, lineno) in self._score_nodes_and_linenos]

        result_df_columns.append(None)
        result_df_errors.append(None)
        result_df_cleaning_methods.append(None)
        for (score_description, lineno) in score_description_and_linenos:
            original_pipeline_result_label = f"original_L{lineno}"
            test_result_column_name = f"{score_description}_L{lineno}"
            test_column_values = result_df_metrics.get(test_result_column_name, [])
            test_column_values.append(singleton.labels_to_extracted_plan_results[original_pipeline_result_label])
            result_df_metrics[test_result_column_name] = test_column_values

        for (column, error_type) in self._columns_with_error:
            for cleaning_method in CLEANING_METHODS_FOR_ERROR_TYPE[error_type]:
                result_df_columns.append(column)
                result_df_errors.append(error_type.value)
                result_df_cleaning_methods.append(cleaning_method.method_name)
                for (score_description, lineno) in score_description_and_linenos:
                    cleaning_result_label = f"data-cleaning-{column}-{cleaning_method.method_name}_L{lineno}"
                    test_result_column_name = f"{score_description}_L{lineno}"
                    test_column_values = result_df_metrics.get(test_result_column_name, [])
                    test_column_values.append(singleton.labels_to_extracted_plan_results[cleaning_result_label])
                    result_df_metrics[test_result_column_name] = test_column_values
        result_df = pandas.DataFrame({'corrupted_column': result_df_columns,
                                      'error': result_df_errors,
                                      'cleaning_method': result_df_cleaning_methods,
                                      **result_df_metrics})
        return result_df
