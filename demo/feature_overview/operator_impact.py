"""
The Operator Impact What-If Analysis
"""
# pylint: disable-all
from functools import partial
from typing import Iterable, Dict

import networkx
import pandas
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from mlmq import OperatorType, DagNode, OperatorContext, DagNodeDetails
from mlmq.analysis._analysis_utils import find_nodes_by_type, find_train_or_test_pipeline_part_end
from mlmq.analysis._patch_creation import get_intermediate_extraction_patch_after_score_nodes
from mlmq.analysis._what_if_analysis import WhatIfAnalysis
from mlmq.execution._patches import PipelinePatch, OperatorReplacement, ModelPatch
from mlmq.execution._pipeline_executor import singleton
from mlmq.monkeypatching._monkey_patching_utils import wrap_in_mlinspect_array_if_necessary


class OperatorImpact(WhatIfAnalysis):
    """
    The Operator Impact What-If Analysis
    """

    def __init__(self, robust_scaling=True, named_model_variants=None):
        if named_model_variants is None:
            named_model_variants = []
        self.robust_scaling = robust_scaling
        self.named_model_variants = named_model_variants
        self._analysis_id = (robust_scaling, *named_model_variants)
        self._transformer_operators_to_test = []
        self._filter_operators_to_test = []
        self._score_nodes_and_linenos = []
        self.model = None

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
        self.model = find_nodes_by_type(dag, OperatorType.ESTIMATOR)[0]
        self._score_nodes_and_linenos = [(node, node.code_location.lineno) for node in score_operators]
        if len(self._score_nodes_and_linenos) != len(set(self._score_nodes_and_linenos)):
            raise Exception("Currently, DataCorruption only supports pipelines where different score operations can "
                            "be uniquely identified by the line number in the code!")
        self._transformer_operators_to_test = self._get_transformer_operators_to_test(dag)
        fairness_patch_sets = []
        for operator_to_replace in self._transformer_operators_to_test:
            patches_for_variant = []
            result_label = self.get_label_for_operator(operator_to_replace)
            extraction_nodes = get_intermediate_extraction_patch_after_score_nodes(singleton, self,
                                                                                   result_label,
                                                                                   self._score_nodes_and_linenos)
            patches_for_variant.extend(extraction_nodes)
            assert operator_to_replace.operator_info.operator == OperatorType.TRANSFORMER
            replacement_node = self.get_transformer_replacement_node(operator_to_replace)
            patch = OperatorReplacement(singleton.get_next_patch_id(), self, True, operator_to_replace,
                                        replacement_node)
            patches_for_variant.append(patch)
            fairness_patch_sets.append(patches_for_variant)
        for (model_description, model_function) in self.named_model_variants:
            patches_for_variant = []
            test_corruption_result_label = f"model-variant-{model_description}"
            extraction_nodes = get_intermediate_extraction_patch_after_score_nodes(singleton, self,
                                                                                   test_corruption_result_label,
                                                                                   self._score_nodes_and_linenos)
            patches_for_variant.extend(extraction_nodes)

            model_patch = self._get_model_patch(dag, model_description, model_function)
            patches_for_variant.append(model_patch)

            fairness_patch_sets.append(patches_for_variant)
        return fairness_patch_sets

    @staticmethod
    def get_transformer_replacement_node(operator_to_replace):
        """Replace a transformer with an alternative that does nothing."""

        def robust_scaler_processing_func(input_df):
            input_df_copy = input_df.copy()
            transformer = RobustScaler()
            transformed_data = transformer.fit_transform(input_df_copy)
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        def passthrough_transformer_processing_func(input_df):
            # This is the passthrough transformer the sklearn ColumnTransformer uses internally, a FunctionTransformer
            #  that uses the identity function
            input_df_copy = input_df.copy()
            transformer = FunctionTransformer(accept_sparse=True, check_inverse=False)
            transformed_data = transformer.fit_transform(input_df_copy)
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        if "Standard" in operator_to_replace.details.description:
            replacement_func = robust_scaler_processing_func
            replacement_desc = "Patched Robust Scaler: fit_transform"
        else:
            replacement_func = passthrough_transformer_processing_func
            replacement_desc = "Do nothing"
        replacement_node = DagNode(singleton.get_next_op_id(),
                                   operator_to_replace.code_location,
                                   OperatorContext(OperatorType.TRANSFORMER, None),
                                   DagNodeDetails(replacement_desc, operator_to_replace.details.columns),
                                   None,
                                   replacement_func)
        return replacement_node

    @staticmethod
    def get_label_for_operator(operator_to_replace):
        """Generate the label under which results for this operator get saved"""
        result_label = f"operator-fairness-{operator_to_replace.operator_info.operator.value}" \
                       f"-OP_L{operator_to_replace.code_location.lineno}"
        return result_label

    def generate_final_report(self, extracted_plan_results: Dict[str, any]) -> any:
        # pylint: disable=too-many-locals
        result_df_op_type = []
        result_df_lineno = []
        result_df_op_code = []
        result_df_replacement_description = []
        result_df_metrics = {}
        score_description_and_linenos = [(score_node.details.description, lineno)
                                         for (score_node, lineno) in self._score_nodes_and_linenos]

        result_df_op_type.append(None)
        result_df_lineno.append(None)
        result_df_op_code.append(None)
        result_df_replacement_description.append(None)
        for (score_description, lineno) in score_description_and_linenos:
            original_pipeline_result_label = f"original_L{lineno}"
            test_result_column_name = f"{score_description}_L{lineno}"
            test_column_values = result_df_metrics.get(test_result_column_name, [])
            test_column_values.append(singleton.labels_to_extracted_plan_results[original_pipeline_result_label])
            result_df_metrics[test_result_column_name] = test_column_values

        # TODO: Maybe the strategy_description should also contain if we found and dropped a corresponding test filter
        main_filter_ops = [train_op or test_op for (train_op, test_op, _) in self._filter_operators_to_test]
        for operator_to_replace in [*self._transformer_operators_to_test, *main_filter_ops]:
            result_df_op_type.append(operator_to_replace.operator_info.operator.value)
            result_df_lineno.append(operator_to_replace.code_location.lineno)
            result_df_op_code.append(operator_to_replace.optional_code_info.source_code)
            replacement_description = self.get_op_replacement_description(operator_to_replace)
            result_df_replacement_description.append(replacement_description)
            label_for_operator = self.get_label_for_operator(operator_to_replace)
            for (score_description, lineno) in score_description_and_linenos:
                test_label = f"{label_for_operator}_L{lineno}"
                test_result_column_name = f"{score_description}_L{lineno}"
                test_column_values = result_df_metrics.get(test_result_column_name, [])
                test_column_values.append(singleton.labels_to_extracted_plan_results[test_label])
                result_df_metrics[test_result_column_name] = test_column_values
        for (model_description, _) in self.named_model_variants:
            result_df_op_type.append(self.model.operator_info.operator.value)
            result_df_lineno.append(self.model.code_location.lineno)
            result_df_op_code.append(self.model.optional_code_info.source_code)
            result_df_replacement_description.append(model_description + " instead")

            for (score_description, lineno) in score_description_and_linenos:
                test_label = f"model-variant-{model_description}_L{lineno}"
                test_result_column_name = f"{score_description}_L{lineno}"
                test_column_values = result_df_metrics.get(test_result_column_name, [])
                test_column_values.append(singleton.labels_to_extracted_plan_results[test_label])
                result_df_metrics[test_result_column_name] = test_column_values
        result_df = pandas.DataFrame({'operator_type': result_df_op_type,
                                      'operator_lineno': result_df_lineno,
                                      'operator_code': result_df_op_code,
                                      'strategy_description': result_df_replacement_description,
                                      **result_df_metrics})
        return result_df

    @staticmethod
    def get_op_replacement_description(operator_to_replace):
        """Generate a description explaining the baseline the operator is being compared to"""
        if operator_to_replace.operator_info.operator == OperatorType.TRANSFORMER:
            if "Standard" in operator_to_replace.details.description:
                replacement_description = "robust scale instead"
            else:
                replacement_description = "passthrough"
        elif operator_to_replace.operator_info.operator == OperatorType.SELECTION:
            replacement_description = "drop the filter"
        else:
            raise Exception(f"Replacing operator type {operator_to_replace.operator_info.operator.value} is "
                            f"not supported yet!")
        return replacement_description

    def _get_transformer_operators_to_test(self, dag):
        """
        For now, we will ignore project modifies and focus on selections and transformers.
        This is because for transformers it is easy to find the corresponding test set operation and for the
        selection we do not need to worry about finding corresponding test set operations.
        """
        search_start_node = find_train_or_test_pipeline_part_end(dag, True)
        nodes_to_search = set(networkx.ancestors(dag, search_start_node))
        all_nodes_to_test = []
        if self.robust_scaling is True:
            transformers_to_replace = [node for node in nodes_to_search if
                                       node.operator_info.operator == OperatorType.TRANSFORMER
                                       and ": fit_transform" in node.details.description
                                       and "Standard" in node.details.description]
            # TODO: We exclude text operators for now. It is a bit unclear what a fair default for text would be.
            all_nodes_to_test.extend(transformers_to_replace)
        return all_nodes_to_test

    def _get_model_patch(self, dag, model_description, model_function):
        estimator_nodes = find_nodes_by_type(dag, OperatorType.ESTIMATOR)
        if len(estimator_nodes) != 1:
            raise Exception(
                "Currently, DataCorruption only supports pipelines with exactly one estimator!")
        estimator_node = estimator_nodes[0]
        new_processing_func = partial(self.fit_model_variant, make_classifier_func=model_function)
        new_description = f"Model Variant: {model_description}"
        new_estimator_node = DagNode(singleton.get_next_op_id(),
                                     estimator_node.code_location,
                                     estimator_node.operator_info,
                                     DagNodeDetails(new_description, estimator_node.details.columns,
                                                    estimator_node.details.optimizer_info),
                                     estimator_node.optional_code_info,
                                     new_processing_func)
        model_patch = ModelPatch(singleton.get_next_patch_id(), self, True, new_estimator_node)
        return model_patch

    @staticmethod
    def fit_model_variant(train_data, train_labels, make_classifier_func):
        """Create the classifier and fit it"""
        estimator = make_classifier_func()
        estimator.fit(train_data, train_labels)
        return estimator
