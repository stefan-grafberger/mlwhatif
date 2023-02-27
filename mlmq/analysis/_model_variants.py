"""
The Model Variants What-If Analysis, which we use as an example to showcase common subexpression elimination
"""
from functools import partial
from typing import Iterable, Dict, Callable

import networkx
import pandas

from mlwhatif import OperatorType, DagNode, DagNodeDetails
from mlwhatif.analysis._analysis_utils import find_nodes_by_type
from mlwhatif.analysis._patch_creation import get_intermediate_extraction_patch_after_score_nodes
from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis
from mlwhatif.execution._patches import PipelinePatch, ModelPatch
from mlwhatif.execution._pipeline_executor import singleton


class ModelVariants(WhatIfAnalysis):
    """
    The Model Variants What-If Analysis
    """

    def __init__(self, named_model_variants: Iterable[tuple[str, Callable]]):
        # pylint: disable=unsubscriptable-object
        self.named_model_variants = list(named_model_variants)
        self._score_nodes_and_linenos = []
        self._analysis_id = (*self.named_model_variants,)

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
        for (model_description, model_function) in self.named_model_variants:
            patches_for_variant = []

            # Test set corruption
            test_corruption_result_label = f"model-variant-{model_description}"
            extraction_nodes = get_intermediate_extraction_patch_after_score_nodes(singleton, self,
                                                                                   test_corruption_result_label,
                                                                                   self._score_nodes_and_linenos)
            patches_for_variant.extend(extraction_nodes)

            model_patch = self._get_model_patch(dag, model_description, model_function)
            patches_for_variant.append(model_patch)

            corruption_patch_sets.append(patches_for_variant)

        return corruption_patch_sets

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

    def generate_final_report(self, extracted_plan_results: Dict[str, any]) -> any:
        # pylint: disable=too-many-locals
        result_df_model_variants = []
        result_df_metrics = {}
        score_description_and_linenos = [(score_node.details.description, lineno)
                                         for (score_node, lineno) in self._score_nodes_and_linenos]

        result_df_model_variants.append("original")
        for (score_description, lineno) in score_description_and_linenos:
            original_pipeline_result_label = f"original_L{lineno}"
            test_result_column_name = f"{score_description}_L{lineno}"
            test_column_values = result_df_metrics.get(test_result_column_name, [])
            metric_value = singleton.labels_to_extracted_plan_results[original_pipeline_result_label]
            test_column_values.append(metric_value)
            result_df_metrics[test_result_column_name] = test_column_values

        for (model_description, _) in self.named_model_variants:
            result_df_model_variants.append(model_description)

            for (score_description, lineno) in score_description_and_linenos:
                test_label = f"model-variant-{model_description}_L{lineno}"
                test_result_column_name = f"{score_description}_L{lineno}"
                test_column_values = result_df_metrics.get(test_result_column_name, [])
                test_column_values.append(singleton.labels_to_extracted_plan_results[test_label])
                result_df_metrics[test_result_column_name] = test_column_values

        result_df = pandas.DataFrame({'model_variant': result_df_model_variants,
                                      **result_df_metrics})
        return result_df

    @staticmethod
    def fit_model_variant(train_data, train_labels, make_classifier_func):
        """Create the classifier and fit it"""
        estimator = make_classifier_func()
        estimator.fit(train_data, train_labels)
        return estimator
