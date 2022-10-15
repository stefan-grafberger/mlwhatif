"""
The Operator Impact What-If Analysis
"""
from collections import defaultdict
from functools import partial
from typing import Iterable, Dict, List

import networkx
import pandas
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from mlmq import OperatorType, DagNode, OperatorContext, DagNodeDetails
from mlmq.analysis._analysis_utils import find_nodes_by_type, find_train_or_test_pipeline_part_end, \
    filter_estimator_transformer_edges
from mlmq.analysis._patch_creation import get_intermediate_extraction_patch_after_score_nodes
from mlmq.analysis._what_if_analysis import WhatIfAnalysis
from mlmq.execution._patches import PipelinePatch, OperatorReplacement, OperatorRemoval
from mlmq.execution._pipeline_executor import singleton
from mlmq.monkeypatching._monkey_patching_utils import wrap_in_mlinspect_array_if_necessary


class OperatorImpact(WhatIfAnalysis):
    """
    The Operator Impact What-If Analysis
    """

    def __init__(self, test_transformers=True, test_selections=False, restrict_to_linenos: List[int] or None = None):
        self._test_transformers = test_transformers
        self._test_selections = test_selections
        self._restrict_to_linenos = restrict_to_linenos
        self._analysis_id = (test_transformers, test_selections)
        self._transformer_operators_to_test = []
        self._filter_operators_to_test = []
        self._score_nodes_and_linenos = []

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
        self._transformer_operators_to_test = self._get_transformer_operators_to_test(dag)
        self._filter_operators_to_test = self._get_filter_operators_to_test(dag)
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
        for (operator_to_replace_train, operator_to_replace_test, before_split) in self._filter_operators_to_test:
            patches_for_variant = []
            if operator_to_replace_train is not None:
                main_operator_to_remove = operator_to_replace_train
                maybe_secondary_operator = operator_to_replace_test
            else:
                main_operator_to_remove = operator_to_replace_test
                maybe_secondary_operator = None
            result_label = self.get_label_for_operator(main_operator_to_remove)
            extraction_nodes = get_intermediate_extraction_patch_after_score_nodes(singleton, self,
                                                                                   result_label,
                                                                                   self._score_nodes_and_linenos)
            patches_for_variant.extend(extraction_nodes)

            assert main_operator_to_remove.operator_info.operator == OperatorType.SELECTION
            removal_patch = OperatorRemoval(singleton.get_next_patch_id(), self, True,
                                            main_operator_to_remove, maybe_secondary_operator, before_split)
            patches_for_variant.append(removal_patch)
            fairness_patch_sets.append(patches_for_variant)
        return fairness_patch_sets

    @staticmethod
    def get_transformer_replacement_node(operator_to_replace):
        """Replace a transformer with an alternative that does nothing."""

        def onehot_transformer_processing_func(input_df):
            input_df_copy = input_df.copy()
            transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')
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

        def imputer_transformer_processing_func(input_df):
            # TODO: Is converting numpy to pandas df necessary or not?
            # if not isinstance(input_df, pandas.DataFrame):
            #     input_df = pandas.DataFrame.from_records(input_df)
            input_df_copy = input_df.copy()
            stringify_and_impute = Pipeline([
                ('stringify_bool',
                 FunctionTransformer(func=lambda maybe_bool: maybe_bool.astype(str), check_inverse=False)),
                ('impute_replacement', SimpleImputer(strategy='constant'))])

            transformer = ColumnTransformer(
                transformers=[
                    ("num", SimpleImputer(strategy='constant'), make_column_selector(dtype_include="number")),
                    ("non_num", stringify_and_impute, make_column_selector(dtype_exclude="number"))
                ]
            )
            transformed_data = transformer.fit_transform(input_df_copy)
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        # TODO: When else do we need to use a OneHotEncoder?
        if "Word2Vec" in operator_to_replace.details.description:
            replacement_func = onehot_transformer_processing_func
            replacement_desc = "One-hot encoding"
        elif "Imputer" in operator_to_replace.details.description or \
                (operator_to_replace.optional_code_info and
                 operator_to_replace.optional_code_info.source_code and
                 "imputer" in operator_to_replace.optional_code_info.source_code):
            # TODO: This is a hacky workaround for our end_to_end experiments. Normally the restrict_lineno arg
            #  is for operations that operator_impact cannot handle, but for the end_to_end experiments
            #  hard-coding the linenos is not a good idea
            replacement_func = imputer_transformer_processing_func
            replacement_desc = "Replace nan with constant"
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
            if "Word2Vec" in operator_to_replace.details.description:
                replacement_description = "one-hot encode instead"
            elif "Imputer" in operator_to_replace.details.description or \
                    (operator_to_replace.optional_code_info and
                     operator_to_replace.optional_code_info.source_code and
                     "imputer" in operator_to_replace.optional_code_info.source_code):
                # TODO: This is a hacky workaround for our end_to_end experiments. Normally the restrict_lineno arg
                #  is for operations that operator_impact cannot handle, but for the end_to_end experiments
                #  hard-coding the linenos is not a good idea
                replacement_description = "replace nan with constant"
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
        if self._test_transformers is True:
            transformers_to_replace = [node for node in nodes_to_search if
                                       node.operator_info.operator == OperatorType.TRANSFORMER
                                       and ": fit_transform" in node.details.description
                                       and "One-Hot" not in node.details.description
                                       and "Hashing Vectorizer" not in node.details.description
                                       and "Count Vectorizer" not in node.details.description
                                       and "Word2Vec" not in node.details.description]
            # TODO: We exclude text operators for now. It is a bit unclear what a fair default for text would be.
            all_nodes_to_test.extend(transformers_to_replace)
        if self._restrict_to_linenos is not None:
            lineno_set = set(self._restrict_to_linenos)
            all_nodes_to_test = set(node for node in all_nodes_to_test if node.code_location.lineno in lineno_set)
        return all_nodes_to_test

    def _get_filter_operators_to_test(self, dag):
        """
        For now, we will ignore project modifies and focus on selections and transformers.
        This is because for transformers it is easy to find the corresponding test set operation and for the
        selection we do not need to worry about finding corresponding test set operations.
        """
        search_start_node_train = find_train_or_test_pipeline_part_end(dag, True)
        search_start_node_test = find_train_or_test_pipeline_part_end(dag, False)
        nodes_to_search_train = set(networkx.ancestors(dag, search_start_node_train))
        nodes_to_search_test = set(networkx.ancestors(dag, search_start_node_test))
        nodes_to_search = set(nodes_to_search_train).union(nodes_to_search_test)

        # Filter out filters that are the train test split
        dag_to_consider = networkx.subgraph_view(dag, filter_edge=filter_estimator_transformer_edges)
        last_op_before_train_test_split = networkx.lowest_common_ancestor(dag_to_consider, search_start_node_train,
                                                                          search_start_node_test)

        all_nodes_to_test = []
        if self._test_selections is True:
            selections_to_replace = [node for node in nodes_to_search if
                                     node.operator_info.operator == OperatorType.SELECTION
                                     and dag.has_edge(last_op_before_train_test_split, node) is False
                                     and node.details.description != "dropna"]
            all_nodes_to_test.extend(selections_to_replace)
            # TODO: We need to process filters that are present on both test and train side at the same time
        if self._restrict_to_linenos is not None:
            lineno_set = set(self._restrict_to_linenos)
            all_nodes_to_test = set(node for node in all_nodes_to_test if node.code_location.lineno in lineno_set)

        return_value = self._detect_if_filters_shared_between_train_test_or_duplicated(all_nodes_to_test, dag)
        return return_value

    @staticmethod
    def _detect_if_filters_shared_between_train_test_or_duplicated(all_nodes_to_test, dag):
        """Detect if a filter is on the train or test side only, if it is shared between the two sides, or if there are
        similar filters on both sides."""
        filter_pairs_to_consider = defaultdict(partial(list, [None, None, False]))
        dag_to_consider = networkx.subgraph_view(dag, filter_edge=filter_estimator_transformer_edges)
        train_search_start_node = find_train_or_test_pipeline_part_end(dag_to_consider, True)
        train_nodes = set(networkx.ancestors(dag, train_search_start_node))
        test_search_start_node = find_train_or_test_pipeline_part_end(dag_to_consider, False)
        test_nodes = set(networkx.ancestors(dag_to_consider, test_search_start_node))
        for node in all_nodes_to_test:
            filter_key = (node.details.description, tuple(node.details.columns))
            if node in train_nodes and node in test_nodes:
                filter_pairs_to_consider[filter_key][0] = node
                filter_pairs_to_consider[filter_key][1] = None
                filter_pairs_to_consider[filter_key][2] = True
            elif node in train_nodes:
                filter_pairs_to_consider[filter_key][0] = node
            elif node in test_nodes:
                filter_pairs_to_consider[filter_key][1] = node
            else:
                raise Exception("A node must be either in the train or test part of the pipeline")
        return_value = list(filter_pairs_to_consider.values())
        return return_value
