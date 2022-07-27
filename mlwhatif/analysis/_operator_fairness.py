"""
The Operator Fairness What-If Analysis
"""
from typing import Iterable, Dict, List

import networkx
import pandas
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from mlwhatif import OperatorType, DagNode, OperatorContext, DagNodeDetails
from mlwhatif.analysis._analysis_utils import find_nodes_by_type, add_intermediate_extraction_after_node, replace_node, \
    get_sorted_parent_nodes
from mlwhatif.analysis._data_corruption import DataCorruption
from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis
from mlwhatif.instrumentation._pipeline_executor import singleton
from mlwhatif.monkeypatching._monkey_patching_utils import wrap_in_mlinspect_array_if_necessary


class OperatorFairness(WhatIfAnalysis):
    """
    The Operator Fairness What-If Analysis
    """

    def __init__(self, test_transformers=True, test_selections=False, restrict_to_linenos: List[int] or None = None):
        self._test_transformers = test_transformers
        self._test_selections = test_selections
        self._restrict_to_linenos = restrict_to_linenos
        self._analysis_id = (test_transformers, test_selections)
        self._operators_to_test = []
        self._score_nodes_and_linenos = []

    @property
    def analysis_id(self):
        return self._analysis_id

    def generate_plans_to_try(self, dag: networkx.DiGraph) \
            -> Iterable[networkx.DiGraph]:
        predict_operators = find_nodes_by_type(dag, OperatorType.PREDICT)
        if len(predict_operators) != 1:
            raise Exception("Currently, DataCorruption only supports pipelines with exactly one predict call which "
                            "must be on the test set!")
        score_operators = find_nodes_by_type(dag, OperatorType.SCORE)
        self._score_nodes_and_linenos = [(node, node.code_location.lineno) for node in score_operators]
        if len(self._score_nodes_and_linenos) != len(set(self._score_nodes_and_linenos)):
            raise Exception("Currently, DataCorruption only supports pipelines where different score operations can "
                            "be uniquely identified by the line number in the code!")
        self._operators_to_test = self.get_operators_to_test(dag)
        result_dags = []
        for operator_to_replace in self._operators_to_test:
            modified_dag = dag.copy()
            result_label = self.get_label_for_operator(operator_to_replace)
            self.add_intermediate_extraction_after_score_nodes(modified_dag, result_label)
            # TODO: Replace operator, two operator types exist here currently: transformers and selections

            # This is the passthrough transformer the sklearn ColumnTransformer uses internally, a FunctionTransformer
            #  that uses the identity function
            if operator_to_replace.operator_info.operator == OperatorType.TRANSFORMER:
                replacement_node = self.get_transformer_replacement_node(operator_to_replace)
                replace_node(modified_dag, operator_to_replace, replacement_node)
            elif operator_to_replace.operator_info.operator == OperatorType.SELECTION:
                self.remove_nodes_associated_with_selection(modified_dag, operator_to_replace)
            else:
                raise Exception(f"Replacing operator type {operator_to_replace.operator_info.operator.value} is "
                                f"not supported yet!")

            result_dags.append(modified_dag)
        return result_dags

    @staticmethod
    def remove_nodes_associated_with_selection(modified_dag, operator_to_replace):
        """Removes the selection node as well as nodes that are there for the selection condition"""
        # TODO: This code assumes a selection based on a simple condition like in our example pipelines for now.
        operator_parent_nodes = get_sorted_parent_nodes(modified_dag, operator_to_replace)
        selection_parent_a = operator_parent_nodes[0]
        selection_parent_b = operator_parent_nodes[-1]
        # We want to introduce the change before all subscript behavior
        operator_after_which_cutoff_required = networkx.lowest_common_ancestor(modified_dag, selection_parent_a,
                                                                               selection_parent_b)
        paths_between_generator = networkx.all_simple_paths(modified_dag,
                                                            source=operator_after_which_cutoff_required,
                                                            target=operator_to_replace)
        nodes_between_set = {node for path in paths_between_generator for node in path}
        nodes_between_set.remove(operator_after_which_cutoff_required)
        children_before_modifications = list(modified_dag.successors(operator_to_replace))
        for child_node in children_before_modifications:
            edge_data = modified_dag.get_edge_data(operator_to_replace, child_node)
            modified_dag.add_edge(operator_after_which_cutoff_required, child_node, **edge_data)
        modified_dag.remove_node(operator_to_replace)
        modified_dag.remove_nodes_from(nodes_between_set)

    @staticmethod
    def get_transformer_replacement_node(operator_to_replace):
        """Replace a transformer with an alternative that does nothing."""
        def onehot_transformer_processing_func(input_df):
            transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')
            transformed_data = transformer.fit_transform(input_df)
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        def passthrough_transformer_processing_func(input_df):
            transformer = FunctionTransformer(accept_sparse=True, check_inverse=False)
            transformed_data = transformer.fit_transform(input_df)
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        def imputer_transformer_processing_func(input_df):
            # TODO: Is converting numpy to pandas df necessary or not?
            # if not isinstance(input_df, pandas.DataFrame):
            #     input_df = pandas.DataFrame.from_records(input_df)
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
            transformed_data = transformer.fit_transform(input_df)
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        # TODO: When else do we need to use a OneHotEncoder?
        if "Word2Vec" in operator_to_replace.details.description:
            replacement_func = onehot_transformer_processing_func
        elif "Imputer" in operator_to_replace.details.description:
            replacement_func = imputer_transformer_processing_func
        else:
            replacement_func = passthrough_transformer_processing_func
        replacement_node = DagNode(singleton.get_next_op_id(),
                                   operator_to_replace.code_location,
                                   OperatorContext(OperatorType.TRANSFORMER, None),
                                   DagNodeDetails(f"Do nothing",
                                                  operator_to_replace.details.columns),
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
        result_df_metrics = {}
        score_linenos = [lineno for (_, lineno) in self._score_nodes_and_linenos]
        for operator_to_replace in self._operators_to_test:
            result_df_op_type.append(operator_to_replace.operator_info.operator.value)
            result_df_lineno.append(operator_to_replace.code_location.lineno)
            result_df_op_code.append(operator_to_replace.optional_code_info.source_code)
            label_for_operator = self.get_label_for_operator(operator_to_replace)
            for lineno in score_linenos:
                test_label = f"{label_for_operator}_L{lineno}"
                test_result_column_name = f"score_L{lineno}"
                test_column_values = result_df_metrics.get(test_result_column_name, [])
                test_column_values.append(singleton.labels_to_extracted_plan_results[test_label])
                result_df_metrics[test_result_column_name] = test_column_values
        result_df = pandas.DataFrame({'operator_type': result_df_op_type,
                                      'operator_lineno': result_df_lineno,
                                      'operator_code': result_df_op_code,
                                      **result_df_metrics})
        return result_df

    def add_intermediate_extraction_after_score_nodes(self, dag: networkx.DiGraph, label: str):
        """Add a new node behind some given node to extract the intermediate result of that given node"""
        node_linenos = []
        for node, lineno in self._score_nodes_and_linenos:
            node_linenos.append(lineno)
            node_label = f"{label}_L{lineno}"
            add_intermediate_extraction_after_node(dag, node, node_label)
        return node_linenos

    def get_operators_to_test(self, dag):
        """
        For now, we will ignore project modifies and focus on selections and transformers.
        This is because for transformers it is easy to find the corresponding test set operation and for the
        selection we do not need to worry about finding corresponding test set operations.
        """
        search_start_node = DataCorruption.find_train_or_test_pipeline_part_end(dag, False)
        nodes_to_search = set(networkx.ancestors(dag, search_start_node))
        all_nodes_to_test = []
        if self._test_transformers is True:
            transformers_to_replace = [node for node in nodes_to_search if
                                       node.operator_info.operator == OperatorType.TRANSFORMER
                                       and ": fit_transform" in node.details.description
                                       and "One-Hot" not in node.details.description]
            all_nodes_to_test.extend(transformers_to_replace)
        if self._test_selections is True:
            selections_to_replace = [node for node in nodes_to_search if
                                     node.operator_info.operator == OperatorType.SELECTION]
            all_nodes_to_test.extend(selections_to_replace)
        if self._restrict_to_linenos is not None:
            lineno_set = set(self._restrict_to_linenos)
            all_nodes_to_test = set(node for node in all_nodes_to_test if node.code_location.lineno in lineno_set)
        return all_nodes_to_test
