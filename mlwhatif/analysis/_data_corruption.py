"""
The Data Corruption What-If Analysis
"""
import functools
from types import FunctionType
from typing import Iterable, Dict

import networkx
import numpy
import pandas

from mlwhatif import OperatorType, DagNode, OperatorContext, DagNodeDetails
from mlwhatif.analysis._analysis_utils import add_intermediate_extraction_after_node, find_nodes_by_type, \
    find_first_op_modifying_a_column, add_new_node_between_nodes
from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis
from mlwhatif.instrumentation._pipeline_executor import singleton


class DataCorruption(WhatIfAnalysis):
    """
    The Data Corruption What-If Analysis
    """

    def __init__(self,
                 column_to_corruption: Dict[str, FunctionType],
                 corruption_percentages: Iterable[float] or None = None,
                 also_corrupt_train: bool = False):
        self.column_to_corruption = list(column_to_corruption.items())
        self.also_corrupt_train = also_corrupt_train
        if corruption_percentages is None:
            self.corruption_percentages = [0.2, 0.5, 0.9]
        else:
            self.corruption_percentages = corruption_percentages
        self._analysis_id = (*self.column_to_corruption, *self.corruption_percentages, self.also_corrupt_train)

    @property
    def analysis_id(self):
        return self._analysis_id

    def generate_plans_to_try(self, dag: networkx.DiGraph) \
            -> Iterable[networkx.DiGraph]:
        score_operators = find_nodes_by_type(dag, OperatorType.PREDICT)
        if len(score_operators) != 1:
            raise Exception("Currently, DataCorruption only supports pipelines with exactly one predict call which "
                            "must be on the test set!")
        final_result_value = score_operators[0]
        # TODO: Performance optimisation: deduplication for transformers that process multiple columns at once
        #  For project_modify, we can think about a similar deduplication: splitting operations on multiple columns and
        #  then using a concat in the end.

        corruption_dags = []
        for corruption_percentage in self.corruption_percentages:
            for (column, corruption_function) in self.column_to_corruption:
                train_corruption_location = self.find_dag_location_for_corruption(column, dag, True)
                test_corruption_location = self.find_dag_location_for_corruption(column, dag, False)

                # Test set corruption
                test_corruption_result_label = f"data-corruption-test-{column}-{corruption_percentage}"
                corruption_dag = dag.copy()
                add_intermediate_extraction_after_node(corruption_dag, final_result_value, test_corruption_result_label)

                self.add_corruption_in_location(column, corruption_dag, corruption_function, train_corruption_location,
                                                corruption_percentage)
                corruption_dags.append(corruption_dag)

                # Train and test set corruption
                if self.also_corrupt_train is True:
                    test_corruption_result_label = f"data-corruption-train-{column}-{corruption_percentage}"
                    corruption_dag = dag.copy()
                    add_intermediate_extraction_after_node(corruption_dag, final_result_value,
                                                           test_corruption_result_label)
                    self.add_corruption_in_location(column, corruption_dag, corruption_function,
                                                    train_corruption_location, corruption_percentage)
                    self.add_corruption_in_location(column, corruption_dag, corruption_function,
                                                    test_corruption_location, corruption_percentage)
                    corruption_dags.append(corruption_dag)

        return corruption_dags

    def add_corruption_in_location(self, column, corruption_dag, corruption_function, corruption_location,
                                   corruption_percentage):
        # pylint: disable=too-many-arguments
        """We now know where to apply the corruption, and use this function to actually apply it."""
        new_corruption_node = self.create_corruption_node(column, corruption_function,
                                                          corruption_percentage,
                                                          corruption_location)
        add_new_node_between_nodes(corruption_dag, new_corruption_node, corruption_location)

    @staticmethod
    def find_dag_location_for_corruption(column, dag, test_not_train):
        """Find out between which two nodes to apply the corruption"""
        search_start_node = DataCorruption.find_train_or_test_pipeline_part_end(dag, test_not_train)
        first_op_requiring_corruption = find_first_op_modifying_a_column(dag, search_start_node, column, test_not_train)
        operator_parent_nodes = DataCorruption.get_sorted_parent_nodes(dag, first_op_requiring_corruption)
        first_op_requiring_corruption, operator_to_apply_corruption_after = DataCorruption\
            .find_where_to_apply_corruption_exactly(dag, first_op_requiring_corruption, operator_parent_nodes)
        return operator_to_apply_corruption_after, first_op_requiring_corruption

    @staticmethod
    def find_train_or_test_pipeline_part_end(dag, test_not_train):
        """We want to start at the end of the pipeline to find the relevant train or test operations"""
        if test_not_train is True:
            search_start_nodes = find_nodes_by_type(dag, OperatorType.PREDICT)
            if len(search_start_nodes) != 1:
                raise Exception("Currently, DataCorruption only supports pipelines with exactly one predict call "
                                "for the test set!")

            search_start_node = search_start_nodes[0]
        else:
            search_start_nodes = find_nodes_by_type(dag, OperatorType.ESTIMATOR)
            if len(search_start_nodes) != 1:
                raise Exception("Currently, DataCorruption only supports pipelines with exactly one estimator!")
            search_start_node = search_start_nodes[0]
        return search_start_node

    @staticmethod
    def get_sorted_parent_nodes(dag: networkx.DiGraph, first_op_requiring_corruption):
        """Get the parent nodes of a node sorted by arg_index"""
        operator_parent_nodes = list(dag.predecessors(first_op_requiring_corruption))
        parent_nodes_with_arg_index = [(parent_node, dag.get_edge_data(parent_node, first_op_requiring_corruption))
                                       for parent_node in operator_parent_nodes]
        parent_nodes_with_arg_index = sorted(parent_nodes_with_arg_index, key=lambda x: x[1]['arg_index'])
        operator_parent_nodes = [node for (node, _) in parent_nodes_with_arg_index]
        return operator_parent_nodes

    @staticmethod
    def get_sorted_children_nodes(dag: networkx.DiGraph, first_op_requiring_corruption):
        """Get the parent nodes of a node sorted by arg_index"""
        operator_child_nodes = list(dag.successors(first_op_requiring_corruption))
        sorted_operator_child_nodes = sorted(operator_child_nodes, key=lambda x: x.node_id)
        return sorted_operator_child_nodes

    @staticmethod
    def find_where_to_apply_corruption_exactly(dag, first_op_requiring_corruption, operator_parent_nodes):
        """
        We know which operator requires the corruption to be present already; now we need to decide between which
        parent node and the current node we need to insert the corruption node.
        """
        if first_op_requiring_corruption.operator_info.operator == OperatorType.TRANSFORMER:
            operator_to_apply_corruption_after = operator_parent_nodes[-1]
        elif first_op_requiring_corruption.operator_info.operator == OperatorType.PREDICT:
            operator_to_apply_corruption_after = operator_parent_nodes[1]
        elif first_op_requiring_corruption.operator_info.operator == OperatorType.ESTIMATOR:
            operator_to_apply_corruption_after = operator_parent_nodes[0]
        elif first_op_requiring_corruption.operator_info.operator == OperatorType.PROJECTION_MODIFY:
            project_modify_parent_a = operator_parent_nodes[0]
            project_modify_parent_b = operator_parent_nodes[-1]
            # We want to introduce the change before all subscript behavior
            operator_to_apply_corruption_after = networkx.lowest_common_ancestor(dag, project_modify_parent_a,
                                                                                 project_modify_parent_b)
            sorted_successors = DataCorruption.get_sorted_children_nodes(dag, operator_to_apply_corruption_after)
            first_op_requiring_corruption = sorted_successors[0]
        else:
            raise Exception("Either a column was changed by a transformer or project_modify or we can apply"
                            "the corruption right before the estimator operation!")
        return first_op_requiring_corruption, operator_to_apply_corruption_after

    @staticmethod
    def create_corruption_node(column, corruption_function, corruption_percentage, corruption_location):
        """Create the node that applies the specified corruption"""
        operator_to_apply_corruption_after, first_op_requiring_corruption = corruption_location

        def corrupt_df(pandas_df, corruption_percentage, corruption_function, column):
            # TODO: If we model this as 3 operations instead of one, optimization should be easy
            # TODO: Think about when we actually want to be defensive and call copy and when not
            # TODO: Think about datatypes. corruption_function currently assumes pandas DataFrames.
            #  We may want to automatically convert data formats as needed, here and in other places.
            completely_corrupted_df = pandas_df.copy()
            completely_corrupted_df = corruption_function(completely_corrupted_df)
            corrupt_count = int(len(pandas_df) * corruption_percentage)
            indexes_to_corrupt = numpy.random.permutation(pandas_df.index)[:corrupt_count]
            return_df = pandas_df.copy()
            return_df.loc[indexes_to_corrupt, column] = completely_corrupted_df.loc[indexes_to_corrupt, column]
            return return_df

        # We need to use partial here to avoid problems with late bindings, see
        #  https://stackoverflow.com/questions/3431676/creating-functions-in-a-loop
        corrupt_df_with_proper_bindings = functools.partial(corrupt_df,
                                                            corruption_percentage=corruption_percentage,
                                                            corruption_function=corruption_function,
                                                            column=column)
        new_corruption_node = DagNode(singleton.get_next_op_id(),
                                      first_op_requiring_corruption.code_location,
                                      OperatorContext(OperatorType.PROJECTION_MODIFY, None),
                                      DagNodeDetails(f"Corrupt {corruption_percentage * 100}% of '{column}'",
                                                     operator_to_apply_corruption_after.details.columns),
                                      None,
                                      corrupt_df_with_proper_bindings)
        return new_corruption_node

    def generate_final_report(self, extracted_plan_results: Dict[str, any]) -> any:
        result_df_columns = []
        result_df_percentages = []
        result_df_metrics_corrupt_test_only = []
        result_df_metrics_corrupt_train_and_test = []
        for (column, _) in self.column_to_corruption:
            for corruption_percentage in self.corruption_percentages:
                test_label = f"data-corruption-test-{column}-{corruption_percentage}"
                result_df_columns.append(column)
                result_df_percentages.append(corruption_percentage)
                result_df_metrics_corrupt_test_only.append(singleton.labels_to_extracted_plan_results[test_label])

                train_label = f"data-corruption-train-{column}-{corruption_percentage}"
                if train_label in singleton.labels_to_extracted_plan_results:
                    train_and_test_metric = singleton.labels_to_extracted_plan_results[train_label]
                    result_df_metrics_corrupt_train_and_test.append(train_and_test_metric)
        if self.also_corrupt_train is True:
            result_df = pandas.DataFrame({'column': result_df_columns,
                                          'corruption_percentage': result_df_percentages,
                                          'metric_corrupt_test_only': result_df_metrics_corrupt_test_only,
                                          'metric_corrupt_train_and_test:': result_df_metrics_corrupt_train_and_test})
        else:
            result_df = pandas.DataFrame({'column': result_df_columns,
                                          'corruption_percentage': result_df_percentages,
                                          'metric_corrupt_test_only': result_df_metrics_corrupt_test_only})
        return result_df
