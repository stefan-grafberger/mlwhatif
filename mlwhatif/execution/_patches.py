""" Contains all the different patch-related classes """
import dataclasses
import logging
from abc import ABC, abstractmethod
from typing import List, Callable, Iterable

import networkx

from mlwhatif.instrumentation._operator_types import OperatorType
from mlwhatif.analysis._analysis_utils import find_dag_location_for_data_patch, add_new_node_after_node, \
    find_nodes_by_type, replace_node, remove_node
from mlwhatif.instrumentation._dag_node import DagNode


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Patch(ABC):
    """ Basic Patch class """
    patch_id: int
    analysis: any  # WhatIfAnalyis but this would be a circular import currently
    changes_following_results: bool

    @abstractmethod
    def apply(self, dag: networkx.DiGraph):
        """Apply the patch to some DAG"""
        raise NotImplementedError

    @abstractmethod
    def get_nodes_needing_recomputation(self, old_dag: networkx.DiGraph, new_dag: networkx.DiGraph):
        """Get all nodes that need to be recomputed after patching the DAG"""
        raise NotImplementedError

    def _get_nodes_needing_recomputation(self,
                                         old_dag: networkx.DiGraph,
                                         new_dag: networkx.DiGraph,
                                         removed_nodes: Iterable[DagNode],
                                         added_nodes: Iterable[DagNode]):
        """Get all nodes that need to be recomputed after patching the DAG"""
        all_nodes_needing_recomputation = set()
        if self.changes_following_results is True:
            for added_node in added_nodes:
                local_nodes_needing_recomputation = set(networkx.descendants(new_dag, added_node))
                all_nodes_needing_recomputation.update(local_nodes_needing_recomputation)
            for removed_node in removed_nodes:
                original_nodes_needing_recomputation = set(networkx.descendants(old_dag, removed_node))
                all_nodes_needing_recomputation.update(original_nodes_needing_recomputation)
        return all_nodes_needing_recomputation


@dataclasses.dataclass
class PipelinePatch(Patch, ABC):
    """ Parent class for pipeline patches """


@dataclasses.dataclass
class OperatorReplacement(PipelinePatch):
    """ Replace a DAG node with another one """

    operator_to_replace: DagNode
    replacement_operator: DagNode

    def apply(self, dag: networkx.DiGraph) -> networkx.DiGraph:
        replace_node(dag, self.operator_to_replace, self.replacement_operator)
        return dag

    def get_nodes_needing_recomputation(self, old_dag: networkx.DiGraph, new_dag: networkx.DiGraph):
        return self._get_nodes_needing_recomputation(old_dag, new_dag, [self.operator_to_replace],
                                                     [self.replacement_operator])


@dataclasses.dataclass
class OperatorRemoval(PipelinePatch):
    """ Remove a DAG node """

    # While operators to remove is a list, it should always be one main operator only, like a selection.
    #  The other operators are then the projections and subscripts for the filter condition.
    main_operator: DagNode
    operators_to_remove: List[DagNode]
    maybe_corresponding_test_set_operators: List[DagNode] or None
    before_train_test_split: bool

    def apply(self, dag: networkx.DiGraph):
        for node in self.operators_to_remove:
            remove_node(dag, node)
        for node in self.maybe_corresponding_test_set_operators or []:
            remove_node(dag, node)

    def get_nodes_needing_recomputation(self, old_dag: networkx.DiGraph, new_dag: networkx.DiGraph):
        return self._get_nodes_needing_recomputation(old_dag, new_dag, [*self.operators_to_remove,
                                                                        *(self.maybe_corresponding_test_set_operators
                                                                          or [])], [])


@dataclasses.dataclass
class AppendNodeAfterOperator(PipelinePatch):
    """ Remove a DAG node """

    operator_to_add_node_after: DagNode
    node_to_insert: DagNode

    def apply(self, dag: networkx.DiGraph):
        add_new_node_after_node(dag, self.node_to_insert, self.operator_to_add_node_after)

    def get_nodes_needing_recomputation(self, old_dag: networkx.DiGraph, new_dag: networkx.DiGraph):
        return self._get_nodes_needing_recomputation(old_dag, new_dag, [], [self.node_to_insert])


@dataclasses.dataclass
class DataPatch(Patch, ABC):
    """ Parent class for data patches """


@dataclasses.dataclass
class DataFiltering(DataPatch):
    """ Filter the train or test side """

    filter_operator: DagNode
    train_not_test: bool
    only_reads_column: List[str]

    def apply(self, dag: networkx.DiGraph):
        location, is_before_slit = find_dag_location_for_data_patch(self.only_reads_column, dag, self.train_not_test)
        if is_before_slit is False:
            add_new_node_after_node(dag, self.filter_operator, location)
        elif self.train_not_test is True:
            dag_part_name = "train" if self.train_not_test is True else "test"
            logger.warning(
                f"Columns {self.only_reads_column} not present in {dag_part_name} DAG after the train "
                f"test split, only before!")
            logger.warning(f"This could hint at data leakage!")
            add_new_node_after_node(dag, self.filter_operator, location)

    def get_nodes_needing_recomputation(self, old_dag: networkx.DiGraph, new_dag: networkx.DiGraph):
        _, is_before_slit = find_dag_location_for_data_patch(self.only_reads_column, old_dag, self.train_not_test)
        if (is_before_slit is False) or (self.train_not_test is True):
            added_operators = [self.filter_operator]
        else:
            added_operators = []
        return self._get_nodes_needing_recomputation(old_dag, new_dag, [], added_operators)


@dataclasses.dataclass
class DataProjection(DataPatch):
    """ Apply some map-like operation without fitting on the train or test side"""

    projection_operator: DagNode
    train_not_test: bool
    modifies_column: str
    only_reads_column: List[str]
    index_selection_func: Callable or None = None  # A function that can be used to select which rows to modify
    # A function that can be combined with index_selection_func to replace the projection_operator processing_func
    projection_func_only: Callable or None = None

    def apply(self, dag: networkx.DiGraph):

        columns_required = set()
        if self.only_reads_column is not None:
            columns_required.update(columns_required)
        columns_required.add(self.modifies_column)
        # This columns_required approach is not totally robust yet for user defined functions, will need to make the
        #  interface more explicit at some point.
        location, is_before_slit = find_dag_location_for_data_patch(columns_required, dag, self.train_not_test)
        if is_before_slit is False:
            add_new_node_after_node(dag, self.projection_operator, location)
        elif self.train_not_test is True:
            dag_part_name = "train" if self.train_not_test is True else "test"
            logger.warning(
                f"Columns {self.only_reads_column} not present in {dag_part_name} DAG after the train "
                f"test split, only before!")
            logger.warning(f"This could hint at data leakage!")
            add_new_node_after_node(dag, self.projection_operator, location)

    def get_nodes_needing_recomputation(self, old_dag: networkx.DiGraph, new_dag: networkx.DiGraph):
        return self._get_nodes_needing_recomputation(old_dag, new_dag, [], [self.projection_operator])


@dataclasses.dataclass
class DataTransformer(DataPatch):
    """ Fit a transformer on the train side and apply it to train and test side """

    fit_transform_operator: DagNode
    transform_operator: DagNode
    modifies_column: str

    def apply(self, dag: networkx.DiGraph):
        train_location, _ = find_dag_location_for_data_patch([self.modifies_column], dag, True)
        test_location, _ = find_dag_location_for_data_patch([self.modifies_column], dag, False)
        add_new_node_after_node(dag, self.fit_transform_operator, train_location)
        if train_location != test_location:
            add_new_node_after_node(dag, self.transform_operator, test_location, arg_index=1)
            dag.add_edge(self.fit_transform_operator, self.transform_operator, arg_index=0)

    def get_nodes_needing_recomputation(self, old_dag: networkx.DiGraph, new_dag: networkx.DiGraph):
        return self._get_nodes_needing_recomputation(old_dag, new_dag, [],
                                                     [self.fit_transform_operator, self.transform_operator])


@dataclasses.dataclass
class ModelPatch(Patch):
    """ Patch the model node by replacing with with another node """

    replace_with_node: DagNode

    def apply(self, dag: networkx.DiGraph):
        estimator_nodes = find_nodes_by_type(dag, OperatorType.ESTIMATOR)
        if len(estimator_nodes) != 1:
            raise Exception("Currently, ModelPatch only supports pipelines with exactly one estimator!")
        estimator_node = estimator_nodes[0]
        replace_node(dag, estimator_node, self.replace_with_node)

    def get_nodes_needing_recomputation(self, old_dag: networkx.DiGraph, new_dag: networkx.DiGraph):
        return self._get_nodes_needing_recomputation(old_dag, new_dag, [], [self.replace_with_node])
