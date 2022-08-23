""" Contains all the different patch-related classes """
import dataclasses
import logging
from abc import ABC, abstractmethod
from typing import List, Callable, Iterable

import networkx

from mlwhatif.instrumentation._operator_types import OperatorType
from mlwhatif.analysis._analysis_utils import find_dag_location_for_data_patch, add_new_node_after_node, \
    find_nodes_by_type, replace_node, remove_node, get_sorted_parent_nodes
from mlwhatif.instrumentation._dag_node import DagNode, OptimizerInfo, DagNodeDetails

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PipelinePatch(ABC):
    """ Basic Patch class """
    patch_id: int
    analysis: any  # WhatIfAnalyis but this would be a circular import currently
    changes_following_results: bool

    @abstractmethod
    def apply(self, dag: networkx.DiGraph, pipeline_executor):
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
class OperatorPatch(PipelinePatch, ABC):
    """ Parent class for pipeline patches """


@dataclasses.dataclass
class OperatorReplacement(OperatorPatch):
    """ Replace a DAG node with another one """

    operator_to_replace: DagNode
    replacement_operator: DagNode

    def apply(self, dag: networkx.DiGraph, pipeline_executor) -> networkx.DiGraph:
        replace_node(dag, self.operator_to_replace, self.replacement_operator)
        return dag

    def get_nodes_needing_recomputation(self, old_dag: networkx.DiGraph, new_dag: networkx.DiGraph):
        return self._get_nodes_needing_recomputation(old_dag, new_dag, [self.operator_to_replace],
                                                     [self.replacement_operator])


@dataclasses.dataclass
class OperatorRemoval(OperatorPatch):
    """ Remove a DAG node """

    # While operators to remove is a list, it should always be one main operator only, like a selection.
    #  The other operators are then the projections and subscripts for the filter condition.
    operator_to_remove: DagNode
    maybe_corresponding_test_set_operator: DagNode or None
    before_train_test_split: bool

    def apply(self, dag: networkx.DiGraph, pipeline_executor):
        # We only have one use-case with OperatorRemoval, some very minor updates are required here once this changes,
        #  we only wan to update selectivities if we actually have a filter
        assert self.operator_to_remove.operator_info.operator == OperatorType.SELECTION
        selectivity = self.compute_filter_selectivity(dag)

        all_operators_to_remove = self.get_all_operators_associated_with_filter(dag, self.operator_to_remove)
        all_operators_to_remove.update(self.get_all_operators_associated_with_filter(
            dag, self.maybe_corresponding_test_set_operator))

        all_nodes_to_update = set()
        for removed_node in all_operators_to_remove:
            original_nodes_needing_recomputation = set(networkx.descendants(dag, removed_node))
            all_nodes_to_update.update(original_nodes_needing_recomputation)
        all_nodes_to_update.difference_update(all_operators_to_remove)

        for node in all_operators_to_remove:
            remove_node(dag, node)

        self.update_optimizer_info_with_selectivity_info(dag, all_nodes_to_update, selectivity)

    def get_nodes_needing_recomputation(self, old_dag: networkx.DiGraph, new_dag: networkx.DiGraph):
        nodes_being_removed_train = self.get_all_operators_associated_with_filter(old_dag, self.operator_to_remove)
        nodes_being_removed_test = self.get_all_operators_associated_with_filter(
            old_dag, self.maybe_corresponding_test_set_operator)
        all_nodes_being_removed = nodes_being_removed_train.union(nodes_being_removed_test)
        return self._get_nodes_needing_recomputation(old_dag, new_dag, all_nodes_being_removed, [])

    def get_all_operators_associated_with_filter(self, modified_dag, operator_to_replace, keep_root=False) \
            -> set[DagNode]:
        """Get all ops associated with a filter, like subscripts and projections for the filter condition"""
        if operator_to_replace is None or operator_to_replace.operator_info.operator != OperatorType.SELECTION:
            return set()
        operator_after_which_cutoff_required = self.filter_get_operator_after_which_cutoff_required(
            modified_dag, operator_to_replace)
        paths_between_generator = networkx.all_simple_paths(modified_dag,
                                                            source=operator_after_which_cutoff_required,
                                                            target=operator_to_replace)
        associated_operators = {node for path in paths_between_generator for node in path}
        if keep_root is False:
            associated_operators.remove(operator_after_which_cutoff_required)
        return associated_operators

    @staticmethod
    def filter_get_operator_after_which_cutoff_required(modified_dag, operator_to_replace) -> DagNode:
        """Get the operator after which the filter code starts with the filter condition eval etc."""
        if operator_to_replace.operator_info.operator != OperatorType.SELECTION:
            return operator_to_replace
        operator_parent_nodes = get_sorted_parent_nodes(modified_dag, operator_to_replace)
        selection_parent_a = operator_parent_nodes[0]
        selection_parent_b = operator_parent_nodes[-1]
        # We want to introduce the change before all subscript behavior
        operator_after_which_cutoff_required = networkx.lowest_common_ancestor(modified_dag, selection_parent_a,
                                                                               selection_parent_b)
        return operator_after_which_cutoff_required

    @staticmethod
    def update_optimizer_info_with_selectivity_info(dag: networkx.DiGraph, nodes_to_update: Iterable[DagNode],
                                                    selectivity: float):
        """ This updates the optimizer info of all dependent dag nodes """
        nodes_to_update = set(nodes_to_update)
        node_ids_to_fix = set()

        def update_selectivity_func(dag_node: DagNode) -> DagNode:
            if dag_node not in nodes_to_update:
                result = dag_node
            else:
                result = OperatorRemoval._update_dag_node_optimizer_info(dag_node, selectivity)
                node_ids_to_fix.add(result.node_id)
            return result

        def finalise_node_update(dag_node: DagNode) -> DagNode:
            result = dag_node
            if dag_node.node_id in node_ids_to_fix:
                # necessary because otherwise equals true and network does nothing
                result = DagNode(
                    -dag_node.node_id,
                    dag_node.code_location,
                    dag_node.operator_info,
                    dag_node.details,
                    dag_node.optional_code_info,
                    dag_node.processing_func,
                    dag_node.make_classifier_func
                )
            return result

        # two relabels necessary because otherwise equals true and network does nothing
        # noinspection PyTypeChecker
        networkx.relabel_nodes(dag, update_selectivity_func, copy=False)
        # noinspection PyTypeChecker
        networkx.relabel_nodes(dag, finalise_node_update, copy=False)

    @staticmethod
    def _update_dag_node_optimizer_info(node_to_recompute, selectivity) -> DagNode:
        """Updates the OptimizerInfo for a single DagNode"""
        # This assumes filters are not correlated etc
        if node_to_recompute.details.optimizer_info is not None:
            if node_to_recompute.details.optimizer_info.shape is not None:
                updated_shape = (node_to_recompute.details.optimizer_info.shape[0] * (1 / selectivity),
                                 node_to_recompute.details.optimizer_info.shape[1])
            else:
                updated_shape = None
            if node_to_recompute.operator_info.operator != OperatorType.ESTIMATOR:
                new_memory_value = node_to_recompute.details.optimizer_info.memory * (1 / selectivity)
            else:
                new_memory_value = node_to_recompute.details.optimizer_info.memory
            updated_optimizer_info = OptimizerInfo(
                node_to_recompute.details.optimizer_info.runtime * (1 / selectivity),
                updated_shape,
                new_memory_value
            )
        else:
            updated_optimizer_info = None
        replacement_node = DagNode(-node_to_recompute.node_id,
                                   node_to_recompute.code_location,
                                   node_to_recompute.operator_info,
                                   DagNodeDetails(
                                       node_to_recompute.details.description,
                                       node_to_recompute.details.columns,
                                       updated_optimizer_info
                                   ),
                                   node_to_recompute.optional_code_info,
                                   node_to_recompute.processing_func,
                                   node_to_recompute.make_classifier_func)
        return replacement_node

    def compute_filter_selectivity(self, dag):
        """Compute the selecivity of the filter being removed"""
        assert self.operator_to_remove.operator_info.operator == OperatorType.SELECTION
        parent = list(dag.predecessors(self.operator_to_remove))[0]
        parent_row_count = parent.details.optimizer_info.shape[0]
        current_row_count = self.operator_to_remove.details.optimizer_info.shape[0]
        selectivity = current_row_count / parent_row_count
        return selectivity


@dataclasses.dataclass
class AppendNodeAfterOperator(OperatorPatch):
    """ Remove a DAG node """

    operator_to_add_node_after: DagNode
    node_to_insert: DagNode

    def apply(self, dag: networkx.DiGraph, pipeline_executor):
        add_new_node_after_node(dag, self.node_to_insert, self.operator_to_add_node_after)

    def get_nodes_needing_recomputation(self, old_dag: networkx.DiGraph, new_dag: networkx.DiGraph):
        return self._get_nodes_needing_recomputation(old_dag, new_dag, [], [self.node_to_insert])


@dataclasses.dataclass
class DataPatch(PipelinePatch, ABC):
    """ Parent class for data patches """


@dataclasses.dataclass
class DataFiltering(DataPatch):
    """ Filter the train or test side """

    filter_operator: DagNode
    train_not_test: bool
    only_reads_column: List[str]

    def apply(self, dag: networkx.DiGraph, pipeline_executor):
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
class UdfSplitInfo:
    """
    Info required to split up expensive udf projections that only repeatedly modify subsets of rows into
    two parts: corrupting all rows once, and then only sampling from the corrupted variants, isntead of calling the
    expensive udf repeatedly
    """
    index_selection_func_id: int
    index_selection_func: Callable  # A function that can be used to select which rows to modify
    # A function that can be combined with index_selection_func to replace the projection_operator processing_func
    projection_func_only_id: int
    projection_func_only: Callable
    column_name_to_corrupt: str
    maybe_selectivity_info: float or None = None


@dataclasses.dataclass
class DataProjection(DataPatch):
    """ Apply some map-like operation without fitting on the train or test side"""
    # pylint: disable=too-many-instance-attributes

    projection_operator: DagNode
    train_not_test: bool
    modifies_column: str
    only_reads_column: List[str]
    maybe_udf_split_info: UdfSplitInfo


    def apply(self, dag: networkx.DiGraph, pipeline_executor):

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

    def apply(self, dag: networkx.DiGraph, pipeline_executor):
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
class ModelPatch(PipelinePatch):
    """ Patch the model node by replacing with with another node """

    replace_with_node: DagNode

    def apply(self, dag: networkx.DiGraph, pipeline_executor):
        estimator_nodes = find_nodes_by_type(dag, OperatorType.ESTIMATOR)
        if len(estimator_nodes) != 1:
            raise Exception("Currently, ModelPatch only supports pipelines with exactly one estimator!")
        estimator_node = estimator_nodes[0]
        replace_node(dag, estimator_node, self.replace_with_node)

    def get_nodes_needing_recomputation(self, old_dag: networkx.DiGraph, new_dag: networkx.DiGraph):
        return self._get_nodes_needing_recomputation(old_dag, new_dag, [], [self.replace_with_node])
