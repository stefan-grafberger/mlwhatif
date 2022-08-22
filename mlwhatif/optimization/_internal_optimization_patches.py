""" Contains all the different patch-related classes """
import dataclasses
import logging

import networkx

from mlwhatif.analysis._analysis_utils import add_new_node_between_nodes
from mlwhatif.execution._patches import OperatorPatch, UdfSplitInfo
from mlwhatif.instrumentation._dag_node import DagNode

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AppendNodeBetweenOperators(OperatorPatch):
    """ Remove a DAG node """

    node_to_insert: DagNode
    operator_to_add_node_after: DagNode
    operator_to_add_node_before: DagNode
    maybe_udf_split_info: UdfSplitInfo
    train_not_test: bool

    def apply(self, dag: networkx.DiGraph):
        add_new_node_between_nodes(dag, self.node_to_insert, self.operator_to_add_node_after,
                                   self.operator_to_add_node_before)
        # TODO: Handle case where maybe_udf_split_info is not None and use it

    def get_nodes_needing_recomputation(self, old_dag: networkx.DiGraph, new_dag: networkx.DiGraph):
        # TODO: Handle case where maybe_udf_split_info is not None and use it
        #   Then, we only want to pass the second node as needing recomputation
        return self._get_nodes_needing_recomputation(old_dag, new_dag, [], [self.node_to_insert])


@dataclasses.dataclass
class UdfSplitAndReuseAppendNodeBetweenOperators(OperatorPatch):
    """ Remove a DAG node """

    operator_to_add_node_after: DagNode
    operator_to_add_node_before: DagNode
    corruption_node: DagNode
    index_selection_node: DagNode
    apply_corruption_to_fraction_node: DagNode
    train_not_test: bool
    first_index_function_occurrence: bool

    def apply(self, dag: networkx.DiGraph):
        add_new_node_between_nodes(dag, self.apply_corruption_to_fraction_node, self.operator_to_add_node_after,
                                   self.operator_to_add_node_before)
        if self.first_index_function_occurrence is True:
            # We need to get the data frame len once only to create the index selection mask
            dag.add_edge(self.operator_to_add_node_after, self.index_selection_node, arg_index=0)
        dag.add_edge(self.operator_to_add_node_after, self.corruption_node, arg_index=0)
        dag.add_edge(self.corruption_node, self.apply_corruption_to_fraction_node, arg_index=1)
        dag.add_edge(self.index_selection_node, self.apply_corruption_to_fraction_node, arg_index=2)

    def get_nodes_needing_recomputation(self, old_dag: networkx.DiGraph, new_dag: networkx.DiGraph):
        return self._get_nodes_needing_recomputation(old_dag, new_dag, [], [self.apply_corruption_to_fraction_node])


@dataclasses.dataclass
class OperatorTransformerInsertion(OperatorPatch):
    """ Remove a DAG node """

    fit_transform_node_to_insert: DagNode
    operator_to_add_fit_transform_node_after: DagNode
    operator_to_add_fit_transform_node_before: DagNode

    transform_node_to_insert: DagNode
    operator_to_add_transform_node_after: DagNode
    operator_to_add_transform_node_before: DagNode

    def apply(self, dag: networkx.DiGraph):
        add_new_node_between_nodes(dag, self.fit_transform_node_to_insert,
                                   self.operator_to_add_fit_transform_node_after,
                                   self.operator_to_add_fit_transform_node_before)
        if (self.operator_to_add_fit_transform_node_after, self.operator_to_add_fit_transform_node_before) != \
                (self.operator_to_add_transform_node_after, self.operator_to_add_transform_node_before):
            add_new_node_between_nodes(dag, self.transform_node_to_insert,
                                       self.operator_to_add_transform_node_after,
                                       self.operator_to_add_transform_node_before,
                                       arg_index=1)
            dag.add_edge(self.fit_transform_node_to_insert, self.transform_node_to_insert, arg_index=0)

    def get_nodes_needing_recomputation(self, old_dag: networkx.DiGraph, new_dag: networkx.DiGraph):
        return self._get_nodes_needing_recomputation(old_dag, new_dag, [], [self.fit_transform_node_to_insert,
                                                                            self.transform_node_to_insert])
