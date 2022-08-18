""" Contains all the different patch-related classes """
import dataclasses
import logging

import networkx

from mlwhatif.analysis._analysis_utils import add_new_node_between_nodes
from mlwhatif.execution._patches import PipelinePatch
from mlwhatif.instrumentation._dag_node import DagNode

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AppendNodeBetweenOperators(PipelinePatch):
    """ Remove a DAG node """

    node_to_insert: DagNode
    operator_to_add_node_after: DagNode
    operator_to_add_node_before: DagNode

    def apply(self, dag: networkx.DiGraph):
        add_new_node_between_nodes(dag, self.node_to_insert, self.operator_to_add_node_after,
                                   self.operator_to_add_node_before)

    def get_nodes_needing_recomputation(self, old_dag: networkx.DiGraph, new_dag: networkx.DiGraph):
        return self._get_nodes_needing_recomputation(old_dag, new_dag, [], [self.node_to_insert])


@dataclasses.dataclass
class PipelineTransformerInsertion(PipelinePatch):
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
