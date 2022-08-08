""" Contains all the different patch-related classes """
import dataclasses

from mlwhatif.instrumentation._dag_node import DagNode
from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis


@dataclasses.dataclass
class Patch:
    """ Basic Patch class """
    patch_id: int
    analysis: WhatIfAnalysis
    changes_following_results: bool


@dataclasses.dataclass
class PipelinePatch(Patch):
    """ Parent class for pipeline patches """


@dataclasses.dataclass
class OperatorReplacement(PipelinePatch):
    """ Replace a DAG node with another one """
    operator_to_replace: DagNode
    replacement_operator: DagNode


@dataclasses.dataclass
class OperatorRemoval(PipelinePatch):
    """ Remove a DAG node """
    operator_to_remove: DagNode


@dataclasses.dataclass
class AppendNodeAfterOperator(PipelinePatch):
    """ Remove a DAG node """
    operator_to_add_node_after: DagNode
    node_to_insert: DagNode


@dataclasses.dataclass
class DataPatch(Patch):
    """ Parent class for data patches """


@dataclasses.dataclass
class DataFiltering(DataPatch):
    """ Filter the train or test side """
    filter_operator: DagNode
    train_not_test: bool


@dataclasses.dataclass
class DataProjection(DataPatch):
    """ Apply some map-like operation without fitting on the train or test side"""
    projection_operator: DagNode
    train_not_test: bool


@dataclasses.dataclass
class DataTransformer(DataPatch):
    """ Fit a transformer on the train side and apply it to train and test side """
    filter_operator: DagNode


@dataclasses.dataclass
class ModelPatch(Patch):
    """ Patch the model node by replacing with with another node """
    replace_with_node: DagNode
