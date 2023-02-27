"""
Util functions to make writing What-If Analyses easier
"""
import logging
from typing import Iterable

from mlwhatif.execution._patches import AppendNodeAfterOperator
from mlwhatif.instrumentation._dag_node import DagNode, DagNodeDetails
from mlwhatif.instrumentation._dag_node import OperatorContext
from mlwhatif.instrumentation._operator_types import OperatorType

logger = logging.getLogger(__name__)


def get_intermediate_extraction_patch_after_node(singleton, analysis: any or None, dag_node: DagNode, label: str):
    """Add a new node behind some given node to extract the intermediate result of that given node"""

    def extract_intermediate(intermediate_value):
        singleton.labels_to_extracted_plan_results[label] = intermediate_value
        return intermediate_value

    new_extraction_node = DagNode(singleton.get_next_op_id(),
                                  dag_node.code_location,
                                  OperatorContext(OperatorType.EXTRACT_RESULT, None),
                                  DagNodeDetails(None, dag_node.details.columns),
                                  None,
                                  extract_intermediate)
    return AppendNodeAfterOperator(singleton.get_next_patch_id(), analysis, False, dag_node, new_extraction_node)


def get_intermediate_extraction_patch_after_score_nodes(singleton, analysis: any or None, label: str,
                                                        score_nodes_and_linenos: Iterable[tuple[DagNode, int]]):
    """Add a new node behind some given node to extract the intermediate result of that given node"""
    patches = []
    for node, lineno in score_nodes_and_linenos:
        node_label = f"{label}_L{lineno}"
        patch = get_intermediate_extraction_patch_after_node(singleton, analysis, node, node_label)
        patches.append(patch)
    return patches
