"""
The Nodes used in the DAG as nodes for the networkx.DiGraph
"""
import dataclasses
from typing import List, Callable, Tuple

from mlmq.instrumentation._operator_types import OperatorContext


@dataclasses.dataclass
class OptimizerInfo:
    """
    Additional info about the DAG node
    """
    runtime: int or None = None  # in ms
    shape: Tuple[int, int] or None = None
    memory: int or None = None  # in bytes


@dataclasses.dataclass(frozen=True)
class CodeReference:
    """
    Identifies a function call in the user pipeline code
    """
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int


@dataclasses.dataclass
class BasicCodeLocation:
    """
    Basic information that can be collected even if `set_code_reference_tracking` is disabled
    """
    caller_filename: str
    lineno: int or None


@dataclasses.dataclass
class OptionalCodeInfo:
    """
    The additional information collected by mlmq if `set_code_reference_tracking` is enabled
    """
    code_reference: CodeReference
    source_code: str


@dataclasses.dataclass
class DagNodeDetails:
    """
    Additional info about the DAG node
    """
    description: str or None = None
    columns: List[str] or None = None
    optimizer_info: OptimizerInfo or None = dataclasses.field(compare=False, default=None)


@dataclasses.dataclass
class DagNode:
    """
    A DAG Node
    """

    node_id: int
    code_location: BasicCodeLocation
    operator_info: OperatorContext
    details: DagNodeDetails
    optional_code_info: OptionalCodeInfo or None = None
    processing_func: Callable or None = None
    make_classifier_func: Callable or None = None

    def __hash__(self):
        return hash(self.node_id)
