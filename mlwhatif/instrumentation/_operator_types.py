"""
Data classes used as input for the inspections
"""
import dataclasses
from enum import Enum
from typing import List


@dataclasses.dataclass(frozen=True)
class ColumnInfo:
    """
    A class we use to efficiently pass pandas/sklearn rows
    """
    fields: List[str]

    def get_index_of_column(self, column_name):
        """
        Get the values index for some column
        """
        if column_name in self.fields:
            return self.fields.index(column_name)
        return None

    def __eq__(self, other):
        return (isinstance(other, ColumnInfo) and
                self.fields == other.fields)


@dataclasses.dataclass(frozen=True)
class FunctionInfo:
    """
    Contains the function name and its path
    """
    module: str
    function_name: str


class OperatorType(Enum):
    """
    The different operator types in our DAG
    """
    DATA_SOURCE = "Data Source"
    MISSING_OP = "Encountered unsupported operation! Fallback: Data Source"
    SELECTION = "Selection"
    PROJECTION = "Projection"
    PROJECTION_MODIFY = "Projection (Modify)"
    TRANSFORMER = "Transformer"
    CONCATENATION = "Concatenation"
    ESTIMATOR = "Estimator"
    PREDICT = "Predict"
    SCORE = "Score"
    TRAIN_DATA = "Train Data"
    TRAIN_LABELS = "Train Labels"
    TEST_DATA = "Test Data"
    TEST_LABELS = "Test Labels"
    JOIN = "Join"
    GROUP_BY_AGG = "Groupby and Aggregate"
    TRAIN_TEST_SPLIT = "Train Test Split"
    SUBSCRIPT = "Subscript"
    EXTRACT_RESULT = "Extract intermediate result"


@dataclasses.dataclass(frozen=True)
class OperatorContext:
    """
    Additional context for the inspection. Contains, most importantly, the operator type.
    """
    operator: OperatorType
    function_info: FunctionInfo or None
