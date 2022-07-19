"""
Functionality to capture optimisation-relevant stats for instrumented operators
"""
import sys
import time
from functools import partial
from typing import Tuple

import pandas

from mlwhatif.instrumentation._dag_node import OptimizerInfo


def capture_optimizer_info(instrumented_function_call: partial, obj_for_inplace_ops: any or None = None)\
        -> Tuple[OptimizerInfo, any]:
    """Function to measure the runtime of instrumented user function calls and get output metadata"""
    execution_start = time.time()
    result = instrumented_function_call()
    execution_duration = time.time() - execution_start
    execution_duration_in_ms = execution_duration * 1000
    if result is not None:
        result_or_inplace_obj = result
    else:
        result_or_inplace_obj = obj_for_inplace_ops

    if isinstance(result_or_inplace_obj, pandas.DataFrame):
        shape = result_or_inplace_obj.shape
    elif isinstance(result_or_inplace_obj, pandas.Series):
        shape = len(result_or_inplace_obj), 1
    elif isinstance(result_or_inplace_obj, pandas.core.groupby.generic.DataFrameGroupBy):
        shape = None  # Not needed because we only consider combined groupby/agg nodes
    else:
        raise Exception(f"Result type {type(result).__name__} not supported yet!")
    size = sys.getsizeof(result_or_inplace_obj)
    return OptimizerInfo(execution_duration_in_ms, shape, size), result
