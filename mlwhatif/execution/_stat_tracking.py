"""
Functionality to capture optimisation-relevant stats for instrumented operators
"""
import sys
import time
from functools import partial
from typing import Tuple

import pandas

from mlwhatif.instrumentation._dag_node import OptimizerInfo


def capture_optimzier_info(instrumented_function_call: partial) -> Tuple[OptimizerInfo, any]:
    """Function to measure the runtime of instrumented user function calls and get output metadata"""
    execution_start = time.time()
    result = instrumented_function_call()
    execution_duration = time.time() - execution_start
    execution_duration_in_ms = execution_duration * 1000
    if isinstance(result, pandas.DataFrame):
        shape = result.shape
    else:
        raise Exception(f"Result type {type(result).__name__} not supported yet!")
    size = sys.getsizeof(result)
    return OptimizerInfo(execution_duration_in_ms, shape, size), result
