"""
Functionality to capture optimisation-relevant stats for instrumented operators
"""
import sys
import time
from functools import partial
from typing import Tuple

import numpy
import pandas
import sklearn
from scipy.sparse import csr_matrix
from tensorflow.python.keras.callbacks import History  # pylint: disable=no-name-in-module

from mlwhatif.instrumentation._dag_node import OptimizerInfo


def capture_optimizer_info(instrumented_function_call: partial, obj_for_inplace_ops: any or None = None,
                           estimator_transformer_state: any or None = None)\
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

    shape = get_df_shape(result_or_inplace_obj)
    size = get_df_memory(result_or_inplace_obj, estimator_transformer_state)
    return OptimizerInfo(execution_duration_in_ms, shape, size), result


def get_df_memory(result_or_inplace_obj, estimator_transformer_state: any or None = None):
    """Get the size in bytes of a df-like object"""
    # Just using sys.getsize of is not sufficient. See this section of its documentation:
    #  Only the memory consumption directly attributed to the object is accounted for,
    #  not the memory consumption of objects it refers to.
    if isinstance(result_or_inplace_obj, pandas.DataFrame):
        # For pandas, sizeof seems to work as expected
        size = sys.getsizeof(result_or_inplace_obj)
    elif isinstance(result_or_inplace_obj, numpy.ndarray):
        system_size = sys.getsizeof(result_or_inplace_obj)
        numpy_self_report = result_or_inplace_obj.nbytes
        if system_size >= numpy_self_report:
            size = system_size
        else:
            size = system_size + numpy_self_report
    elif isinstance(result_or_inplace_obj, csr_matrix):
        size = result_or_inplace_obj.data.nbytes + sys.getsizeof(result_or_inplace_obj)
    else:
        size = sys.getsizeof(result_or_inplace_obj)
    if estimator_transformer_state is not None:
        size += sys.getsizeof(estimator_transformer_state)
    return size


def get_df_shape(result_or_inplace_obj):
    """Get the shape of a df-like object"""
    if isinstance(result_or_inplace_obj, pandas.DataFrame):
        shape = result_or_inplace_obj.shape
    elif isinstance(result_or_inplace_obj, numpy.ndarray):
        if result_or_inplace_obj.ndim == 2:
            shape = result_or_inplace_obj.shape
        elif result_or_inplace_obj.ndim == 1:
            shape = len(result_or_inplace_obj), 1
        else:
            raise Exception("Currently only numpy arrays with 1 and 2 dims are supported!")
    elif isinstance(result_or_inplace_obj, pandas.Series):
        shape = len(result_or_inplace_obj), 1
    elif isinstance(result_or_inplace_obj, pandas.core.groupby.generic.DataFrameGroupBy):
        shape = None  # Not needed because we only consider combined groupby/agg nodes
    elif isinstance(result_or_inplace_obj, list):
        # A few operations like train_test_split return a list
        assert len(result_or_inplace_obj) == 2
        shape_a = get_df_shape(result_or_inplace_obj[0])
        shape_b = get_df_shape(result_or_inplace_obj[1])
        assert shape_a[1] == shape_b[1]
        shape = shape_a[0] + shape_b[0], shape_a[1]
    elif isinstance(result_or_inplace_obj, csr_matrix):
        # Here we use the csr_matrix column count as width instead of treating it as 1 column only as we do logically.
        #  This is because we might need it for optimisation purposes to choose whether dense or sparse matrices are
        #  better. We might want to potentially change this in the future.
        shape = result_or_inplace_obj.shape
    elif isinstance(result_or_inplace_obj, (sklearn.base.BaseEstimator, History)):
        # Estimator fit only, not fit_transform
        shape = None
    elif isinstance(result_or_inplace_obj, float):
        # E.g., a score metric output from estimator.score
        shape = (1, 1)
    else:
        raise Exception(f"Result type {type(result_or_inplace_obj).__name__} not supported yet!")
    return shape
