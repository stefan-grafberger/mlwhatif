"""
Functionality to capture optimisation-relevant stats for instrumented operators
"""
import sys
import time
from functools import partial
from typing import Tuple

from dill import dumps
import numpy
import pandas
import sklearn
import tensorflow
from fairlearn.metrics import MetricFrame
from scipy.sparse import csr_matrix
from tensorflow.python.keras.callbacks import History  # pylint: disable=no-name-in-module
from tensorflow.python.keras.wrappers.scikit_learn import BaseWrapper  # pylint: disable=no-name-in-module

from mlwhatif.instrumentation._dag_node import OptimizerInfo


def capture_optimizer_info(instrumented_function_call: partial, obj_for_inplace_ops: any or None = None,
                           estimator_transformer_state: any or None = None,
                           keras_batch_size: int or None = None)\
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
    size = get_df_memory(result_or_inplace_obj, estimator_transformer_state, keras_batch_size)
    return OptimizerInfo(execution_duration_in_ms, shape, size), result


def get_df_memory(result_or_inplace_obj, estimator_transformer_state: any or None = None,
                  keras_batch_size: int or None = None):
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
    elif isinstance(result_or_inplace_obj, (tuple, list)):
        size = sum([get_df_memory(elem) for elem in result_or_inplace_obj])
    else:
        size = sys.getsizeof(result_or_inplace_obj)
    if estimator_transformer_state is not None:
        if isinstance(estimator_transformer_state, (sklearn.base.BaseEstimator, sklearn.base.TransformerMixin)):
            size += sys.getsizeof(dumps(estimator_transformer_state))
        elif isinstance(estimator_transformer_state, BaseWrapper):
            size += get_model_memory_usage_in_bytes(estimator_transformer_state.model, keras_batch_size)
        else:
            raise Exception(f"Measuring the memory size of {type(estimator_transformer_state).__name__} is "
                            f"not supported yet!")
    return size


def get_model_memory_usage_in_bytes(model, batch_size):
    """Function to get the memory size of a Keras model"""
    # Based on https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        layer_type = layer.__class__.__name__
        if layer_type == 'Model':
            internal_model_mem_count += get_model_memory_usage_in_bytes(batch_size, layer)
        single_layer_mem = 1
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for dimension in out_shape:
            if dimension is None:
                continue
            single_layer_mem *= dimension
        shapes_mem_count += single_layer_mem

    trainable_count = numpy.sum([tensorflow.keras.backend.count_params(p) for p in model.trainable_weights])
    non_trainable_count = numpy.sum([tensorflow.keras.backend.count_params(p) for p in model.non_trainable_weights])

    number_size = 4
    if tensorflow.keras.backend.floatx() == 'float16':
        number_size = 2
    if tensorflow.keras.backend.floatx() == 'float64':
        number_size = 8

    total_memory = number_size * (batch_size * shapes_mem_count + trainable_count + non_trainable_count)
    return round(total_memory + internal_model_mem_count)


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
        if len(result_or_inplace_obj) > 1 and isinstance(result_or_inplace_obj[0], str):
            shape = (len(result_or_inplace_obj), 1)
        else:
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
    elif isinstance(result_or_inplace_obj, (float, MetricFrame)):
        # E.g., a score metric output from estimator.score
        shape = (1, 1)
    elif type(result_or_inplace_obj).__name__ == "TrainTestSplitResult":  # TODO: Clean up, currently circular import
        shape = None
    else:
        raise Exception(f"Result type {type(result_or_inplace_obj).__name__} not supported yet!")
    return shape
