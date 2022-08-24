"""
Functions for the implementation for the monkey patched functions
"""
import ast
import dataclasses
import sys
from typing import List

import numpy
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix

from mlwhatif.execution._stat_tracking import get_df_shape, get_df_memory
from mlwhatif.instrumentation._operator_types import OperatorContext, OperatorType
from mlwhatif.instrumentation import _pipeline_executor
from mlwhatif.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo, OptimizerInfo
from mlwhatif.instrumentation._pipeline_executor import singleton
from mlwhatif.monkeypatching._mlinspect_ndarray import MlinspectNdarray


@dataclasses.dataclass(frozen=False)
class FunctionCallResult:
    """ The annotated dataframe and the annotations for the current DAG node """
    function_result: any or None
    other: any = None  # TODO: input/output cardinality,


@dataclasses.dataclass(frozen=True)
class AnnotatedDfObject:
    """ A dataframe-like object and its annotations """
    result_data: any
    result_annotation: any


@dataclasses.dataclass(frozen=True)
class InputInfo:
    """ WIP experiments """
    dag_node: DagNode
    annotated_dfobject: AnnotatedDfObject


def execute_patched_func(original_func, execute_inspections_func, *args, **kwargs):
    """
    Detects whether the function call comes directly from user code and decides whether to execute the original
    function or the patched variant.
    """
    # Performance aspects: https://gist.github.com/JettJones/c236494013f22723c1822126df944b12
    # CPython implementation detail: This function should be used for internal and specialized purposes only.
    #  It is not guaranteed to exist in all implementations of Python.
    #  inspect.getcurrentframe() also only does return `sys._getframe(1) if hasattr(sys, "_getframe") else None`
    #  We can execute one hasattr check right at the beginning of the mlwhatif execution

    caller_filename = sys._getframe(2).f_code.co_filename  # pylint: disable=protected-access

    if caller_filename != singleton.source_code_path:
        result = original_func(*args, **kwargs)
    elif singleton.track_code_references:
        call_ast_node = ast.Call(lineno=singleton.lineno_next_call_or_subscript,
                                 col_offset=singleton.col_offset_next_call_or_subscript,
                                 end_lineno=singleton.end_lineno_next_call_or_subscript,
                                 end_col_offset=singleton.end_col_offset_next_call_or_subscript)
        caller_source_code = ast.get_source_segment(singleton.source_code, node=call_ast_node)
        caller_lineno = singleton.lineno_next_call_or_subscript
        op_id = singleton.get_next_op_id()
        caller_code_reference = CodeReference(singleton.lineno_next_call_or_subscript,
                                              singleton.col_offset_next_call_or_subscript,
                                              singleton.end_lineno_next_call_or_subscript,
                                              singleton.end_col_offset_next_call_or_subscript)
        result = execute_inspections_func(op_id, caller_filename, caller_lineno, caller_code_reference,
                                          caller_source_code)
    else:
        op_id = singleton.get_next_op_id()
        caller_lineno = sys._getframe(2).f_lineno  # pylint: disable=protected-access
        result = execute_inspections_func(op_id, caller_filename, caller_lineno, None, None)
    return result


def execute_patched_internal_func_with_depth(original_func, execute_inspections_func, depth, *args, **kwargs):
    """
    Detects whether the function call comes directly from user code and decides whether to execute the original
    function or the patched variant.
    """
    # Performance aspects: https://gist.github.com/JettJones/c236494013f22723c1822126df944b12
    # CPython implementation detail: This function should be used for internal and specialized purposes only.
    #  It is not guaranteed to exist in all implementations of Python.
    #  inspect.getcurrentframe() also only does return `sys._getframe(1) if hasattr(sys, "_getframe") else None`
    #  We can execute one hasattr check right at the beginning of the mlwhatif execution

    caller_filename = sys._getframe(depth).f_code.co_filename  # pylint: disable=protected-access

    if caller_filename != singleton.source_code_path:
        result = original_func(*args, **kwargs)
    elif singleton.track_code_references:
        call_ast_node = ast.Call(lineno=singleton.lineno_next_call_or_subscript,
                                 col_offset=singleton.col_offset_next_call_or_subscript,
                                 end_lineno=singleton.end_lineno_next_call_or_subscript,
                                 end_col_offset=singleton.end_col_offset_next_call_or_subscript)
        caller_source_code = ast.get_source_segment(singleton.source_code, node=call_ast_node)
        caller_lineno = singleton.lineno_next_call_or_subscript
        op_id = singleton.get_next_op_id()
        caller_code_reference = CodeReference(singleton.lineno_next_call_or_subscript,
                                              singleton.col_offset_next_call_or_subscript,
                                              singleton.end_lineno_next_call_or_subscript,
                                              singleton.end_col_offset_next_call_or_subscript)
        result = execute_inspections_func(op_id, caller_filename, caller_lineno, caller_code_reference,
                                          caller_source_code)
    else:
        op_id = singleton.get_next_op_id()
        caller_lineno = sys._getframe(2).f_lineno  # pylint: disable=protected-access
        result = execute_inspections_func(op_id, caller_filename, caller_lineno, None, None)
    return result


def execute_patched_func_no_op_id(original_func, execute_inspections_func, *args, **kwargs):
    """
    Detects whether the function call comes directly from user code and decides whether to execute the original
    function or the patched variant.
    """
    caller_filename = sys._getframe(2).f_code.co_filename  # pylint: disable=protected-access

    if caller_filename != singleton.source_code_path:
        result = original_func(*args, **kwargs)
    elif singleton.track_code_references:
        call_ast_node = ast.Call(lineno=singleton.lineno_next_call_or_subscript,
                                 col_offset=singleton.col_offset_next_call_or_subscript,
                                 end_lineno=singleton.end_lineno_next_call_or_subscript,
                                 end_col_offset=singleton.end_col_offset_next_call_or_subscript)
        caller_source_code = ast.get_source_segment(singleton.source_code, node=call_ast_node)
        caller_lineno = singleton.lineno_next_call_or_subscript
        caller_code_reference = CodeReference(singleton.lineno_next_call_or_subscript,
                                              singleton.col_offset_next_call_or_subscript,
                                              singleton.end_lineno_next_call_or_subscript,
                                              singleton.end_col_offset_next_call_or_subscript)
        result = execute_inspections_func(-1, caller_filename, caller_lineno, caller_code_reference,
                                          caller_source_code)
    else:
        caller_lineno = sys._getframe(2).f_lineno  # pylint: disable=protected-access
        result = execute_inspections_func(-1, caller_filename, caller_lineno, None, None)
    return result


def execute_patched_func_indirect_allowed(execute_inspections_func):
    """
    Detects whether the function call comes directly from user code and decides whether to execute the original
    function or the patched variant.
    """
    # Performance aspects: https://gist.github.com/JettJones/c236494013f22723c1822126df944b12
    # CPython implementation detail: This function should be used for internal and specialized purposes only.
    #  It is not guaranteed to exist in all implementations of Python.
    #  inspect.getcurrentframe() also only does return `sys._getframe(1) if hasattr(sys, "_getframe") else None`
    #  We can execute one hasattr check right at the beginning of the mlwhatif execution

    frame = sys._getframe(2)  # pylint: disable=protected-access
    while frame.f_code.co_filename != singleton.source_code_path:
        frame = frame.f_back

    caller_filename = frame.f_code.co_filename

    if singleton.track_code_references:
        call_ast_node = ast.Call(lineno=singleton.lineno_next_call_or_subscript,
                                 col_offset=singleton.col_offset_next_call_or_subscript,
                                 end_lineno=singleton.end_lineno_next_call_or_subscript,
                                 end_col_offset=singleton.end_col_offset_next_call_or_subscript)
        caller_source_code = ast.get_source_segment(singleton.source_code, node=call_ast_node)
        caller_lineno = singleton.lineno_next_call_or_subscript
        caller_code_reference = CodeReference(singleton.lineno_next_call_or_subscript,
                                              singleton.col_offset_next_call_or_subscript,
                                              singleton.end_lineno_next_call_or_subscript,
                                              singleton.end_col_offset_next_call_or_subscript)
        result = execute_inspections_func(-1, caller_filename, caller_lineno, caller_code_reference,
                                          caller_source_code)
    else:
        caller_lineno = sys._getframe(2).f_lineno  # pylint: disable=protected-access
        result = execute_inspections_func(-1, caller_filename, caller_lineno, None, None)
    return result


def get_input_info(df_object, caller_filename, lineno, function_info, optional_code_reference, optional_source_code) \
        -> InputInfo:
    """
    Uses the patched _mlinspect_dag_node attribute and the singleton.op_id_to_dag_node map to find the parent DAG node
    for the DAG node we want to insert in the next step.
    """
    # pylint: disable=too-many-arguments, unused-argument, protected-access, unused-variable, too-many-locals
    columns = get_column_names(df_object)
    if hasattr(df_object, "_mlinspect_dag_node"):
        input_op_id = df_object._mlinspect_dag_node
        input_dag_node = singleton.op_id_to_dag_node[input_op_id]
        input_info = InputInfo(input_dag_node, AnnotatedDfObject(df_object, None))  # TODO: Remove annotation stuff
    else:
        if optional_code_reference:
            code_reference = "({})".format(optional_source_code)
        else:
            code_reference = ""
        description = "Warning! Operator {}:{} {} encountered a DataFrame resulting from an operation " \
                      "without mlwhatif support!".format(caller_filename, lineno, code_reference)
        missing_op_id = singleton.get_next_missing_op_id()
        input_dag_node = DagNode(missing_op_id,
                                 BasicCodeLocation(caller_filename, lineno),
                                 OperatorContext(OperatorType.MISSING_OP, None),
                                 DagNodeDetails(description, columns,
                                                OptimizerInfo(None, get_df_shape(df_object), get_df_memory(df_object))),
                                 OptionalCodeInfo(optional_code_reference, optional_source_code))
        function_call_result = FunctionCallResult(df_object)
        add_dag_node(input_dag_node, [], function_call_result)
        input_info = InputInfo(input_dag_node, AnnotatedDfObject(df_object, None))  # TODO: Remove annotation stuff
    return input_info


def get_column_names(df_object):
    """Get column names for a dataframe ojbect"""
    if isinstance(df_object, DataFrame):
        columns = list(df_object.columns)  # TODO: Update this for numpy arrays etc. later
    elif isinstance(df_object, Series):
        columns = [df_object.name]
    elif isinstance(df_object, (csr_matrix, numpy.ndarray)):
        columns = ['array']
    else:
        raise NotImplementedError("TODO: Type: '{}' still is not supported!".format(type(df_object)))
    return columns


def wrap_in_mlinspect_array_if_necessary(df_object):
    """
    Makes sure annotations can be stored in a df_object. For example, numpy arrays need a wrapper for this.
    """
    if isinstance(df_object, numpy.ndarray):
        df_object = MlinspectNdarray(df_object)
    return df_object


def get_dag_node_copy_with_optimizer_info(dag_node: DagNode, optimizer_info: OptimizerInfo):
    """Because DagNodes are immutable, we need this fuction to create a copy where the optimizer_info is set"""
    new_node = DagNode(dag_node.node_id,
                       dag_node.code_location,
                       dag_node.operator_info,
                       DagNodeDetails(dag_node.details.description, dag_node.details.columns,
                                      optimizer_info),
                       dag_node.optional_code_info,
                       dag_node.processing_func)
    return new_node


def add_dag_node(dag_node: DagNode, dag_node_parents: List[DagNode], function_call_result: FunctionCallResult):
    """
    Inserts a new node into the DAG
    """
    # pylint: disable=protected-access
    # print("")
    # print("{}:{}: {}".format(dag_node.caller_filename, dag_node.lineno, dag_node.module))

    # print("source code: {}".format(dag_node.optional_source_code))
    if function_call_result.function_result is not None and dag_node.operator_info.operator != OperatorType.SCORE:
        function_call_result.function_result = wrap_in_mlinspect_array_if_necessary(
            function_call_result.function_result)
        function_call_result.function_result._mlinspect_dag_node = dag_node.node_id
    elif dag_node.operator_info.operator == OperatorType.SCORE:
        result_label = f"original_L{dag_node.code_location.lineno}"
        result_value = function_call_result.function_result
        singleton.original_pipeline_labels_to_extracted_plan_results[result_label] = result_value
    if dag_node_parents:
        for parent_index, parent in enumerate(dag_node_parents):
            singleton.analysis_results.original_dag.add_edge(parent, dag_node, arg_index=parent_index)
    else:
        singleton.analysis_results.original_dag.add_node(dag_node)
    singleton.op_id_to_dag_node[dag_node.node_id] = dag_node
    # if function_call_result.other is not None:
    # singleton.inspection_results.dag_node_to_inspection_results[dag_node] = backend_result.dag_node_annotation
    # TODO: Do we want to capture other meta information here? Or as part of the DAG node?


def get_dag_node_for_id(dag_node_id: int):
    """
    Get a DAG node by id
    """
    # pylint: disable=protected-access
    return singleton.op_id_to_dag_node[dag_node_id]


def get_optional_code_info_or_none(optional_code_reference: CodeReference or None,
                                   optional_source_code: str or None) -> OptionalCodeInfo or None:
    """
    If code reference tracking is enabled, return OptionalCodeInfo, otherwise None
    """
    if singleton.track_code_references:
        assert optional_code_reference is not None
        assert optional_source_code is not None
        code_info_or_none = OptionalCodeInfo(optional_code_reference, optional_source_code)
    else:
        assert optional_code_reference is None
        assert optional_source_code is None
        code_info_or_none = None
    return code_info_or_none


def add_train_label_node(estimator, train_label_arg, function_info):
    """Add a Train Data DAG Node for a estimator.fit call"""
    operator_context = OperatorContext(OperatorType.TRAIN_LABELS, function_info)
    input_info_train_labels = get_input_info(train_label_arg, estimator.mlinspect_caller_filename,
                                             estimator.mlinspect_lineno, function_info,
                                             estimator.mlinspect_optional_code_reference,
                                             estimator.mlinspect_optional_source_code)
    train_label_op_id = _pipeline_executor.singleton.get_next_op_id()
    process_func = lambda df_object: df_object
    train_labels_dag_node = DagNode(train_label_op_id,
                                    BasicCodeLocation(estimator.mlinspect_caller_filename, estimator.mlinspect_lineno),
                                    operator_context,
                                    DagNodeDetails(None, ["array"], OptimizerInfo(0, get_df_shape(train_label_arg),
                                                                                  get_df_memory(train_label_arg))),
                                    get_optional_code_info_or_none(estimator.mlinspect_optional_code_reference,
                                                                   estimator.mlinspect_optional_source_code),
                                    process_func)
    function_call_result = FunctionCallResult(train_label_arg)
    add_dag_node(train_labels_dag_node, [input_info_train_labels.dag_node], function_call_result)
    train_labels_result = function_call_result.function_result
    return function_call_result, train_labels_dag_node, train_labels_result


def add_train_data_node(estimator, train_data_arg, function_info):
    """Add a Train Label DAG Node for a estimator.fit call"""
    input_info_train_data = get_input_info(train_data_arg, estimator.mlinspect_caller_filename,
                                           estimator.mlinspect_lineno, function_info,
                                           estimator.mlinspect_optional_code_reference,
                                           estimator.mlinspect_optional_source_code)
    train_data_op_id = _pipeline_executor.singleton.get_next_op_id()
    operator_context = OperatorContext(OperatorType.TRAIN_DATA, function_info)
    process_func = lambda df_object: df_object
    train_data_dag_node = DagNode(train_data_op_id,
                                  BasicCodeLocation(estimator.mlinspect_caller_filename, estimator.mlinspect_lineno),
                                  operator_context,
                                  DagNodeDetails(None, ["array"], OptimizerInfo(0, get_df_shape(train_data_arg),
                                                                                get_df_memory(train_data_arg))),
                                  get_optional_code_info_or_none(estimator.mlinspect_optional_code_reference,
                                                                 estimator.mlinspect_optional_source_code),
                                  process_func)
    function_call_result = FunctionCallResult(train_data_arg)
    add_dag_node(train_data_dag_node, [input_info_train_data.dag_node], function_call_result)
    train_data_result = function_call_result.function_result
    return function_call_result, train_data_dag_node, train_data_result


def add_test_data_dag_node(test_data_arg, function_info, lineno, optional_code_reference, optional_source_code,
                           caller_filename):
    # pylint: disable=too-many-arguments
    """Add a Test Data DAG Node for a estimator.score call"""
    input_info_test_data = get_input_info(test_data_arg, caller_filename, lineno, function_info,
                                          optional_code_reference, optional_source_code)
    operator_context = OperatorContext(OperatorType.TEST_DATA, function_info)
    test_data_op_id = _pipeline_executor.singleton.get_next_op_id()
    process_func = lambda df_object: df_object
    test_data_dag_node = DagNode(test_data_op_id,
                                 BasicCodeLocation(caller_filename, lineno),
                                 operator_context,
                                 DagNodeDetails(None, get_column_names(test_data_arg),
                                                OptimizerInfo(0, get_df_shape(test_data_arg),
                                                              get_df_memory(test_data_arg))),
                                 get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                                 process_func)
    function_call_result = FunctionCallResult(test_data_arg)
    add_dag_node(test_data_dag_node, [input_info_test_data.dag_node], function_call_result)
    test_data_result = function_call_result.function_result
    return function_call_result, test_data_dag_node, test_data_result


def add_test_label_node(test_label_arg, caller_filename, function_info, lineno, optional_code_reference,
                        optional_source_code):
    """Add a Test Label DAG Node for a estimator.score call"""
    # pylint: disable=too-many-arguments
    operator_context = OperatorContext(OperatorType.TEST_LABELS, function_info)
    input_info_test_labels = get_input_info(test_label_arg, caller_filename, lineno, function_info,
                                            optional_code_reference, optional_source_code)
    test_label_op_id = _pipeline_executor.singleton.get_next_op_id()
    process_func = lambda df_object: df_object
    test_labels_dag_node = DagNode(test_label_op_id,
                                   BasicCodeLocation(caller_filename, lineno),
                                   operator_context,
                                   DagNodeDetails(None, get_column_names(test_label_arg),
                                                  OptimizerInfo(0, get_df_shape(test_label_arg),
                                                                get_df_memory(test_label_arg))),
                                   get_optional_code_info_or_none(optional_code_reference,
                                                                  optional_source_code),
                                   process_func)
    function_call_result = FunctionCallResult(test_label_arg)
    add_dag_node(test_labels_dag_node, [input_info_test_labels.dag_node], function_call_result)
    test_labels_result = function_call_result.function_result
    return function_call_result, test_labels_dag_node, test_labels_result
