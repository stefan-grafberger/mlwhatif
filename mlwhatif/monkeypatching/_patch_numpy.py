"""
Monkey patching for numpy
"""
from functools import partial

import gorilla
from numpy import random

from mlwhatif import DagNode, BasicCodeLocation, DagNodeDetails
from mlwhatif.instrumentation._operator_types import OperatorContext, FunctionInfo, OperatorType
from mlwhatif.monkeypatching._monkey_patching_utils import execute_patched_func, add_dag_node, \
    get_optional_code_info_or_none, FunctionCallResult


@gorilla.patches(random)
class NumpyRandomPatching:
    """ Patches for sklearn """

    # pylint: disable=too-few-public-methods

    @gorilla.name('random')
    @gorilla.settings(allow_hit=True)
    def patched_random(*args, **kwargs):
        """ Patch for ('numpy.random', 'random') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(random, 'random')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('numpy.random', 'random')
            operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
            result = original(*args, **kwargs)
            processing_func = partial(original, *args, **kwargs)
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("random", ['array']),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                               processing_func)
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [], function_call_result)
            new_return_value = function_call_result.function_result
            return new_return_value

        return execute_patched_func(original, execute_inspections, *args, **kwargs)
