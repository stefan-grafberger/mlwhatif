"""
Monkey patching for pandas
"""
# pylint: disable=too-many-lines
from functools import partial

import gorilla
import fuzzy_pandas

from mlmq import OperatorType, DagNode, BasicCodeLocation, DagNodeDetails
from mlmq.execution._stat_tracking import capture_optimizer_info
from mlmq.instrumentation._operator_types import OperatorContext, FunctionInfo
from mlmq.monkeypatching._monkey_patching_utils import execute_patched_func, get_input_info, add_dag_node, \
    get_optional_code_info_or_none, FunctionCallResult


@gorilla.patches(fuzzy_pandas)
class FuzzyPandasPatching:
    """ Patches for pandas """

    # pylint: disable=too-few-public-methods

    @gorilla.name('fuzzy_merge')
    @gorilla.settings(allow_hit=True)
    def patched_fuzzy_merge(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', 'merge') """
        original = gorilla.get_original_attribute(fuzzy_pandas, 'fuzzy_merge')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('fuzzy_pandas.fuzzy_merge', 'fuzzy_merge')

            input_info_a = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                          optional_source_code)
            if 'right' in kwargs:
                right_df = kwargs.pop('right')
                args_start_index = 0
            else:
                right_df = args[0]
                args_start_index = 1
            input_info_b = get_input_info(right_df, caller_filename, lineno, function_info, optional_code_reference,
                                          optional_source_code)
            operator_context = OperatorContext(OperatorType.JOIN, function_info)
            # No input_infos copy needed because it's only a selection and the rows not being removed don't change
            initial_func = partial(original, input_info_a.annotated_dfobject.result_data,
                                   input_info_b.annotated_dfobject.result_data,
                                   *args[args_start_index:],
                                   **kwargs)
            optimizer_info, result = capture_optimizer_info(initial_func)
            description = FuzzyPandasPatching.get_fuzzy_merge_description(**kwargs)
            processing_func = lambda df_a, df_b: original(df_a, df_b, *args[args_start_index:], **kwargs)
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, list(result.columns), optimizer_info),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                               processing_func)
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [input_info_a.dag_node, input_info_b.dag_node], function_call_result)
            new_result = function_call_result.function_result

            return new_result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @staticmethod
    def get_fuzzy_merge_description(**kwargs):
        """Get the description for a pd.merge call"""
        if 'on' in kwargs:
            description = f"approximately on '{kwargs['on']}'"
        elif ('left_on' in kwargs or kwargs['left_index'] is True) and ('right_on' in kwargs or
                                                                        kwargs['right_index'] is True):
            if 'left_on' in kwargs:
                left_column = f"'{kwargs['left_on']}'"
            else:
                left_column = 'left_index'
            if 'right_on' in kwargs:
                right_column = f"'{kwargs['right_on']}'"
            else:
                right_column = 'right_index'
            description = f"on {left_column} ~~ {right_column}"
        else:
            description = None
        if 'how' in kwargs and description is not None:
            description += f" ({kwargs['how']})"
        elif 'how' in kwargs:
            description = f"({kwargs['how']})"
        return description
