"""
Monkey patching for FairLearn
"""
# pylint: disable=too-many-lines
from functools import partial

import gorilla
from fairlearn import metrics

from mlwhatif import OperatorType, DagNode, BasicCodeLocation, DagNodeDetails
from mlwhatif.execution._stat_tracking import capture_optimizer_info
from mlwhatif.instrumentation._operator_types import OperatorContext, FunctionInfo
from mlwhatif.instrumentation._pipeline_executor import singleton
from mlwhatif.monkeypatching._monkey_patching_utils import execute_patched_func, add_dag_node, \
    get_optional_code_info_or_none, FunctionCallResult, \
    add_test_label_node, get_input_info, execute_patched_func_no_op_id


@gorilla.patches(metrics.MetricFrame)
class MetricFramePatching:
    """ Patches for 'pandas.core.frame' """

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *args, **kwargs):
        """ Patch for ('fairlearn.metrics._metric_frame', 'MetricFrame') """
        original = gorilla.get_original_attribute(metrics.MetricFrame, '__init__')

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('fairlearn.metrics._metric_frame', 'MetricFrame')

            input_info_pred = get_input_info(kwargs['y_pred'], caller_filename, lineno, function_info,
                                             optional_code_reference, optional_source_code)

            input_info_sensitive_cols = get_input_info(kwargs['sensitive_features'], caller_filename, lineno,
                                                       function_info, optional_code_reference, optional_source_code)

            # Test labels
            _, test_labels_node, test_labels_result = add_test_label_node(kwargs['y_true'],
                                                                          caller_filename,
                                                                          function_info,
                                                                          lineno,
                                                                          optional_code_reference,
                                                                          optional_source_code)

            operator_context = OperatorContext(OperatorType.SCORE, function_info)
            initial_func = partial(original, self, *args, **kwargs)
            optimizer_info, result = capture_optimizer_info(initial_func, self)

            def process_metric_frame(bound_metric, y_true, y_pred, sensitive_features):
                return metrics.MetricFrame(metrics=bound_metric, y_true=y_true, y_pred=y_pred,
                                           sensitive_features=sensitive_features)

            process_func = partial(process_metric_frame, kwargs['metrics'])

            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(kwargs['metrics'].__name__, [], optimizer_info),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                               process_func)
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [input_info_pred.dag_node, test_labels_node, input_info_sensitive_cols.dag_node],
                         function_call_result)

        execute_patched_func_no_op_id(original, execute_inspections, *args, **kwargs)
