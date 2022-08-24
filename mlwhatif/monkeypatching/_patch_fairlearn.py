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
from mlwhatif.monkeypatching._monkey_patching_utils import add_dag_node, \
    get_optional_code_info_or_none, FunctionCallResult, \
    add_test_label_node, get_input_info, execute_patched_func_no_op_id


class FairLearnCallInfo:
    """ Contains info like whether some score function is already being processed """
    # pylint: disable=too-few-public-methods

    score_active = False


call_info_singleton = FairLearnCallInfo()


@gorilla.patches(metrics.MetricFrame)
class MetricFramePatching:
    """ Patches for 'pandas.core.frame' """

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *args, **kwargs):
        """ Patch for ('fairlearn.metrics._metric_frame', 'MetricFrame') """
        # pylint: disable=too-many-locals
        original = gorilla.get_original_attribute(metrics.MetricFrame, '__init__')

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('fairlearn.metrics._metric_frame', 'MetricFrame')

            input_info_pred = get_input_info(kwargs['y_pred'], caller_filename, lineno, function_info,
                                             optional_code_reference, optional_source_code)

            input_info_sensitive_cols = get_input_info(kwargs['sensitive_features'], caller_filename, lineno,
                                                       function_info, optional_code_reference, optional_source_code)

            # Test labels
            _, test_labels_node, _ = add_test_label_node(kwargs['y_true'],
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
            function_call_result = FunctionCallResult(self)
            add_dag_node(dag_node, [input_info_pred.dag_node, test_labels_node, input_info_sensitive_cols.dag_node],
                         function_call_result)
            return result

        if call_info_singleton.score_active is False:
            new_result = execute_patched_func_no_op_id(original, execute_inspections, self, *args, **kwargs)
        else:
            new_result = original(self, *args, **kwargs)
        return new_result


@gorilla.patches(metrics)
class MetricsPatching:
    """ Patches for 'fairlearn.metrics' """
    # pylint: disable=too-few-public-methods

    @gorilla.name('equalized_odds_difference')
    @gorilla.settings(allow_hit=True)
    def patched_equalized_odds_difference(y_true, y_pred, *args, **kwargs):
        """ Patch for ('fairlearn.metrics._disparities', 'equalized_odds_difference') """
        # pylint: disable=too-many-locals, no-self-argument
        original = gorilla.get_original_attribute(metrics, 'equalized_odds_difference')

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('fairlearn.metrics._disparities', 'equalized_odds_difference')

            input_info_pred = get_input_info(y_pred, caller_filename, lineno, function_info,
                                             optional_code_reference, optional_source_code)

            input_info_sensitive_cols = get_input_info(kwargs['sensitive_features'], caller_filename, lineno,
                                                       function_info, optional_code_reference, optional_source_code)

            # Test labels
            _, test_labels_node, _ = add_test_label_node(y_true,
                                                         caller_filename,
                                                         function_info,
                                                         lineno,
                                                         optional_code_reference,
                                                         optional_source_code)

            operator_context = OperatorContext(OperatorType.SCORE, function_info)
            initial_func = partial(original, y_true, y_pred, *args, **kwargs)
            call_info_singleton.score_active = True
            optimizer_info, result = capture_optimizer_info(initial_func)
            call_info_singleton.score_active = False

            def process_metric_frame(y_true, y_pred, sensitive_features):
                return original(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)

            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails('equalized_odds_difference', [], optimizer_info),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                               process_metric_frame)
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [input_info_pred.dag_node, test_labels_node, input_info_sensitive_cols.dag_node],
                         function_call_result)
            return result

        return execute_patched_func_no_op_id(original, execute_inspections, y_true, y_pred, *args, **kwargs)
