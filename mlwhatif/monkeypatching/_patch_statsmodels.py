"""
Monkey patching for numpy
"""
import gorilla
from statsmodels import api
from statsmodels.api import datasets

from mlwhatif import DagNode, BasicCodeLocation, DagNodeDetails
from mlwhatif.instrumentation._operator_types import OperatorContext, FunctionInfo, OperatorType
from mlwhatif.instrumentation._pipeline_executor import singleton
from mlwhatif.monkeypatching._monkey_patching_utils import execute_patched_func, add_dag_node, \
    get_optional_code_info_or_none, get_input_info, execute_patched_func_no_op_id, add_train_data_node, \
    add_train_label_node, FunctionCallResult


@gorilla.patches(api)
class StatsmodelApiPatching:
    """ Patches for statsmodel """
    # pylint: disable=too-few-public-methods

    @gorilla.name('add_constant')
    @gorilla.settings(allow_hit=True)
    def patched_random(*args, **kwargs):
        """ Patch for ('statsmodel.api', 'add_constant') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(api, 'add_constant')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('statsmodel.api', 'add_constant')
            input_info = get_input_info(args[0], caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)

            operator_context = OperatorContext(OperatorType.PROJECTION_MODIFY, function_info)
            result = original(input_info.annotated_dfobject.result_data, *args[1:], **kwargs)

            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("Adds const column", ["array"]),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))

            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result

            return new_result
        return execute_patched_func(original, execute_inspections, *args, **kwargs)


@gorilla.patches(datasets)
class StatsmodelsDatasetPatching:
    """ Patches for pandas """

    # pylint: disable=too-few-public-methods

    @gorilla.name('get_rdataset')
    @gorilla.settings(allow_hit=True)
    def patched_read_csv(*args, **kwargs):
        """ Patch for ('statsmodels.datasets', 'get_rdataset') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(datasets, 'get_rdataset')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('statsmodels.datasets', 'get_rdataset')

            operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
            result = original(*args, **kwargs)
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(result.title, list(result.data.columns)),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))

            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [], function_call_result)
            new_result = function_call_result.function_result
            return new_result

        return execute_patched_func(original, execute_inspections, *args, **kwargs)


@gorilla.patches(api.OLS)
class StatsmodelsOlsPatching:
    """ Patches for statsmodel OLS"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *args, **kwargs):
        """ Patch for ('statsmodel.api', 'OLS') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals
        original = gorilla.get_original_attribute(api.OLS, '__init__')

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, *args, **kwargs)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, *args, **kwargs)

    @gorilla.name('fit')
    @gorilla.settings(allow_hit=True)
    def patched_fit(self, *args, **kwargs):
        """ Patch for ('statsmodel.api.OLS', 'fit') """
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(api.OLS, 'fit')
        function_info = FunctionInfo('statsmodel.api.OLS', 'fit')

        # Train data
        # pylint: disable=no-member
        _, train_data_node, train_data_result = add_train_data_node(self, self.data.exog, function_info)
        self.data.exog = train_data_result
        # pylint: disable=no-member
        _, train_labels_node, train_labels_result = add_train_label_node(self, self.data.endog, function_info)
        self.data.endog = train_labels_result

        # Estimator
        operator_context = OperatorContext(OperatorType.ESTIMATOR, function_info)
        # input_dfs = [data_backend_result.annotated_dfobject, label_backend_result.annotated_dfobject]
        result = original(self, *args, **kwargs)

        dag_node = DagNode(singleton.get_next_op_id(),
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Decision Tree", []),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code))
        function_call_result = FunctionCallResult(result)
        add_dag_node(dag_node, [train_data_node, train_labels_node], function_call_result)
        new_result = function_call_result.function_result
        return new_result
