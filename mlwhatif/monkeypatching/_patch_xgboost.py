"""
Monkey patching for sklearn
"""
# pylint: disable=too-many-lines
from functools import partial

import gorilla
import xgboost
from sklearn.metrics import accuracy_score

from mlwhatif.execution._pipeline_executor import singleton
from mlwhatif.execution._stat_tracking import capture_optimizer_info
from mlwhatif.instrumentation._dag_node import DagNode, BasicCodeLocation, DagNodeDetails, OptimizerInfo
from mlwhatif.instrumentation._operator_types import OperatorContext, FunctionInfo, OperatorType
from mlwhatif.monkeypatching._monkey_patching_utils import add_dag_node, \
    execute_patched_func_indirect_allowed, execute_patched_func_no_op_id, \
    get_optional_code_info_or_none, get_dag_node_for_id, add_train_data_node, \
    add_train_label_node, add_test_label_node, add_test_data_dag_node, FunctionCallResult
from mlwhatif.monkeypatching._patch_sklearn import call_info_singleton


@gorilla.patches(xgboost.XGBClassifier)
class XGBoostXGBClassifierPatching:
    """ Patches for sklearn LogisticRegression"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, objective="binary:logistic", use_label_encoder=False, mlinspect_caller_filename=None,
                        mlinspect_lineno=None, mlinspect_optional_code_reference=None,
                        mlinspect_optional_source_code=None, mlinspect_estimator_node_id=None, **kwargs):
        """ Patch for ('xgboost.sklearn', 'XGBClassifier') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals, too-many-arguments
        original = gorilla.get_original_attribute(xgboost.XGBClassifier, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_estimator_node_id = mlinspect_estimator_node_id

        self.mlinspect_non_data_func_args = {'objective': objective, 'use_label_encoder': use_label_encoder, **kwargs}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.name('fit')
    @gorilla.settings(allow_hit=True)
    def patched_fit(self, *args, **kwargs):
        """ Patch for ('xgboost.sklearn.XGBClassifier', 'fit') """
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(xgboost.XGBClassifier, 'fit')
        if not call_info_singleton.param_search_active:
            function_info = FunctionInfo('xgboost.sklearn', 'XGBClassifier')

            _, train_data_node, train_data_result = add_train_data_node(self, args[0], function_info)
            _, train_labels_node, train_labels_result = add_train_label_node(self, args[1], function_info)

            if call_info_singleton.make_grid_search_func is None:
                def processing_func(train_data, train_labels):
                    estimator = xgboost.XGBClassifier(**self.mlinspect_non_data_func_args)
                    fitted_estimator = estimator.fit(train_data, train_labels, *args[2:], **kwargs)
                    return fitted_estimator
                param_search_runtime = 0
                create_func = partial(xgboost.XGBClassifier, **self.mlinspect_non_data_func_args)
            else:
                def processing_func_with_grid_search(make_grid_search_func, train_data, train_labels):
                    estimator = make_grid_search_func(xgboost.XGBClassifier(
                        **self.mlinspect_non_data_func_args))
                    fitted_estimator = estimator.fit(train_data, train_labels, *args[2:], **kwargs)
                    return fitted_estimator
                processing_func = partial(processing_func_with_grid_search, call_info_singleton.make_grid_search_func)

                def create_func_with_grid_search(make_grid_search_func):
                    return make_grid_search_func(xgboost.XGBClassifier(**self.mlinspect_non_data_func_args))
                create_func = partial(create_func_with_grid_search, call_info_singleton.make_grid_search_func)

                call_info_singleton.make_grid_search_func = None
                param_search_runtime = call_info_singleton.param_search_duration
                call_info_singleton.param_search_duration = 0

            # Estimator
            operator_context = OperatorContext(OperatorType.ESTIMATOR, function_info)
            # input_dfs = [data_backend_result.annotated_dfobject, label_backend_result.annotated_dfobject]
            initial_func = partial(original, self, train_data_result, train_labels_result, *args[2:], **kwargs)
            optimizer_info, _ = capture_optimizer_info(initial_func, self, estimator_transformer_state=self)
            optimizer_info_with_search = OptimizerInfo(optimizer_info.runtime + param_search_runtime,
                                                       optimizer_info.shape, optimizer_info.memory)
            self.mlinspect_estimator_node_id = singleton.get_next_op_id()  # pylint: disable=attribute-defined-outside-init
            dag_node = DagNode(self.mlinspect_estimator_node_id,
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("XGB Classifier", [], optimizer_info_with_search),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code),
                               processing_func,
                               create_func)
            function_call_result = FunctionCallResult(None)
            add_dag_node(dag_node, [train_data_node, train_labels_node], function_call_result)
        else:
            original(self, *args, **kwargs)
        return self

    @gorilla.name('score')
    @gorilla.settings(allow_hit=True)
    def patched_score(self, *args, **kwargs):
        """ Patch for ('xgboost.sklearn.XGBClassifier', 'score') """

        # pylint: disable=no-method-argument
        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            if len(kwargs) != 0:
                raise Exception("TODO: Support other metrics in model.score calls!")

            function_info = FunctionInfo('xgboost.sklearn.XGBClassifier', 'score')
            # Test data
            _, test_data_node, test_data_result = add_test_data_dag_node(args[0],
                                                                         function_info,
                                                                         lineno,
                                                                         optional_code_reference,
                                                                         optional_source_code,
                                                                         caller_filename)

            # Test labels
            _, test_labels_node, test_labels_result = add_test_label_node(args[1],
                                                                          caller_filename,
                                                                          function_info,
                                                                          lineno,
                                                                          optional_code_reference,
                                                                          optional_source_code)

            def processing_func_predict(estimator, test_data):
                predictions = estimator.predict(test_data)
                return predictions

            def processing_func_score(predictions, test_labels):
                score = accuracy_score(test_labels, predictions)
                return score

            # input_dfs = [data_backend_result.annotated_dfobject, label_backend_result.annotated_dfobject]

            original_predict = gorilla.get_original_attribute(xgboost.XGBClassifier, 'predict')
            initial_func_predict = partial(original_predict, self, test_data_result)
            optimizer_info_predict, result_predict = capture_optimizer_info(initial_func_predict)
            operator_context_predict = OperatorContext(OperatorType.PREDICT, function_info)
            dag_node_predict = DagNode(singleton.get_next_op_id(),
                                       BasicCodeLocation(caller_filename, lineno),
                                       operator_context_predict,
                                       DagNodeDetails("XGB Classifier", [], optimizer_info_predict),
                                       get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                                       processing_func_predict)
            estimator_dag_node = get_dag_node_for_id(self.mlinspect_estimator_node_id)
            function_call_result = FunctionCallResult(result_predict)
            add_dag_node(dag_node_predict, [estimator_dag_node, test_data_node], function_call_result)

            initial_func_score = partial(processing_func_score, result_predict, test_labels_result)
            optimizer_info_score, result_score = capture_optimizer_info(initial_func_score)
            operator_context_score = OperatorContext(OperatorType.SCORE, function_info)
            dag_node_score = DagNode(singleton.get_next_op_id(),
                                     BasicCodeLocation(caller_filename, lineno),
                                     operator_context_score,
                                     DagNodeDetails("Accuracy", [], optimizer_info_score),
                                     get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                                     processing_func_score)
            function_call_result = FunctionCallResult(result_score)
            add_dag_node(dag_node_score, [dag_node_predict, test_labels_node],
                         function_call_result)
            return result_score

        if not call_info_singleton.param_search_active:
            new_result = execute_patched_func_indirect_allowed(execute_inspections)
        else:
            original = gorilla.get_original_attribute(xgboost.XGBClassifier, 'score')
            new_result = original(self, *args, **kwargs)
        return new_result

    @gorilla.name('predict')
    @gorilla.settings(allow_hit=True)
    def patched_predict(self, *args):
        """ Patch for ('xgboost.sklearn.XGBClassifier', 'predict') """

        # pylint: disable=no-method-argument
        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('xgboost.sklearn.XGBClassifier', 'predict')
            # Test data
            _, test_data_node, test_data_result = add_test_data_dag_node(args[0],
                                                                         function_info,
                                                                         lineno,
                                                                         optional_code_reference,
                                                                         optional_source_code,
                                                                         caller_filename)

            def processing_func_predict(estimator, test_data):
                predictions = estimator.predict(test_data)
                return predictions

            original_predict = gorilla.get_original_attribute(xgboost.XGBClassifier, 'predict')
            initial_func_predict = partial(original_predict, self, test_data_result)
            optimizer_info_predict, result_predict = capture_optimizer_info(initial_func_predict)
            operator_context_predict = OperatorContext(OperatorType.PREDICT, function_info)
            dag_node_predict = DagNode(singleton.get_next_op_id(),
                                       BasicCodeLocation(caller_filename, lineno),
                                       operator_context_predict,
                                       DagNodeDetails("XGB Classifier", [], optimizer_info_predict),
                                       get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                                       processing_func_predict)
            estimator_dag_node = get_dag_node_for_id(self.mlinspect_estimator_node_id)
            function_call_result = FunctionCallResult(result_predict)
            add_dag_node(dag_node_predict, [estimator_dag_node, test_data_node], function_call_result)
            new_result = function_call_result.function_result
            return new_result

        if not call_info_singleton.param_search_active:
            new_result = execute_patched_func_indirect_allowed(execute_inspections)
        else:
            original = gorilla.get_original_attribute(xgboost.XGBClassifier, 'predict')
            new_result = original(self, *args)
        return new_result
