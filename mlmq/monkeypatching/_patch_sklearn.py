"""
Monkey patching for sklearn
"""
# pylint: disable=too-many-lines
import dataclasses
import warnings
from functools import partial
from typing import Callable

import gorilla
import numpy
import pandas
import scipy
import tensorflow
from joblib import Parallel, delayed
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn import preprocessing, compose, tree, impute, linear_model, model_selection, metrics, dummy, decomposition, \
    svm, pipeline
from sklearn.feature_extraction import text
from sklearn.linear_model._stochastic_gradient import DEFAULT_EPSILON
from sklearn.metrics import accuracy_score
from sklearn.pipeline import _fit_transform_one, _transform_one
from tensorflow.keras.wrappers import scikit_learn as keras_sklearn_external  # pylint: disable=no-name-in-module
from tensorflow.python.keras.wrappers import scikit_learn as keras_sklearn_internal  # pylint: disable=no-name-in-module

from mlmq.execution._stat_tracking import capture_optimizer_info, get_df_shape, get_df_memory
from mlmq.instrumentation._operator_types import OperatorContext, FunctionInfo, OperatorType
from mlmq.instrumentation._dag_node import DagNode, BasicCodeLocation, DagNodeDetails, CodeReference, OptimizerInfo
from mlmq.execution._pipeline_executor import singleton
from mlmq.monkeypatching._mlinspect_ndarray import MlinspectNdarray
from mlmq.monkeypatching._monkey_patching_utils import execute_patched_func, add_dag_node, \
    execute_patched_func_indirect_allowed, get_input_info, execute_patched_func_no_op_id, \
    get_optional_code_info_or_none, get_dag_node_for_id, add_train_data_node, \
    add_train_label_node, add_test_label_node, add_test_data_dag_node, FunctionCallResult, \
    wrap_in_mlinspect_array_if_necessary


@gorilla.patches(preprocessing)
class SklearnPreprocessingPatching:
    """ Patches for sklearn """

    # pylint: disable=too-few-public-methods

    @gorilla.name('label_binarize')
    @gorilla.settings(allow_hit=True)
    def patched_label_binarize(*args, **kwargs):
        """ Patch for ('sklearn.preprocessing._label', 'label_binarize') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing, 'label_binarize')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('sklearn.preprocessing._label', 'label_binarize')
            input_info = get_input_info(args[0], caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)

            operator_context = OperatorContext(OperatorType.PROJECTION_MODIFY, function_info)
            initial_func = partial(original, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
            optimizer_info, result = capture_optimizer_info(initial_func)
            processing_func = lambda df: original(df, *args[1:], **kwargs)

            classes = kwargs['classes']
            description = "label_binarize, classes: {}".format(classes)
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, ["array"], optimizer_info),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                               processing_func)
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result

            return new_result

        return execute_patched_func(original, execute_inspections, *args, **kwargs)


@dataclasses.dataclass
class TrainTestSplitResult:
    """
    Additional info about the DAG node
    """
    train: any or None = None
    test: any or None = None


@gorilla.patches(model_selection)
class SklearnModelSelectionPatching:
    """ Patches for sklearn """

    # pylint: disable=too-few-public-methods

    @gorilla.name('train_test_split')
    @gorilla.settings(allow_hit=True)
    def patched_train_test_split(*args, **kwargs):
        """ Patch for ('sklearn.model_selection._split', 'train_test_split') """
        # pylint: disable=no-method-argument,too-many-locals
        original = gorilla.get_original_attribute(model_selection, 'train_test_split')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('sklearn.model_selection._split', 'train_test_split')
            input_info = get_input_info(args[0], caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)

            operator_context = OperatorContext(OperatorType.TRAIN_TEST_SPLIT, function_info)
            initial_func = partial(original, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
            optimizer_info, result = capture_optimizer_info(initial_func)

            def train_test_split_and_wrapping(df_object):
                split_result = original(df_object, *args[1:], **kwargs)
                return TrainTestSplitResult(*split_result)

            def train_test_split_train(split_result):
                return split_result.train

            def train_test_split_test(split_result):
                return split_result.test

            columns = list(result[0].columns)
            main_dag_node = DagNode(op_id,
                                    BasicCodeLocation(caller_filename, lineno),
                                    operator_context,
                                    DagNodeDetails(None, columns, optimizer_info),
                                    get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                                    train_test_split_and_wrapping)
            add_dag_node(main_dag_node, [input_info.dag_node], FunctionCallResult(None))

            description = "(Train Data)"
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, columns, OptimizerInfo(0, get_df_shape(result[0]),
                                                                                  get_df_memory(result[0]))),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                               train_test_split_train)

            train_function_call_result = FunctionCallResult(result[0])
            add_dag_node(dag_node, [main_dag_node], train_function_call_result)
            new_train_result = train_function_call_result.function_result

            description = "(Test Data)"
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, columns, OptimizerInfo(0, get_df_shape(result[1]),
                                                                                  get_df_memory(result[1]))),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                               train_test_split_test)

            test_function_call_result = FunctionCallResult(result[1])
            add_dag_node(dag_node, [main_dag_node], test_function_call_result)
            new_test_result = test_function_call_result.function_result

            new_result = (new_train_result, new_test_result)

            return new_result

        return execute_patched_func(original, execute_inspections, *args, **kwargs)


class SklearnCallInfo:
    """ Contains info like lineno from the current Transformer so indirect utility function calls can access it """
    # pylint: disable=too-few-public-methods,too-many-instance-attributes

    transformer_filename: str or None = None
    transformer_lineno: int or None = None
    transformer_function_info: FunctionInfo or None = None
    transformer_optional_code_reference: CodeReference or None = None
    transformer_optional_source_code: str or None = None
    column_transformer_active: bool = False
    score_active: bool = False
    param_search_active: bool = False
    make_grid_search_func: Callable or None = None
    param_search_duration: int = 0


call_info_singleton = SklearnCallInfo()


@gorilla.patches(model_selection.GridSearchCV)
class SklearnGridSearchCVPatching:
    """ Patches for sklearn GridSearchCV"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, estimator, param_grid, *, scoring=None,
                        n_jobs=None, iid='deprecated', refit=True, cv=None,
                        verbose=0, pre_dispatch='2*n_jobs',
                        error_score=numpy.nan, return_train_score=False):
        """ Patch for ('sklearn.compose.model_selection._search', 'GridSearchCV') """
        # pylint: disable=no-method-argument,invalid-name
        original = gorilla.get_original_attribute(model_selection.GridSearchCV, '__init__')

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=attribute-defined-outside-init
            supported_estimators = (tree.DecisionTreeClassifier, linear_model.SGDClassifier,
                                    linear_model.LogisticRegression, keras_sklearn_external.KerasClassifier)
            if not isinstance(estimator, supported_estimators):  # pylint: disable=no-member
                raise NotImplementedError(f"TODO: Estimator is an instance of "
                                          f"{type(self.estimator)}, "  # pylint: disable=no-member
                                          f"which is not supported yet!")

            original(self, estimator, param_grid, scoring=scoring, n_jobs=n_jobs,
                     iid=iid, refit=refit, cv=cv, verbose=verbose, pre_dispatch=pre_dispatch,
                     error_score=error_score, return_train_score=return_train_score)

            self.mlinspect_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

            make_grid_search_kwargs = {'param_grid': param_grid, 'scoring': scoring, 'n_jobs': n_jobs,
                                       'iid': iid, 'refit': refit, 'cv': cv, 'verbose': verbose,
                                       'pre_dispatch': pre_dispatch, 'error_score': error_score,
                                       'return_train_score': return_train_score}

            def make_grid_search(grid_search_kwargs, estimator_to_wrap):
                return model_selection.GridSearchCV(estimator=estimator_to_wrap, **grid_search_kwargs)

            call_info_singleton.make_grid_search_func = partial(make_grid_search, make_grid_search_kwargs)

        return execute_patched_func_indirect_allowed(execute_inspections)

    @gorilla.name('_run_search')
    @gorilla.settings(allow_hit=True)
    def patched__run_search(self, *args, **kwargs):
        """ Patch for ('sklearn.compose.model_selection._search', 'GridSearchCV') """
        # pylint: disable=no-method-argument
        call_info_singleton.transformer_filename = self.mlinspect_filename
        call_info_singleton.transformer_lineno = self.mlinspect_lineno
        call_info_singleton.transformer_function_info = FunctionInfo(
            'sklearn.compose.model_selection._search.GridSearchCV', 'fit_transform')
        call_info_singleton.transformer_optional_code_reference = self.mlinspect_optional_code_reference
        call_info_singleton.transformer_optional_source_code = self.mlinspect_optional_source_code

        call_info_singleton.param_search_active = True
        original = gorilla.get_original_attribute(model_selection.GridSearchCV, '_run_search')
        initial_func = partial(original, self, *args, **kwargs)
        optimizer_info, result = capture_optimizer_info(initial_func, self)
        call_info_singleton.param_search_active = False
        call_info_singleton.param_search_duration = optimizer_info.runtime

        return result


@gorilla.patches(compose.ColumnTransformer)
class SklearnComposePatching:
    """ Patches for sklearn ColumnTransformer"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self,
                        transformers, *,
                        remainder='drop',
                        sparse_threshold=0.3,
                        n_jobs=None,
                        transformer_weights=None,
                        verbose=False):
        """ Patch for ('sklearn.compose._column_transformer', 'ColumnTransformer') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(compose.ColumnTransformer, '__init__')

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=attribute-defined-outside-init
            original(self, transformers, remainder=remainder, sparse_threshold=sparse_threshold, n_jobs=n_jobs,
                     transformer_weights=transformer_weights, verbose=verbose)

            self.mlinspect_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

            self.mlinspect_non_data_func_args = {'transformers': transformers, 'remainder': remainder,
                                                 'sparse_threshold': sparse_threshold, 'n_jobs': n_jobs,
                                                 'transformer_weights': transformer_weights, 'verbose': verbose
                                                 }

        return execute_patched_func_indirect_allowed(execute_inspections)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.compose._column_transformer', 'ColumnTransformer') """
        # pylint: disable=no-method-argument
        call_info_singleton.transformer_filename = self.mlinspect_filename
        call_info_singleton.transformer_lineno = self.mlinspect_lineno
        call_info_singleton.transformer_function_info = FunctionInfo('sklearn.compose._column_transformer',
                                                                     'ColumnTransformer')
        call_info_singleton.transformer_optional_code_reference = self.mlinspect_optional_code_reference
        call_info_singleton.transformer_optional_source_code = self.mlinspect_optional_source_code

        call_info_singleton.column_transformer_active = True
        original = gorilla.get_original_attribute(compose.ColumnTransformer, 'fit_transform')
        result = original(self, *args, **kwargs)
        call_info_singleton.column_transformer_active = False

        return result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.compose._column_transformer', 'ColumnTransformer') """
        # pylint: disable=no-method-argument
        call_info_singleton.transformer_filename = self.mlinspect_filename
        call_info_singleton.transformer_lineno = self.mlinspect_lineno
        call_info_singleton.transformer_function_info = FunctionInfo('sklearn.compose._column_transformer',
                                                                     'ColumnTransformer')
        call_info_singleton.transformer_optional_code_reference = self.mlinspect_optional_code_reference
        call_info_singleton.transformer_optional_source_code = self.mlinspect_optional_source_code

        call_info_singleton.column_transformer_active = True
        original = gorilla.get_original_attribute(compose.ColumnTransformer, 'transform')
        result = original(self, *args, **kwargs)
        call_info_singleton.column_transformer_active = False

        return result

    @gorilla.name('_hstack')
    @gorilla.settings(allow_hit=True)
    def patched_hstack(self, *args, **kwargs):
        """ Patch for ('sklearn.compose._column_transformer', 'ColumnTransformer') """
        # pylint: disable=no-method-argument, unused-argument, too-many-locals
        original = gorilla.get_original_attribute(compose.ColumnTransformer, '_hstack')

        if not call_info_singleton.column_transformer_active:
            return original(self, *args, **kwargs)

        input_tuple = args[0]
        function_info = FunctionInfo('sklearn.compose._column_transformer', 'ColumnTransformer')
        input_infos = []
        for input_df_obj in input_tuple:
            input_info = get_input_info(input_df_obj, self.mlinspect_filename, self.mlinspect_lineno, function_info,
                                        self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)
            input_infos.append(input_info)

        operator_context = OperatorContext(OperatorType.CONCATENATION, function_info)
        # input_annotated_dfs = [input_info.annotated_dfobject for input_info in input_infos]
        # No input_infos copy needed because it's only a selection and the rows not being removed don't change
        initial_func = partial(original, self, *args, **kwargs)
        optimizer_info, result = capture_optimizer_info(initial_func)

        def processing_func(*input_dfs):
            transformer = compose.ColumnTransformer(**self.mlinspect_non_data_func_args)
            # This is code out of the ColumnTransformer, maybe we can find a cleaner concat solution in the future
            if any(scipy.sparse.issparse(df) for df in input_dfs):
                nnz = sum(df.nnz if scipy.sparse.issparse(df) else df.size for df in input_dfs)
                total = sum(df.shape[0] * df.shape[1] if scipy.sparse.issparse(df)
                            else df.size for df in input_dfs)
                density = nnz / total
                transformer.sparse_output_ = density < self.sparse_threshold  # pylint: disable=no-member
            else:
                transformer.sparse_output_ = False
            transformed_data = transformer._hstack(input_dfs)  # pylint: disable=protected-access
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            # Not sure if this might be necessary at some point
            # transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        dag_node = DagNode(singleton.get_next_op_id(),
                           BasicCodeLocation(self.mlinspect_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails(None, ['array'], optimizer_info),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code),
                           processing_func)
        input_dag_nodes = [input_info.dag_node for input_info in input_infos]
        function_call_result = FunctionCallResult(result)
        add_dag_node(dag_node, input_dag_nodes, function_call_result)
        new_result = function_call_result.function_result

        return new_result


@gorilla.patches(preprocessing.StandardScaler)
class SklearnStandardScalerPatching:
    """ Patches for sklearn StandardScaler"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, copy=True, with_mean=True, with_std=True,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_fit_transform_active=False, mlinspect_transformer_node_id=None):
        """ Patch for ('sklearn.preprocessing._data', 'StandardScaler') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(preprocessing.StandardScaler, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_transformer_node_id = mlinspect_transformer_node_id

        self.mlinspect_non_data_func_args = {'copy': copy, 'with_mean': with_mean, 'with_std': with_std}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, copy=copy, with_mean=with_mean, with_std=with_std)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._data.StandardScaler', 'fit_transform') """
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = True  # pylint: disable=attribute-defined-outside-init
        original = gorilla.get_original_attribute(preprocessing.StandardScaler, 'fit_transform')
        function_info = FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        def processing_func(input_df):
            transformer = preprocessing.StandardScaler(**self.mlinspect_non_data_func_args)
            transformed_data = transformer.fit_transform(input_df, *args[1:], **kwargs)
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
        optimizer_info, result = capture_optimizer_info(initial_func, estimator_transformer_state=self)
        dag_node_id = singleton.get_next_op_id()
        self.mlinspect_transformer_node_id = dag_node_id  # pylint: disable=attribute-defined-outside-init
        dag_node = DagNode(dag_node_id,
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Standard Scaler: fit_transform", ['array'], optimizer_info),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code),
                           processing_func)

        function_call_result = FunctionCallResult(result)
        add_dag_node(dag_node, [input_info.dag_node], function_call_result)
        new_result = function_call_result.function_result
        assert isinstance(new_result, MlinspectNdarray)
        self.mlinspect_fit_transform_active = False  # pylint: disable=attribute-defined-outside-init
        return new_result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._data.StandardScaler', 'transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing.StandardScaler, 'transform')
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')
            input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                        self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

            def processing_func(fit_data, input_df):
                transformer = fit_data._mlinspect_annotation  # pylint: disable=protected-access
                transformed_data = transformer.transform(input_df, *args[1:], **kwargs)
                return transformed_data

            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
            optimizer_info, result = capture_optimizer_info(initial_func)
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("Standard Scaler: transform", ['array'], optimizer_info),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code),
                               processing_func)

            function_call_result = FunctionCallResult(result)
            transformer_dag_node = get_dag_node_for_id(self.mlinspect_transformer_node_id)
            add_dag_node(dag_node, [transformer_dag_node, input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result
            assert isinstance(new_result, MlinspectNdarray)
        else:
            new_result = original(self, *args, **kwargs)
        return new_result


@gorilla.patches(preprocessing.RobustScaler)
class SklearnRobustScalerPatching:
    """ Patches for sklearn RobustScaler"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, with_centering=True, with_scaling=True,
                        quantile_range=(25.0, 75.0), copy=True,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_fit_transform_active=False, mlinspect_transformer_node_id=None):
        """ Patch for ('sklearn.preprocessing._data', 'RobustScaler') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(preprocessing.RobustScaler, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_transformer_node_id = mlinspect_transformer_node_id

        self.mlinspect_non_data_func_args = {'with_centering': with_centering, 'with_scaling': with_scaling,
                                             'quantile_range': quantile_range, 'copy': copy}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._data.RobustScaler', 'fit_transform') """
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = True  # pylint: disable=attribute-defined-outside-init
        original = gorilla.get_original_attribute(preprocessing.RobustScaler, 'fit_transform')
        function_info = FunctionInfo('sklearn.preprocessing._data', 'RobustScaler')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        def processing_func(input_df):
            transformer = preprocessing.RobustScaler(**self.mlinspect_non_data_func_args)
            transformed_data = transformer.fit_transform(input_df, *args[1:], **kwargs)
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
        optimizer_info, result = capture_optimizer_info(initial_func, estimator_transformer_state=self)
        dag_node_id = singleton.get_next_op_id()
        self.mlinspect_transformer_node_id = dag_node_id  # pylint: disable=attribute-defined-outside-init
        dag_node = DagNode(dag_node_id,
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Robust Scaler: fit_transform", ['array'], optimizer_info),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code),
                           processing_func)

        function_call_result = FunctionCallResult(result)
        add_dag_node(dag_node, [input_info.dag_node], function_call_result)
        new_result = function_call_result.function_result
        assert isinstance(new_result, MlinspectNdarray)
        self.mlinspect_fit_transform_active = False  # pylint: disable=attribute-defined-outside-init
        return new_result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._data.RobustScaler', 'transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing.RobustScaler, 'transform')
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo('sklearn.preprocessing._data', 'RobustScaler')
            input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                        self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

            def processing_func(fit_data, input_df):
                transformer = fit_data._mlinspect_annotation  # pylint: disable=protected-access
                transformed_data = transformer.transform(input_df, *args[1:], **kwargs)
                return transformed_data

            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
            optimizer_info, result = capture_optimizer_info(initial_func)
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("Robust Scaler: transform", ['array'], optimizer_info),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code),
                               processing_func)

            function_call_result = FunctionCallResult(result)
            transformer_dag_node = get_dag_node_for_id(self.mlinspect_transformer_node_id)
            add_dag_node(dag_node, [transformer_dag_node, input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result
            assert isinstance(new_result, MlinspectNdarray)
        else:
            new_result = original(self, *args, **kwargs)
        return new_result


@gorilla.patches(text.CountVectorizer)
class SklearnCountVectorizerPatching:
    """ Patches for sklearn RobustScaler"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, input='content', encoding='utf-8',
                        decode_error='strict', strip_accents=None,
                        lowercase=True, preprocessor=None, tokenizer=None,
                        stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                        ngram_range=(1, 1), analyzer='word',
                        max_df=1.0, min_df=1, max_features=None,
                        vocabulary=None, binary=False, dtype=numpy.int64,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_fit_transform_active=False, mlinspect_transformer_node_id=None):
        """ Patch for ('sklearn.feature_extraction.text', 'CountVectorizer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init,redefined-builtin,too-many-locals
        original = gorilla.get_original_attribute(text.CountVectorizer, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_transformer_node_id = mlinspect_transformer_node_id

        self.mlinspect_non_data_func_args = {'input': input, 'encoding': encoding,
                                             'decode_error': decode_error, 'strip_accents': strip_accents,
                                             'lowercase': lowercase, 'preprocessor': preprocessor,
                                             'tokenizer': tokenizer, 'stop_words': stop_words,
                                             'token_pattern': token_pattern, 'ngram_range': ngram_range,
                                             'analyzer': analyzer, 'max_df': max_df, 'min_df': min_df,
                                             'max_features': max_features, 'vocabulary': vocabulary,
                                             'binary': binary, 'dtype': dtype}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.feature_extraction.text.CountVectorizer', 'fit_transform') """
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = True  # pylint: disable=attribute-defined-outside-init
        original = gorilla.get_original_attribute(text.CountVectorizer, 'fit_transform')
        function_info = FunctionInfo('sklearn.feature_extraction.text', 'CountVectorizer')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        def processing_func(input_df):
            transformer = text.CountVectorizer(**self.mlinspect_non_data_func_args)
            transformed_data = transformer.fit_transform(input_df, *args[1:], **kwargs)
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
        optimizer_info, result = capture_optimizer_info(initial_func, estimator_transformer_state=self)
        dag_node_id = singleton.get_next_op_id()
        self.mlinspect_transformer_node_id = dag_node_id  # pylint: disable=attribute-defined-outside-init
        dag_node = DagNode(dag_node_id,
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Count Vectorizer: fit_transform", ['array'], optimizer_info),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code),
                           processing_func)

        function_call_result = FunctionCallResult(result)
        add_dag_node(dag_node, [input_info.dag_node], function_call_result)
        new_result = function_call_result.function_result
        assert isinstance(new_result, (MlinspectNdarray, csr_matrix))
        self.mlinspect_fit_transform_active = False  # pylint: disable=attribute-defined-outside-init
        return new_result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.feature_extraction.text.CountVectorizer', 'transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(text.CountVectorizer, 'transform')
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo('sklearn.feature_extraction.text', 'CountVectorizer')
            input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                        self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

            def processing_func(fit_data, input_df):
                transformer = fit_data._mlinspect_annotation  # pylint: disable=protected-access
                transformed_data = transformer.transform(input_df, *args[1:], **kwargs)
                return transformed_data

            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
            optimizer_info, result = capture_optimizer_info(initial_func)
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("Count Vectorizer: transform", ['array'], optimizer_info),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code),
                               processing_func)

            function_call_result = FunctionCallResult(result)
            transformer_dag_node = get_dag_node_for_id(self.mlinspect_transformer_node_id)
            add_dag_node(dag_node, [transformer_dag_node, input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result
            assert isinstance(new_result, (MlinspectNdarray, csr_matrix))
        else:
            new_result = original(self, *args, **kwargs)
        return new_result


@gorilla.patches(text.TfidfTransformer)
class SklearnTfidfTransformerPatching:
    """ Patches for sklearn RobustScaler"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_fit_transform_active=False, mlinspect_transformer_node_id=None):
        """ Patch for ('sklearn.feature_extraction.text', 'TfidfTransformer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(text.TfidfTransformer, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_transformer_node_id = mlinspect_transformer_node_id

        self.mlinspect_non_data_func_args = {'norm': norm, 'use_idf': use_idf, 'smooth_idf': smooth_idf,
                                             'sublinear_tf': sublinear_tf}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.feature_extraction.text.TfidfTransformer', 'fit_transform') """
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = True  # pylint: disable=attribute-defined-outside-init
        original = gorilla.get_original_attribute(text.TfidfTransformer, 'fit_transform')
        function_info = FunctionInfo('sklearn.feature_extraction.text', 'TfidfTransformer')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        def processing_func(input_df):
            transformer = text.TfidfTransformer(**self.mlinspect_non_data_func_args)
            transformed_data = transformer.fit_transform(input_df, *args[1:], **kwargs)
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
        optimizer_info, result = capture_optimizer_info(initial_func, estimator_transformer_state=self)
        dag_node_id = singleton.get_next_op_id()
        self.mlinspect_transformer_node_id = dag_node_id  # pylint: disable=attribute-defined-outside-init
        dag_node = DagNode(dag_node_id,
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Tfidf Transformer: fit_transform", ['array'], optimizer_info),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code),
                           processing_func)

        function_call_result = FunctionCallResult(result)
        add_dag_node(dag_node, [input_info.dag_node], function_call_result)
        new_result = function_call_result.function_result
        assert isinstance(new_result, (MlinspectNdarray, csr_matrix))
        self.mlinspect_fit_transform_active = False  # pylint: disable=attribute-defined-outside-init
        return new_result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.feature_extraction.text.TfidfTransformer', 'transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(text.TfidfTransformer, 'transform')
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo('sklearn.feature_extraction.text', 'TfidfTransformer')
            input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                        self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

            def processing_func(fit_data, input_df):
                transformer = fit_data._mlinspect_annotation  # pylint: disable=protected-access
                transformed_data = transformer.transform(input_df, *args[1:], **kwargs)
                return transformed_data

            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
            optimizer_info, result = capture_optimizer_info(initial_func)
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("Tfidf Transformer: transform", ['array'], optimizer_info),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code),
                               processing_func)

            function_call_result = FunctionCallResult(result)
            transformer_dag_node = get_dag_node_for_id(self.mlinspect_transformer_node_id)
            add_dag_node(dag_node, [transformer_dag_node, input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result
            assert isinstance(new_result, (MlinspectNdarray, csr_matrix))
        else:
            new_result = original(self, *args, **kwargs)
        return new_result


@gorilla.patches(decomposition.TruncatedSVD)
class SklearnTruncatedSVDPatching:
    """ Patches for sklearn TruncatedSVD"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, n_components=2, *, algorithm="randomized", n_iter=5,
                        random_state=None, tol=0., mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_fit_transform_active=False, mlinspect_transformer_node_id=None):
        """ Patch for ('sklearn.decomposition._truncated_svd', 'TruncatedSVD') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(decomposition.TruncatedSVD, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_transformer_node_id = mlinspect_transformer_node_id

        self.mlinspect_non_data_func_args = {'n_components': n_components, 'algorithm': algorithm, 'n_iter': n_iter,
                                             'random_state': random_state, 'tol': tol}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.decomposition._truncated_svd.TruncatedSVD', 'fit_transform') """
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = True  # pylint: disable=attribute-defined-outside-init
        original = gorilla.get_original_attribute(decomposition.TruncatedSVD, 'fit_transform')
        function_info = FunctionInfo('sklearn.decomposition._truncated_svd', 'TruncatedSVD')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        def processing_func(input_df):
            transformer = decomposition.TruncatedSVD(**self.mlinspect_non_data_func_args)
            transformed_data = transformer.fit_transform(input_df, *args[1:], **kwargs)
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
        optimizer_info, result = capture_optimizer_info(initial_func, estimator_transformer_state=self)
        dag_node_id = singleton.get_next_op_id()
        self.mlinspect_transformer_node_id = dag_node_id  # pylint: disable=attribute-defined-outside-init
        dag_node = DagNode(dag_node_id,
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Truncated SVD: fit_transform", ['array'], optimizer_info),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code),
                           processing_func)

        function_call_result = FunctionCallResult(result)
        add_dag_node(dag_node, [input_info.dag_node], function_call_result)
        new_result = function_call_result.function_result
        assert isinstance(new_result, (MlinspectNdarray, csr_matrix))
        self.mlinspect_fit_transform_active = False  # pylint: disable=attribute-defined-outside-init
        return new_result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.decomposition._truncated_svd.TruncatedSVD', 'transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(decomposition.TruncatedSVD, 'transform')
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo('sklearn.decomposition._truncated_svd', 'TruncatedSVD')
            input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                        self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

            def processing_func(fit_data, input_df):
                transformer = fit_data._mlinspect_annotation  # pylint: disable=protected-access
                transformed_data = transformer.transform(input_df, *args[1:], **kwargs)
                return transformed_data

            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
            optimizer_info, result = capture_optimizer_info(initial_func)
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("Truncated SVD: transform", ['array'], optimizer_info),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code),
                               processing_func)

            function_call_result = FunctionCallResult(result)
            transformer_dag_node = get_dag_node_for_id(self.mlinspect_transformer_node_id)
            add_dag_node(dag_node, [transformer_dag_node, input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result
            assert isinstance(new_result, (MlinspectNdarray, csr_matrix))
        else:
            new_result = original(self, *args, **kwargs)
        return new_result


@gorilla.patches(decomposition.PCA)
class SklearnPCAPatching:
    """ Patches for sklearn PCA"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, n_components=None, *, copy=True, whiten=False,
                        svd_solver='auto', tol=0.0, iterated_power='auto',
                        random_state=None,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_fit_transform_active=False):
        """ Patch for ('sklearn.decomposition._pca', 'PCA') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(decomposition.PCA, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active

        self.mlinspect_non_data_func_args = {'n_components': n_components, 'copy': copy, 'whiten': whiten,
                                             'svd_solver': svd_solver, 'tol': tol, 'iterated_power': iterated_power,
                                             'random_state': random_state}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.decomposition._pca.PCA', 'fit_transform') """
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = True  # pylint: disable=attribute-defined-outside-init
        original = gorilla.get_original_attribute(decomposition.PCA, 'fit_transform')
        function_info = FunctionInfo('sklearn.decomposition._pca', 'PCA')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        def processing_func(input_df):
            transformer = decomposition.PCA(**self.mlinspect_non_data_func_args)
            transformed_data = transformer.fit_transform(input_df, *args[1:], **kwargs)
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
        optimizer_info, result = capture_optimizer_info(initial_func, estimator_transformer_state=self)
        dag_node_id = singleton.get_next_op_id()
        self.mlinspect_transformer_node_id = dag_node_id  # pylint: disable=attribute-defined-outside-init
        dag_node = DagNode(dag_node_id,
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("PCA: fit_transform", ['array'], optimizer_info),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code),
                           processing_func)

        function_call_result = FunctionCallResult(result)
        add_dag_node(dag_node, [input_info.dag_node], function_call_result)
        new_result = function_call_result.function_result
        assert isinstance(new_result, MlinspectNdarray)
        self.mlinspect_fit_transform_active = False  # pylint: disable=attribute-defined-outside-init
        return new_result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.decomposition._pca.PCA', 'transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(decomposition.PCA, 'transform')
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo('sklearn.decomposition._pca', 'PCA')
            input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                        self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

            def processing_func(fit_data, input_df):
                transformer = fit_data._mlinspect_annotation  # pylint: disable=protected-access
                transformed_data = transformer.transform(input_df, *args[1:], **kwargs)
                return transformed_data

            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
            optimizer_info, result = capture_optimizer_info(initial_func)
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("PCA: transform", ['array'], optimizer_info),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code),
                               processing_func)

            function_call_result = FunctionCallResult(result)
            transformer_dag_node = get_dag_node_for_id(self.mlinspect_transformer_node_id)
            add_dag_node(dag_node, [transformer_dag_node, input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result
            assert isinstance(new_result, MlinspectNdarray)
        else:
            new_result = original(self, *args, **kwargs)
        return new_result


@gorilla.patches(pipeline.FeatureUnion)
class SklearnFeatureUnionPatching:
    """ Patches for sklearn StandardScaler"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, transformer_list, *, n_jobs=None,
                        transformer_weights=None, verbose=False,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_fit_transform_active=False, mlinspect_transformer_node_id=None):
        """ Patch for ('sklearn.pipeline', 'FeatureUnion') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(pipeline.FeatureUnion, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_transformer_node_id = mlinspect_transformer_node_id

        self.mlinspect_non_data_func_args = {'transformer_list': transformer_list, 'n_jobs': n_jobs,
                                             'transformer_weights': transformer_weights, 'verbose': verbose}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, X, y=None, **fit_params):
        """ Patch for ('sklearn.pipeline.FeatureUnion', 'fit_transform') """
        # pylint: disable=no-method-argument, invalid-name,too-many-locals,invalid-name
        results = self._parallel_func(X, y, fit_params, _fit_transform_one)  # pylint: disable=no-member
        if not results:
            # All transformers are None
            raise Exception("TODO: Implement support for FeatureUnion without transformers")
            # return numpy.zeros((X.shape[0], 0))

        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)  # pylint: disable=no-member

        self.mlinspect_fit_transform_active = True  # pylint: disable=attribute-defined-outside-init
        function_info = FunctionInfo('sklearn.pipeline', 'FeatureUnion')
        input_infos = []
        for input_df_obj in Xs:
            input_info = get_input_info(input_df_obj, self.mlinspect_caller_filename, self.mlinspect_lineno,
                                        function_info, self.mlinspect_optional_code_reference,
                                        self.mlinspect_optional_source_code)
            input_infos.append(input_info)

        def processing_func(*input_dfs):
            # Rest of orignal fit_transform
            if any(sparse.issparse(f) for f in input_dfs):
                hstack_result = sparse.hstack(input_dfs).tocsr()
            else:
                hstack_result = numpy.hstack(input_dfs)
            transformed_data = wrap_in_mlinspect_array_if_necessary(hstack_result)
            return transformed_data

        operator_context = OperatorContext(OperatorType.CONCATENATION, function_info)
        initial_func = partial(processing_func, *Xs)
        optimizer_info, result = capture_optimizer_info(initial_func, estimator_transformer_state=self)
        dag_node_id = singleton.get_next_op_id()
        self.mlinspect_transformer_node_id = dag_node_id  # pylint: disable=attribute-defined-outside-init
        dag_node = DagNode(dag_node_id,
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Feature Union", ['array'], optimizer_info),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code),
                           processing_func)

        function_call_result = FunctionCallResult(result)
        input_dag_nodes = [input_info.dag_node for input_info in input_infos]
        add_dag_node(dag_node, input_dag_nodes, function_call_result)
        new_result = function_call_result.function_result
        assert isinstance(new_result, (MlinspectNdarray, csr_matrix))
        self.mlinspect_fit_transform_active = False  # pylint: disable=attribute-defined-outside-init
        return new_result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, X):
        """ Patch for ('sklearn.pipeline.FeatureUnion', 'transform') """
        # pylint: disable=no-method-argument,invalid-name,too-many-locals,no-member
        original = gorilla.get_original_attribute(pipeline.FeatureUnion, 'transform')
        if not self.mlinspect_fit_transform_active:
            # First part up to concat of the original transform
            for _, t in self.transformer_list:
                # TODO: Remove in 0.24 when None is removed
                if t is None:
                    warnings.warn("Using None as a transformer is deprecated "
                                  "in version 0.22 and will be removed in "
                                  "version 0.24. Please use 'drop' instead.",
                                  FutureWarning)
                    continue
            Xs = Parallel(n_jobs=self.n_jobs)(
                delayed(_transform_one)(trans, X, None, weight)
                for name, trans, weight in self._iter())
            if not Xs:
                # All transformers are None
                raise Exception("TODO: Implement support for FeatureUnion without transformers")
                # return numpy.zeros((X.shape[0], 0))

            function_info = FunctionInfo('sklearn.pipeline', 'FeatureUnion')
            input_infos = []
            for input_df_obj in Xs:
                input_info = get_input_info(input_df_obj, self.mlinspect_caller_filename, self.mlinspect_lineno,
                                            function_info, self.mlinspect_optional_code_reference,
                                            self.mlinspect_optional_source_code)
                input_infos.append(input_info)

            def processing_func(*input_dfs):
                if any(sparse.issparse(f) for f in input_dfs):
                    hstack_result = sparse.hstack(input_dfs).tocsr()
                else:
                    hstack_result = numpy.hstack(input_dfs)
                transformed_data = wrap_in_mlinspect_array_if_necessary(hstack_result)
                return transformed_data

            operator_context = OperatorContext(OperatorType.CONCATENATION, function_info)
            initial_func = partial(processing_func, *Xs)
            optimizer_info, result = capture_optimizer_info(initial_func)
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("Feature Union", ['array'], optimizer_info),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code),
                               processing_func)

            function_call_result = FunctionCallResult(result)
            input_dag_nodes = [input_info.dag_node for input_info in input_infos]
            add_dag_node(dag_node, input_dag_nodes, function_call_result)
            new_result = function_call_result.function_result
            assert isinstance(new_result, (MlinspectNdarray, csr_matrix))
        else:
            new_result = original(self, X)
        return new_result


@gorilla.patches(text.HashingVectorizer)
class SklearnHasingVectorizerPatching:
    """ Patches for sklearn StandardScaler"""

    # pylint: disable=too-few-public-methods, redefined-builtin, too-many-locals

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, input='content', encoding='utf-8', decode_error='strict', strip_accents=None,
                        lowercase=True, preprocessor=None, tokenizer=None, stop_words=None,
                        token_pattern=r"(?u)\b\w\w+\b", ngram_range=(1, 1), analyzer='word', n_features=(2 ** 20),
                        binary=False, norm='l2', alternate_sign=True, dtype=numpy.float64,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_fit_transform_active=False, mlinspect_transformer_node_id=None):
        """ Patch for ('sklearn.feature_extraction.text', 'HashingVectorizer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(text.HashingVectorizer, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_transformer_node_id = mlinspect_transformer_node_id

        self.mlinspect_non_data_func_args = {'input': input, 'encoding': encoding, 'decode_error': decode_error,
                                             'strip_accents': strip_accents, 'lowercase': lowercase,
                                             'preprocessor': preprocessor, 'tokenizer': tokenizer,
                                             'stop_words': stop_words, 'token_pattern': token_pattern,
                                             'ngram_range': ngram_range, 'analyzer': analyzer, 'n_features': n_features,
                                             'binary': binary, 'norm': norm, 'alternate_sign': alternate_sign,
                                             'dtype': dtype}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.feature_extraction.text.HashingVectorizer', 'fit_transform') """
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = True  # pylint: disable=attribute-defined-outside-init
        original = gorilla.get_original_attribute(text.HashingVectorizer, 'fit_transform')
        function_info = FunctionInfo('sklearn.feature_extraction.text', 'HashingVectorizer')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        def processing_func(input_df):
            transformer = text.HashingVectorizer(**self.mlinspect_non_data_func_args)
            transformed_data = transformer.fit_transform(input_df, *args[1:], **kwargs)
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
        optimizer_info, result = capture_optimizer_info(initial_func, estimator_transformer_state=self)
        dag_node_id = singleton.get_next_op_id()
        self.mlinspect_transformer_node_id = dag_node_id  # pylint: disable=attribute-defined-outside-init
        dag_node = DagNode(dag_node_id,
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Hashing Vectorizer: fit_transform", ['array'], optimizer_info),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code),
                           processing_func)
        function_call_result = FunctionCallResult(result)
        add_dag_node(dag_node, [input_info.dag_node], function_call_result)
        new_result = function_call_result.function_result
        self.mlinspect_fit_transform_active = False  # pylint: disable=attribute-defined-outside-init
        return new_result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.feature_extraction.text', 'HashingVectorizer') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(text.HashingVectorizer, 'transform')
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo('sklearn.feature_extraction.text', 'HashingVectorizer')
            input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                        self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

            def processing_func(fit_data, input_df):
                transformer = fit_data._mlinspect_annotation  # pylint: disable=protected-access
                transformed_data = transformer.transform(input_df, *args[1:], **kwargs)
                return transformed_data

            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
            optimizer_info, result = capture_optimizer_info(initial_func)
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("Hashing Vectorizer: transform", ['array'], optimizer_info),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code),
                               processing_func)
            function_call_result = FunctionCallResult(result)
            transformer_dag_node = get_dag_node_for_id(self.mlinspect_transformer_node_id)
            add_dag_node(dag_node, [transformer_dag_node, input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result
        else:
            new_result = original(self, *args, **kwargs)
        return new_result


@gorilla.patches(preprocessing.KBinsDiscretizer)
class SklearnKBinsDiscretizerPatching:
    """ Patches for sklearn KBinsDiscretizer"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, n_bins=5, *, encode='onehot', strategy='quantile',
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_fit_transform_active=False, mlinspect_transformer_node_id=None):
        """ Patch for ('sklearn.preprocessing._discretization', 'KBinsDiscretizer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(preprocessing.KBinsDiscretizer, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_transformer_node_id = mlinspect_transformer_node_id

        self.mlinspect_non_data_func_args = {'n_bins': n_bins, 'encode': encode, 'strategy': strategy}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._discretization.KBinsDiscretizer', 'fit_transform') """
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = True  # pylint: disable=attribute-defined-outside-init
        original = gorilla.get_original_attribute(preprocessing.KBinsDiscretizer, 'fit_transform')
        function_info = FunctionInfo('sklearn.preprocessing._discretization', 'KBinsDiscretizer')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        def processing_func(input_df):
            transformer = preprocessing.KBinsDiscretizer(**self.mlinspect_non_data_func_args)
            transformed_data = transformer.fit_transform(input_df, *args[1:], **kwargs)
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
        optimizer_info, result = capture_optimizer_info(initial_func, estimator_transformer_state=self)
        dag_node_id = singleton.get_next_op_id()
        self.mlinspect_transformer_node_id = dag_node_id  # pylint: disable=attribute-defined-outside-init
        dag_node = DagNode(dag_node_id,
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("K-Bins Discretizer: fit_transform", ['array'], optimizer_info),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code),
                           processing_func)
        function_call_result = FunctionCallResult(result)
        add_dag_node(dag_node, [input_info.dag_node], function_call_result)
        new_result = function_call_result.function_result
        assert isinstance(new_result, MlinspectNdarray)
        self.mlinspect_fit_transform_active = False  # pylint: disable=attribute-defined-outside-init
        return new_result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._discretization.KBinsDiscretizer', 'transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing.KBinsDiscretizer, 'transform')
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo('sklearn.preprocessing._discretization', 'KBinsDiscretizer')
            input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                        self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

            def processing_func(fit_data, input_df):
                transformer = fit_data._mlinspect_annotation  # pylint: disable=protected-access
                transformed_data = transformer.transform(input_df, *args[1:], **kwargs)
                return transformed_data

            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
            optimizer_info, result = capture_optimizer_info(initial_func)
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("K-Bins Discretizer: transform", ['array'], optimizer_info),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code),
                               processing_func)
            function_call_result = FunctionCallResult(result)
            transformer_dag_node = get_dag_node_for_id(self.mlinspect_transformer_node_id)
            add_dag_node(dag_node, [transformer_dag_node, input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result
            assert isinstance(new_result, MlinspectNdarray)
        else:
            new_result = original(self, *args, **kwargs)
        return new_result


@gorilla.patches(preprocessing.OneHotEncoder)
class SklearnOneHotEncoderPatching:
    """ Patches for sklearn OneHotEncoder"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, categories='auto', drop=None, sparse=True,
                        dtype=numpy.float64, handle_unknown='error',
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_fit_transform_active=False, mlinspect_transformer_node_id=None):
        """ Patch for ('sklearn.preprocessing._encoders', 'OneHotEncoder') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init,redefined-outer-name
        original = gorilla.get_original_attribute(preprocessing.OneHotEncoder, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_transformer_node_id = mlinspect_transformer_node_id

        self.mlinspect_non_data_func_args = {'categories': categories, 'drop': drop, 'sparse': sparse, 'dtype': dtype,
                                             'handle_unknown': handle_unknown}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._encoders.OneHotEncoder', 'fit_transform') """
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = True  # pylint: disable=attribute-defined-outside-init
        original = gorilla.get_original_attribute(preprocessing.OneHotEncoder, 'fit_transform')
        function_info = FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        def processing_func(input_df):
            transformer = preprocessing.OneHotEncoder(**self.mlinspect_non_data_func_args)
            transformed_data = transformer.fit_transform(input_df, *args[1:], **kwargs)
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
        optimizer_info, result = capture_optimizer_info(initial_func, estimator_transformer_state=self)
        dag_node_id = singleton.get_next_op_id()
        self.mlinspect_transformer_node_id = dag_node_id  # pylint: disable=attribute-defined-outside-init
        dag_node = DagNode(dag_node_id,
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("One-Hot Encoder: fit_transform", ['array'], optimizer_info),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code),
                           processing_func)
        function_call_result = FunctionCallResult(result)
        add_dag_node(dag_node, [input_info.dag_node], function_call_result)
        new_result = function_call_result.function_result
        self.mlinspect_fit_transform_active = False  # pylint: disable=attribute-defined-outside-init
        return new_result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing._encoders.OneHotEncoder', 'transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(preprocessing.OneHotEncoder, 'transform')
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')
            input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                        self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

            def processing_func(fit_data, input_df):
                transformer = fit_data._mlinspect_annotation  # pylint: disable=protected-access
                transformed_data = transformer.transform(input_df, *args[1:], **kwargs)
                return transformed_data

            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
            optimizer_info, result = capture_optimizer_info(initial_func)
            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("One-Hot Encoder: transform", ['array'], optimizer_info),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code),
                               processing_func)
            function_call_result = FunctionCallResult(result)
            transformer_dag_node = get_dag_node_for_id(self.mlinspect_transformer_node_id)
            add_dag_node(dag_node, [transformer_dag_node, input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result
        else:
            new_result = original(self, *args, **kwargs)
        return new_result


@gorilla.patches(impute.SimpleImputer)
class SklearnSimpleImputerPatching:
    """ Patches for sklearn SimpleImputer"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, missing_values=numpy.nan, strategy="mean",
                        fill_value=None, verbose=0, copy=True, add_indicator=False,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_fit_transform_active=False, mlinspect_transformer_node_id=None):
        """ Patch for ('sklearn.impute._base', 'SimpleImputer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init
        original = gorilla.get_original_attribute(impute.SimpleImputer, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_transformer_node_id = mlinspect_transformer_node_id

        self.mlinspect_non_data_func_args = {'missing_values': missing_values, 'strategy': strategy,
                                             'fill_value': fill_value, 'verbose': verbose, 'copy': copy,
                                             'add_indicator': add_indicator}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.impute._base.SimpleImputer', 'fit_transform') """
        # pylint: disable=no-method-argument,too-many-locals
        self.mlinspect_fit_transform_active = True  # pylint: disable=attribute-defined-outside-init
        original = gorilla.get_original_attribute(impute.SimpleImputer, 'fit_transform')
        function_info = FunctionInfo('sklearn.impute._base', 'SimpleImputer')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        def processing_func(input_df):
            transformer = impute.SimpleImputer(**self.mlinspect_non_data_func_args)
            transformed_data = transformer.fit_transform(input_df, *args[1:], **kwargs)
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
        optimizer_info, result = capture_optimizer_info(initial_func, estimator_transformer_state=self)
        if isinstance(input_info.annotated_dfobject.result_data, pandas.DataFrame):
            columns = list(input_info.annotated_dfobject.result_data.columns)
        else:
            columns = ['array']

        dag_node_id = singleton.get_next_op_id()
        self.mlinspect_transformer_node_id = dag_node_id  # pylint: disable=attribute-defined-outside-init
        dag_node = DagNode(dag_node_id,
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Simple Imputer: fit_transform", columns, optimizer_info),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code),
                           processing_func)
        function_call_result = FunctionCallResult(result)
        add_dag_node(dag_node, [input_info.dag_node], function_call_result)
        new_result = function_call_result.function_result
        self.mlinspect_fit_transform_active = False  # pylint: disable=attribute-defined-outside-init
        return new_result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.impute._base.SimpleImputer', 'transform') """
        # pylint: disable=no-method-argument,too-many-locals
        original = gorilla.get_original_attribute(impute.SimpleImputer, 'transform')
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo('sklearn.impute._base', 'SimpleImputer')
            input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                        self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

            def processing_func(fit_data, input_df):
                transformer = fit_data._mlinspect_annotation  # pylint: disable=protected-access
                transformed_data = transformer.transform(input_df, *args[1:], **kwargs)
                return transformed_data

            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
            optimizer_info, result = capture_optimizer_info(initial_func)
            if isinstance(input_info.annotated_dfobject.result_data, pandas.DataFrame):
                columns = list(input_info.annotated_dfobject.result_data.columns)
            else:
                columns = ['array']

            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("Simple Imputer: transform", columns, optimizer_info),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code),
                               processing_func)
            function_call_result = FunctionCallResult(result)
            transformer_dag_node = get_dag_node_for_id(self.mlinspect_transformer_node_id)
            add_dag_node(dag_node, [transformer_dag_node, input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result
        else:
            new_result = original(self, *args, **kwargs)
        return new_result


@gorilla.patches(preprocessing.FunctionTransformer)
class SklearnFunctionTransformerPatching:
    """ Patches for sklearn FunctionTransformer"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, func=None, inverse_func=None, *, validate=False, accept_sparse=False, check_inverse=True,
                        kw_args=None, inv_kw_args=None, mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_fit_transform_active=False, mlinspect_transformer_node_id=None):
        """ Patch for ('sklearn.preprocessing_function_transformer', 'FunctionTransformer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals
        original = gorilla.get_original_attribute(preprocessing.FunctionTransformer, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active
        self.mlinspect_transformer_node_id = mlinspect_transformer_node_id

        self.mlinspect_non_data_func_args = {'func': func, 'inverse_func': inverse_func, 'validate': validate,
                                             'accept_sparse': accept_sparse, 'check_inverse': check_inverse,
                                             'kw_args': kw_args, 'inv_kw_args': inv_kw_args}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.name('fit_transform')
    @gorilla.settings(allow_hit=True)
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing_function_transformer.FunctionTransformer', 'fit_transform') """
        # pylint: disable=no-method-argument,too-many-locals
        self.mlinspect_fit_transform_active = True  # pylint: disable=attribute-defined-outside-init
        original = gorilla.get_original_attribute(preprocessing.FunctionTransformer, 'fit_transform')
        function_info = FunctionInfo('sklearn.preprocessing_function_transformer', 'FunctionTransformer')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        def processing_func(input_df):
            input_df_copy = input_df.copy()
            transformer = preprocessing.FunctionTransformer(**self.mlinspect_non_data_func_args)
            transformed_data = transformer.fit_transform(input_df_copy, *args[1:], **kwargs)
            transformed_data = wrap_in_mlinspect_array_if_necessary(transformed_data)
            transformed_data._mlinspect_annotation = transformer  # pylint: disable=protected-access
            return transformed_data

        operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
        # This is to prevent udf monkey patching while a FunctionTransformer is active
        singleton.disable_monkey_patching = True
        initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
        optimizer_info, result = capture_optimizer_info(initial_func, estimator_transformer_state=self)
        # Enable monkey patching again
        singleton.disable_monkey_patching = False
        if isinstance(input_info.annotated_dfobject.result_data, pandas.DataFrame):
            columns = list(input_info.annotated_dfobject.result_data.columns)
        else:
            columns = ['array']

        dag_node_id = singleton.get_next_op_id()
        self.mlinspect_transformer_node_id = dag_node_id  # pylint: disable=attribute-defined-outside-init
        dag_node = DagNode(dag_node_id,
                           BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                           operator_context,
                           DagNodeDetails("Function Transformer: fit_transform", columns, optimizer_info),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code),
                           processing_func)
        function_call_result = FunctionCallResult(result)
        add_dag_node(dag_node, [input_info.dag_node], function_call_result)
        new_result = function_call_result.function_result
        self.mlinspect_fit_transform_active = False  # pylint: disable=attribute-defined-outside-init
        return new_result

    @gorilla.name('transform')
    @gorilla.settings(allow_hit=True)
    def patched_transform(self, *args, **kwargs):
        """ Patch for ('sklearn.preprocessing_function_transformer.FunctionTransformer', 'transform') """
        # pylint: disable=no-method-argument,too-many-locals
        original = gorilla.get_original_attribute(preprocessing.FunctionTransformer, 'transform')
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo('sklearn.preprocessing_function_transformer', 'FunctionTransformer')
            input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                        self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

            def processing_func(fit_data, input_df):
                input_df_copy = input_df.copy()
                transformer = fit_data._mlinspect_annotation  # pylint: disable=protected-access
                transformed_data = transformer.transform(input_df_copy, *args[1:], **kwargs)
                return transformed_data

            operator_context = OperatorContext(OperatorType.TRANSFORMER, function_info)
            # This is to prevent udf monkey patching while a FunctionTransformer is active
            singleton.disable_monkey_patching = True
            initial_func = partial(original, self, input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
            optimizer_info, result = capture_optimizer_info(initial_func)
            # Enable monkey patching again
            singleton.disable_monkey_patching = False
            # End disable hack
            if isinstance(input_info.annotated_dfobject.result_data, pandas.DataFrame):
                columns = list(input_info.annotated_dfobject.result_data.columns)
            else:
                columns = ['array']

            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("Function Transformer: transform", columns, optimizer_info),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code),
                               processing_func)
            function_call_result = FunctionCallResult(result)
            transformer_dag_node = get_dag_node_for_id(self.mlinspect_transformer_node_id)
            add_dag_node(dag_node, [transformer_dag_node, input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result
        else:
            new_result = original(self, *args, **kwargs)
        return new_result


@gorilla.patches(tree.DecisionTreeClassifier)
class SklearnDecisionTreePatching:
    """ Patches for sklearn DecisionTree"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, criterion="gini", splitter="best", max_depth=None, min_samples_split=2,
                        min_samples_leaf=1, min_weight_fraction_leaf=0., max_features=None, random_state=None,
                        max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None, class_weight=None,
                        presort='deprecated', ccp_alpha=0.0, mlinspect_caller_filename=None,
                        mlinspect_lineno=None, mlinspect_optional_code_reference=None,
                        mlinspect_optional_source_code=None, mlinspect_estimator_node_id=None):
        """ Patch for ('sklearn.tree._classes', 'DecisionTreeClassifier') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals
        original = gorilla.get_original_attribute(tree.DecisionTreeClassifier, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_estimator_node_id = mlinspect_estimator_node_id

        self.mlinspect_non_data_func_args = {'criterion': criterion, 'splitter': splitter, 'max_depth': max_depth,
                                             'min_samples_split': min_samples_split,
                                             'min_samples_leaf': min_samples_leaf,
                                             'min_weight_fraction_leaf': min_weight_fraction_leaf,
                                             'max_features': max_features, 'random_state': random_state,
                                             'max_leaf_nodes': max_leaf_nodes,
                                             'min_impurity_decrease': min_impurity_decrease,
                                             'min_impurity_split': min_impurity_split, 'class_weight': class_weight,
                                             'presort': presort, 'ccp_alpha': ccp_alpha}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code
            self.mlinspect_estimator_node_id = None

        return execute_patched_func_no_op_id(original, execute_inspections, self,
                                             **self.mlinspect_non_data_func_args)

    @gorilla.name('fit')
    @gorilla.settings(allow_hit=True)
    def patched_fit(self, *args, **kwargs):
        """ Patch for ('sklearn.tree._classes.DecisionTreeClassifier', 'fit') """
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(tree.DecisionTreeClassifier, 'fit')
        if not call_info_singleton.param_search_active:
            function_info = FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')

            _, train_data_node, train_data_result = add_train_data_node(self, args[0], function_info)
            _, train_labels_node, train_labels_result = add_train_label_node(self, args[1],
                                                                             function_info)

            if call_info_singleton.make_grid_search_func is None:
                def processing_func(train_data, train_labels):
                    estimator = tree.DecisionTreeClassifier(**self.mlinspect_non_data_func_args)
                    fitted_estimator = estimator.fit(train_data, train_labels, *args[2:], **kwargs)
                    return fitted_estimator
                create_func = partial(tree.DecisionTreeClassifier, **self.mlinspect_non_data_func_args)
                param_search_runtime = 0
            else:
                def processing_func_with_grid_search(make_grid_search_func, train_data, train_labels):
                    estimator = make_grid_search_func(tree.DecisionTreeClassifier(**self.mlinspect_non_data_func_args))
                    fitted_estimator = estimator.fit(train_data, train_labels, *args[2:], **kwargs)
                    return fitted_estimator
                processing_func = partial(processing_func_with_grid_search, call_info_singleton.make_grid_search_func)

                def create_func_with_grid_search(make_grid_search_func):
                    return make_grid_search_func(tree.DecisionTreeClassifier(**self.mlinspect_non_data_func_args))
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
                               DagNodeDetails("Decision Tree", [], optimizer_info_with_search),
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
        """ Patch for ('sklearn.tree._classes.DecisionTreeClassifier', 'score') """

        # pylint: disable=no-method-argument
        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            if len(kwargs) != 0:
                raise Exception("TODO: Support other metrics in model.score calls!")

            function_info = FunctionInfo('sklearn.tree._classes.DecisionTreeClassifier', 'score')
            _, test_data_node, test_data_result = add_test_data_dag_node(args[0],
                                                                         function_info,
                                                                         lineno,
                                                                         optional_code_reference,
                                                                         optional_source_code,
                                                                         caller_filename)
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
                score = accuracy_score(predictions, test_labels)
                return score

            original_predict = gorilla.get_original_attribute(tree.DecisionTreeClassifier, 'predict')
            initial_func_predict = partial(original_predict, self, test_data_result)
            optimizer_info_predict, result_predict = capture_optimizer_info(initial_func_predict)
            operator_context_predict = OperatorContext(OperatorType.PREDICT, function_info)
            dag_node_predict = DagNode(singleton.get_next_op_id(),
                                       BasicCodeLocation(caller_filename, lineno),
                                       operator_context_predict,
                                       DagNodeDetails("Decision Tree", [], optimizer_info_predict),
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
            original = gorilla.get_original_attribute(tree.DecisionTreeClassifier, 'score')
            new_result = original(self, *args, **kwargs)
        return new_result

    @gorilla.name('predict')
    @gorilla.settings(allow_hit=True)
    def patched_predict(self, *args):
        """ Patch for ('sklearn.tree._classes.DecisionTreeClassifier', 'predict') """

        # pylint: disable=no-method-argument
        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('sklearn.tree._classes.DecisionTreeClassifier', 'predict')
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

            original_predict = gorilla.get_original_attribute(tree.DecisionTreeClassifier, 'predict')
            initial_func_predict = partial(original_predict, self, test_data_result)
            optimizer_info_predict, result_predict = capture_optimizer_info(initial_func_predict)
            operator_context_predict = OperatorContext(OperatorType.PREDICT, function_info)
            dag_node_predict = DagNode(singleton.get_next_op_id(),
                                       BasicCodeLocation(caller_filename, lineno),
                                       operator_context_predict,
                                       DagNodeDetails("Decision Tree", [], optimizer_info_predict),
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
            original = gorilla.get_original_attribute(tree.DecisionTreeClassifier, 'predict')
            new_result = original(self, *args)
        return new_result


@gorilla.patches(linear_model.SGDClassifier)
class SklearnSGDClassifierPatching:
    """ Patches for sklearn SGDClassifier"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, loss="hinge", *, penalty='l2', alpha=0.0001, l1_ratio=0.15,
                        fit_intercept=True, max_iter=1000, tol=1e-3, shuffle=True, verbose=0, epsilon=DEFAULT_EPSILON,
                        n_jobs=None, random_state=None, learning_rate="optimal", eta0=0.0, power_t=0.5,
                        early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None,
                        warm_start=False, average=False, mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_estimator_node_id=None):
        """ Patch for ('sklearn.linear_model._stochastic_gradient', 'SGDClassifier') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals
        original = gorilla.get_original_attribute(linear_model.SGDClassifier, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_estimator_node_id = mlinspect_estimator_node_id

        self.mlinspect_non_data_func_args = {'loss': loss, 'penalty': penalty, 'alpha': alpha, 'l1_ratio': l1_ratio,
                                             'fit_intercept': fit_intercept, 'max_iter': max_iter, 'tol': tol,
                                             'shuffle': shuffle, 'verbose': verbose, 'epsilon': epsilon,
                                             'n_jobs': n_jobs, 'random_state': random_state,
                                             'learning_rate': learning_rate, 'eta0': eta0, 'power_t': power_t,
                                             'early_stopping': early_stopping,
                                             'validation_fraction': validation_fraction,
                                             'n_iter_no_change': n_iter_no_change,
                                             'class_weight': class_weight, 'warm_start': warm_start, 'average': average}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code
            self.mlinspect_estimator_node_id = None

        return execute_patched_func_no_op_id(original, execute_inspections, self,
                                             **self.mlinspect_non_data_func_args)

    @gorilla.name('fit')
    @gorilla.settings(allow_hit=True)
    def patched_fit(self, *args, **kwargs):
        """ Patch for ('sklearn.linear_model._stochastic_gradient', 'fit') """
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(linear_model.SGDClassifier, 'fit')
        if not call_info_singleton.param_search_active:
            function_info = FunctionInfo('sklearn.linear_model._stochastic_gradient', 'SGDClassifier')

            _, train_data_node, train_data_result = add_train_data_node(self, args[0], function_info)
            _, train_labels_node, train_labels_result = add_train_label_node(self, args[1],
                                                                             function_info)
            if call_info_singleton.make_grid_search_func is None:
                def processing_func(train_data, train_labels):
                    estimator = linear_model.SGDClassifier(**self.mlinspect_non_data_func_args)
                    fitted_estimator = estimator.fit(train_data, train_labels, *args[2:], **kwargs)
                    return fitted_estimator
                param_search_runtime = 0
                create_func = partial(linear_model.SGDClassifier, **self.mlinspect_non_data_func_args)
            else:
                def processing_func_with_grid_search(make_grid_search_func, train_data, train_labels):
                    estimator = make_grid_search_func(linear_model.SGDClassifier(**self.mlinspect_non_data_func_args))
                    fitted_estimator = estimator.fit(train_data, train_labels, *args[2:], **kwargs)
                    return fitted_estimator
                processing_func = partial(processing_func_with_grid_search, call_info_singleton.make_grid_search_func)

                def create_func_with_grid_search(make_grid_search_func):
                    return make_grid_search_func(linear_model.SGDClassifier(**self.mlinspect_non_data_func_args))
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
                               DagNodeDetails("SGD Classifier", [], optimizer_info_with_search),
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
        """ Patch for ('sklearn.linear_model._stochastic_gradient.SGDClassifier', 'score') """

        # pylint: disable=no-method-argument
        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            if len(kwargs) != 0:
                raise Exception("TODO: Support other metrics in model.score calls!")

            function_info = FunctionInfo('sklearn.linear_model._stochastic_gradient.SGDClassifier', 'score')
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
                score = accuracy_score(predictions, test_labels)
                return score

            original_predict = gorilla.get_original_attribute(linear_model.SGDClassifier, 'predict')
            initial_func_predict = partial(original_predict, self, test_data_result)
            optimizer_info_predict, result_predict = capture_optimizer_info(initial_func_predict)
            operator_context_predict = OperatorContext(OperatorType.PREDICT, function_info)
            dag_node_predict = DagNode(singleton.get_next_op_id(),
                                       BasicCodeLocation(caller_filename, lineno),
                                       operator_context_predict,
                                       DagNodeDetails("SGD Classifier", [], optimizer_info_predict),
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
            original = gorilla.get_original_attribute(linear_model.SGDClassifier, 'score')
            new_result = original(self, *args, **kwargs)
        return new_result

    @gorilla.name('predict')
    @gorilla.settings(allow_hit=True)
    def patched_predict(self, *args):
        """ Patch for ('sklearn.linear_model._stochastic_gradient.SGDClassifier', 'predict') """

        # pylint: disable=no-method-argument
        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('sklearn.linear_model._stochastic_gradient.SGDClassifier', 'predict')
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

            original_predict = gorilla.get_original_attribute(linear_model.SGDClassifier, 'predict')
            initial_func_predict = partial(original_predict, self, test_data_result)
            optimizer_info_predict, result_predict = capture_optimizer_info(initial_func_predict)
            operator_context_predict = OperatorContext(OperatorType.PREDICT, function_info)
            dag_node_predict = DagNode(singleton.get_next_op_id(),
                                       BasicCodeLocation(caller_filename, lineno),
                                       operator_context_predict,
                                       DagNodeDetails("SGD Classifier", [], optimizer_info_predict),
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
            original = gorilla.get_original_attribute(linear_model.SGDClassifier, 'predict')
            new_result = original(self, *args)
        return new_result


@gorilla.patches(linear_model.LogisticRegression)
class SklearnLogisticRegressionPatching:
    """ Patches for sklearn LogisticRegression"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, penalty='l2', *, dual=False, tol=1e-4, C=1.0,  # pylint: disable=invalid-name
                        fit_intercept=True, intercept_scaling=1, class_weight=None,
                        random_state=None, solver='lbfgs', max_iter=100,
                        multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                        l1_ratio=None, mlinspect_caller_filename=None,
                        mlinspect_lineno=None, mlinspect_optional_code_reference=None,
                        mlinspect_optional_source_code=None, mlinspect_estimator_node_id=None):
        """ Patch for ('sklearn.linear_model._logistic', 'LogisticRegression') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals
        original = gorilla.get_original_attribute(linear_model.LogisticRegression, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_estimator_node_id = mlinspect_estimator_node_id

        self.mlinspect_non_data_func_args = {'penalty': penalty, 'dual': dual, 'tol': tol, 'C': C,
                                             'fit_intercept': fit_intercept, 'intercept_scaling': intercept_scaling,
                                             'class_weight': class_weight, 'random_state': random_state,
                                             'solver': solver, 'max_iter': max_iter, 'multi_class': multi_class,
                                             'verbose': verbose, 'warm_start': warm_start, 'n_jobs': n_jobs,
                                             'l1_ratio': l1_ratio}

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
        """ Patch for ('sklearn.linear_model._logistic.LogisticRegression', 'fit') """
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(linear_model.LogisticRegression, 'fit')
        if not call_info_singleton.param_search_active:
            function_info = FunctionInfo('sklearn.linear_model._logistic', 'LogisticRegression')

            _, train_data_node, train_data_result = add_train_data_node(self, args[0], function_info)
            _, train_labels_node, train_labels_result = add_train_label_node(self, args[1], function_info)

            if call_info_singleton.make_grid_search_func is None:
                def processing_func(train_data, train_labels):
                    estimator = linear_model.LogisticRegression(**self.mlinspect_non_data_func_args)
                    fitted_estimator = estimator.fit(train_data, train_labels, *args[2:], **kwargs)
                    return fitted_estimator
                param_search_runtime = 0
                create_func = partial(linear_model.LogisticRegression, **self.mlinspect_non_data_func_args)
            else:
                def processing_func_with_grid_search(make_grid_search_func, train_data, train_labels):
                    estimator = make_grid_search_func(linear_model.LogisticRegression(
                        **self.mlinspect_non_data_func_args))
                    fitted_estimator = estimator.fit(train_data, train_labels, *args[2:], **kwargs)
                    return fitted_estimator
                processing_func = partial(processing_func_with_grid_search, call_info_singleton.make_grid_search_func)

                def create_func_with_grid_search(make_grid_search_func):
                    return make_grid_search_func(linear_model.LogisticRegression(**self.mlinspect_non_data_func_args))
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
                               DagNodeDetails("Logistic Regression", [], optimizer_info_with_search),
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
        """ Patch for ('sklearn.linear_model._logistic.LogisticRegression', 'score') """

        # pylint: disable=no-method-argument
        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            if len(kwargs) != 0:
                raise Exception("TODO: Support other metrics in model.score calls!")

            function_info = FunctionInfo('sklearn.linear_model._logistic.LogisticRegression', 'score')
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
                score = accuracy_score(predictions, test_labels)
                return score

            # input_dfs = [data_backend_result.annotated_dfobject, label_backend_result.annotated_dfobject]

            original_predict = gorilla.get_original_attribute(linear_model.LogisticRegression, 'predict')
            initial_func_predict = partial(original_predict, self, test_data_result)
            optimizer_info_predict, result_predict = capture_optimizer_info(initial_func_predict)
            operator_context_predict = OperatorContext(OperatorType.PREDICT, function_info)
            dag_node_predict = DagNode(singleton.get_next_op_id(),
                                       BasicCodeLocation(caller_filename, lineno),
                                       operator_context_predict,
                                       DagNodeDetails("Logistic Regression", [], optimizer_info_predict),
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
            original = gorilla.get_original_attribute(linear_model.LogisticRegression, 'score')
            new_result = original(self, *args, **kwargs)
        return new_result

    @gorilla.name('predict')
    @gorilla.settings(allow_hit=True)
    def patched_predict(self, *args):
        """ Patch for ('sklearn.linear_model._logistic.LogisticRegression', 'predict') """

        # pylint: disable=no-method-argument
        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('sklearn.linear_model._logistic.LogisticRegression', 'predict')
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

            original_predict = gorilla.get_original_attribute(linear_model.LogisticRegression, 'predict')
            initial_func_predict = partial(original_predict, self, test_data_result)
            optimizer_info_predict, result_predict = capture_optimizer_info(initial_func_predict)
            operator_context_predict = OperatorContext(OperatorType.PREDICT, function_info)
            dag_node_predict = DagNode(singleton.get_next_op_id(),
                                       BasicCodeLocation(caller_filename, lineno),
                                       operator_context_predict,
                                       DagNodeDetails("Logistic Regression", [], optimizer_info_predict),
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
            original = gorilla.get_original_attribute(linear_model.LogisticRegression, 'predict')
            new_result = original(self, *args)
        return new_result


class SklearnKerasClassifierPatching:
    """ Patches for tensorflow KerasClassifier"""

    # pylint: disable=too-few-public-methods
    @gorilla.patch(keras_sklearn_internal.BaseWrapper, name='__init__', settings=gorilla.Settings(allow_hit=True))
    def patched__init__(self, build_fn, mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_estimator_node_id=None, **sk_params):
        """ Patch for ('tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals, too-many-arguments
        original = gorilla.get_original_attribute(keras_sklearn_internal.BaseWrapper, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_estimator_node_id = mlinspect_estimator_node_id

        self.mlinspect_non_data_func_args = {'build_fn': build_fn, **sk_params}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_indirect_allowed(execute_inspections)

    @gorilla.patch(keras_sklearn_external.KerasClassifier, name='fit', settings=gorilla.Settings(allow_hit=True))
    def patched_fit(self, *args, **kwargs):
        """ Patch for ('tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier', 'fit') """
        # pylint: disable=no-method-argument, too-many-locals
        class KerasClassifierAdjustableInputDims(keras_sklearn_external.KerasClassifier):
            """A Keras Wrapper that sets input_dim on fit"""

            def fit(self, x, y, **kwargs):
                """Create and fit a simple neural network"""
                self.sk_params['input_dim'] = x.shape[1]
                super().fit(x, y, **kwargs)

        original = gorilla.get_original_attribute(keras_sklearn_external.KerasClassifier, 'fit')
        if not call_info_singleton.param_search_active:
            function_info = FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier')

            _, train_data_dag_node, train_data_result = add_train_data_node(self, args[0], function_info)
            _, train_labels_dag_node, train_labels_result = add_train_label_node(self, args[1], function_info)
            self.mlinspect_non_data_func_args.update(self.sk_params)  # pylint: disable=no-member

            if call_info_singleton.make_grid_search_func is None:
                def processing_func(train_data, train_labels):
                    estimator = tensorflow.keras.wrappers.scikit_learn.KerasClassifier(
                        **self.mlinspect_non_data_func_args)
                    estimator.sk_params['input_dim'] = train_data.shape[1]
                    estimator.fit(train_data, train_labels, *args[2:], **kwargs)
                    return estimator
                param_search_runtime = 0
                create_func = partial(KerasClassifierAdjustableInputDims, **self.mlinspect_non_data_func_args)
            else:
                def processing_func_with_grid_search(make_grid_search_func, train_data, train_labels):
                    estimator = make_grid_search_func(tensorflow.keras.wrappers.scikit_learn.KerasClassifier(
                        **self.mlinspect_non_data_func_args))
                    estimator.estimator.sk_params['input_dim'] = train_data.shape[1]
                    estimator.fit(train_data, train_labels, *args[2:], **kwargs)
                    return estimator
                processing_func = partial(processing_func_with_grid_search, call_info_singleton.make_grid_search_func)

                def create_func_with_grid_search(make_grid_search_func):
                    return make_grid_search_func(KerasClassifierAdjustableInputDims(
                        **self.mlinspect_non_data_func_args))
                create_func = partial(create_func_with_grid_search, call_info_singleton.make_grid_search_func)

                call_info_singleton.make_grid_search_func = None
                param_search_runtime = call_info_singleton.param_search_duration
                call_info_singleton.param_search_duration = 0

            # Estimator
            operator_context = OperatorContext(OperatorType.ESTIMATOR, function_info)
            # input_dfs = [data_backend_result.annotated_dfobject, label_backend_result.annotated_dfobject]
            initial_func = partial(original, self, train_data_result, train_labels_result, *args[2:], **kwargs)
            keras_batch_size = self.sk_params.get("batch_size", 32)  # pylint: disable=no-member
            optimizer_info, _ = capture_optimizer_info(initial_func, self, estimator_transformer_state=self,
                                                       keras_batch_size=keras_batch_size)
            optimizer_info_with_search = OptimizerInfo(optimizer_info.runtime + param_search_runtime,
                                                       optimizer_info.shape, optimizer_info.memory)
            self.mlinspect_estimator_node_id = singleton.get_next_op_id()  # pylint: disable=attribute-defined-outside-init
            dag_node = DagNode(self.mlinspect_estimator_node_id,
                               BasicCodeLocation(self.mlinspect_caller_filename, self.mlinspect_lineno),
                               operator_context,
                               DagNodeDetails("Neural Network", [], optimizer_info_with_search),
                               get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                              self.mlinspect_optional_source_code),
                               processing_func,
                               create_func)
            function_call_result = FunctionCallResult(None)
            add_dag_node(dag_node, [train_data_dag_node, train_labels_dag_node], function_call_result)
        else:
            original(self, *args, **kwargs)
        return self

    @gorilla.patch(keras_sklearn_external.KerasClassifier, name='score', settings=gorilla.Settings(allow_hit=True))
    def patched_score(self, *args, **kwargs):
        """ Patch for ('tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier', 'score') """

        # pylint: disable=no-method-argument
        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier', 'score')
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
                one_d_labels = numpy.argmax(test_labels, axis=1)
                score = accuracy_score(predictions, one_d_labels)
                return score

            # Score
            operator_context_predict = OperatorContext(OperatorType.PREDICT, function_info)
            operator_context_score = OperatorContext(OperatorType.SCORE, function_info)
            # input_dfs = [data_backend_result.annotated_dfobject, label_backend_result.annotated_dfobject]

            # This currently calls predict twice, but patching here is complex. Maybe revisit this in future work
            uninstrumented_predict = gorilla.get_original_attribute(keras_sklearn_external.KerasClassifier, 'predict')
            initial_func_predict = partial(uninstrumented_predict, self, test_data_result)  # pylint: disable=no-member
            optimizer_info_predict, result_predict = capture_optimizer_info(initial_func_predict)

            dag_node_predict = DagNode(singleton.get_next_op_id(),
                                       BasicCodeLocation(caller_filename, lineno),
                                       operator_context_predict,
                                       DagNodeDetails("Neural Network", [], optimizer_info_predict),
                                       get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                                       processing_func_predict)
            estimator_dag_node = get_dag_node_for_id(self.mlinspect_estimator_node_id)
            function_call_result = FunctionCallResult(result_predict)
            add_dag_node(dag_node_predict, [estimator_dag_node, test_data_node],
                         function_call_result)

            initial_func_score = partial(processing_func_score, result_predict, test_labels_result, *args[2:],
                                         **kwargs)
            optimizer_info_score, result_score = capture_optimizer_info(initial_func_score)

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
            original = gorilla.get_original_attribute(keras_sklearn_external.KerasClassifier, 'score')
            new_result = original(self, *args, **kwargs)
        return new_result

    @gorilla.patch(keras_sklearn_external.KerasClassifier, name='predict', settings=gorilla.Settings(allow_hit=True))
    def patched_predict(self, *args):
        """ Patch for ('tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier', 'score') """

        # pylint: disable=no-method-argument
        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            original = gorilla.get_original_attribute(keras_sklearn_external.KerasClassifier, 'predict')
            function_info = FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier', 'predict')
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

            # Score
            operator_context_predict = OperatorContext(OperatorType.PREDICT, function_info)

            initial_func_predict = partial(original, self, test_data_result)  # pylint: disable=no-member
            optimizer_info_predict, result_predict = capture_optimizer_info(initial_func_predict)

            dag_node_predict = DagNode(singleton.get_next_op_id(),
                                       BasicCodeLocation(caller_filename, lineno),
                                       operator_context_predict,
                                       DagNodeDetails("Neural Network", [], optimizer_info_predict),
                                       get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                                       processing_func_predict)
            estimator_dag_node = get_dag_node_for_id(self.mlinspect_estimator_node_id)
            function_call_result = FunctionCallResult(result_predict)
            add_dag_node(dag_node_predict, [estimator_dag_node, test_data_node],
                         function_call_result)
            new_result = function_call_result.function_result
            return new_result

        if not call_info_singleton.param_search_active:
            new_result = execute_patched_func_indirect_allowed(execute_inspections)
        else:
            original = gorilla.get_original_attribute(keras_sklearn_external.KerasClassifier, 'predict')
            new_result = original(self, *args)
        return new_result


@gorilla.patches(metrics)
class MetricsPatching:
    """ Patches for 'sklearn.metrics' """

    # pylint: disable=too-few-public-methods

    @gorilla.name('accuracy_score')
    @gorilla.settings(allow_hit=True)
    def patched_accuracy_score(y_true, y_pred, *args, **kwargs):
        """ Patch for ('sklearn.metrics._classification', 'accuracy_score') """
        # pylint: disable=too-many-locals, no-self-argument
        original = gorilla.get_original_attribute(metrics, 'accuracy_score')

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('sklearn.metrics._classification', 'accuracy_score')

            input_info_pred = get_input_info(y_pred, caller_filename, lineno, function_info,
                                             optional_code_reference, optional_source_code)

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

            def process_metric_frame(y_true, y_pred):
                return original(y_true=y_true, y_pred=y_pred)

            dag_node = DagNode(singleton.get_next_op_id(),
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails('accuracy_score', [], optimizer_info),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                               process_metric_frame)
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [input_info_pred.dag_node, test_labels_node], function_call_result)
            return result

        return execute_patched_func_no_op_id(original, execute_inspections, y_true, y_pred, *args, **kwargs)


@gorilla.patches(dummy.DummyClassifier)
class SklearnDummyClassifierPatching:
    """ Patches for sklearn LogisticRegression"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, strategy="warn", random_state=None, constant=None, mlinspect_caller_filename=None,
                        mlinspect_lineno=None, mlinspect_optional_code_reference=None,
                        mlinspect_optional_source_code=None, mlinspect_estimator_node_id=None):
        """ Patch for ('sklearn.dummy.DummyClassifier', 'LogisticRegression') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals, too-many-arguments
        original = gorilla.get_original_attribute(dummy.DummyClassifier, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_estimator_node_id = mlinspect_estimator_node_id

        self.mlinspect_non_data_func_args = {'strategy': strategy, 'random_state': random_state, 'constant': constant}

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
        """ Patch for ('sklearn.dummy.DummyClassifier', 'fit') """
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(dummy.DummyClassifier, 'fit')
        if not call_info_singleton.param_search_active:
            function_info = FunctionInfo('sklearn.dummy', 'DummyClassifier')

            _, train_data_node, train_data_result = add_train_data_node(self, args[0], function_info)
            _, train_labels_node, train_labels_result = add_train_label_node(self, args[1], function_info)

            if call_info_singleton.make_grid_search_func is None:
                def processing_func(train_data, train_labels):
                    estimator = dummy.DummyClassifier(**self.mlinspect_non_data_func_args)
                    fitted_estimator = estimator.fit(train_data, train_labels, *args[2:], **kwargs)
                    return fitted_estimator
                param_search_runtime = 0
                create_func = partial(dummy.DummyClassifier, **self.mlinspect_non_data_func_args)
            else:
                def processing_func_with_grid_search(make_grid_search_func, train_data, train_labels):
                    estimator = make_grid_search_func(dummy.DummyClassifier(
                        **self.mlinspect_non_data_func_args))
                    fitted_estimator = estimator.fit(train_data, train_labels, *args[2:], **kwargs)
                    return fitted_estimator
                processing_func = partial(processing_func_with_grid_search, call_info_singleton.make_grid_search_func)

                def create_func_with_grid_search(make_grid_search_func):
                    return make_grid_search_func(dummy.DummyClassifier(**self.mlinspect_non_data_func_args))
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
                               DagNodeDetails("Dummy Classifier", [], optimizer_info_with_search),
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
        """ Patch for ('sklearn.dummy.DummyClassifier', 'score') """

        # pylint: disable=no-method-argument
        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            if len(kwargs) != 0:
                raise Exception("TODO: Support other metrics in model.score calls!")

            function_info = FunctionInfo('sklearn.dummy.DummyClassifier', 'score')
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
                score = accuracy_score(predictions, test_labels)
                return score

            # input_dfs = [data_backend_result.annotated_dfobject, label_backend_result.annotated_dfobject]

            original_predict = gorilla.get_original_attribute(dummy.DummyClassifier, 'predict')
            initial_func_predict = partial(original_predict, self, test_data_result)
            optimizer_info_predict, result_predict = capture_optimizer_info(initial_func_predict)
            operator_context_predict = OperatorContext(OperatorType.PREDICT, function_info)
            dag_node_predict = DagNode(singleton.get_next_op_id(),
                                       BasicCodeLocation(caller_filename, lineno),
                                       operator_context_predict,
                                       DagNodeDetails("Dummy Classifier", [], optimizer_info_predict),
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
            original = gorilla.get_original_attribute(dummy.DummyClassifier, 'score')
            new_result = original(self, *args, **kwargs)
        return new_result

    @gorilla.name('predict')
    @gorilla.settings(allow_hit=True)
    def patched_predict(self, *args):
        """ Patch for ('sklearn.dummy.DummyClassifier', 'predict') """

        # pylint: disable=no-method-argument
        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('sklearn.dummy.DummyClassifier', 'predict')
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

            original_predict = gorilla.get_original_attribute(dummy.DummyClassifier, 'predict')
            initial_func_predict = partial(original_predict, self, test_data_result)
            optimizer_info_predict, result_predict = capture_optimizer_info(initial_func_predict)
            operator_context_predict = OperatorContext(OperatorType.PREDICT, function_info)
            dag_node_predict = DagNode(singleton.get_next_op_id(),
                                       BasicCodeLocation(caller_filename, lineno),
                                       operator_context_predict,
                                       DagNodeDetails("Dummy Classifier", [], optimizer_info_predict),
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
            original = gorilla.get_original_attribute(dummy.DummyClassifier, 'predict')
            new_result = original(self, *args)
        return new_result


@gorilla.patches(svm.SVC)
class SklearnSVCPatching:
    """ Patches for sklearn SVC"""

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *, C=1.0, kernel='rbf', degree=3, gamma='scale',
                        coef0=0.0, shrinking=True, probability=False,
                        tol=1e-3, cache_size=200, class_weight=None,
                        verbose=False, max_iter=-1, decision_function_shape='ovr',
                        break_ties=False,
                        random_state=None, mlinspect_caller_filename=None,
                        mlinspect_lineno=None, mlinspect_optional_code_reference=None,
                        mlinspect_optional_source_code=None, mlinspect_estimator_node_id=None):
        """ Patch for ('sklearn.svm._classes', 'SVC') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals, invalid-name
        original = gorilla.get_original_attribute(svm.SVC, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_estimator_node_id = mlinspect_estimator_node_id

        self.mlinspect_non_data_func_args = {'C': C, 'kernel': kernel, 'degree': degree, 'gamma': gamma,
                                             'coef0': coef0, 'shrinking': shrinking, 'probability': probability,
                                             'tol': tol, 'cache_size': cache_size, 'class_weight': class_weight,
                                             'verbose': verbose, 'max_iter': max_iter,
                                             'decision_function_shape': decision_function_shape,
                                             'break_ties': break_ties, 'random_state': random_state
                                             }

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self,
                                             **self.mlinspect_non_data_func_args)

    @gorilla.name('fit')
    @gorilla.settings(allow_hit=True)
    def patched_fit(self, *args, **kwargs):
        """ Patch for ('sklearn.svm._classes.SVC', 'fit') """
        # pylint: disable=no-method-argument, too-many-locals
        original = gorilla.get_original_attribute(svm.SVC, 'fit')
        if not call_info_singleton.param_search_active:
            function_info = FunctionInfo('sklearn.svm._classes', 'SVC')

            _, train_data_node, train_data_result = add_train_data_node(self, args[0], function_info)
            _, train_labels_node, train_labels_result = add_train_label_node(self, args[1], function_info)

            if call_info_singleton.make_grid_search_func is None:
                def processing_func(train_data, train_labels):
                    estimator = svm.SVC(**self.mlinspect_non_data_func_args)
                    fitted_estimator = estimator.fit(train_data, train_labels, *args[2:], **kwargs)
                    return fitted_estimator

                param_search_runtime = 0
                create_func = partial(svm.SVC, **self.mlinspect_non_data_func_args)
            else:
                def processing_func_with_grid_search(make_grid_search_func, train_data, train_labels):
                    estimator = make_grid_search_func(svm.SVC(
                        **self.mlinspect_non_data_func_args))
                    fitted_estimator = estimator.fit(train_data, train_labels, *args[2:], **kwargs)
                    return fitted_estimator

                processing_func = partial(processing_func_with_grid_search,
                                          call_info_singleton.make_grid_search_func)

                def create_func_with_grid_search(make_grid_search_func):
                    return make_grid_search_func(svm.SVC(**self.mlinspect_non_data_func_args))

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
                               DagNodeDetails("SVC", [], optimizer_info_with_search),
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
        """ Patch for ('sklearn.svm._classes.SVC', 'score') """

        # pylint: disable=no-method-argument
        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            if len(kwargs) != 0:
                raise Exception("TODO: Support other metrics in model.score calls!")

            function_info = FunctionInfo('sklearn.svm._classes.SVC', 'score')
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
                score = accuracy_score(predictions, test_labels)
                return score

            # input_dfs = [data_backend_result.annotated_dfobject, label_backend_result.annotated_dfobject]

            original_predict = gorilla.get_original_attribute(svm.SVC, 'predict')
            initial_func_predict = partial(original_predict, self, test_data_result)
            optimizer_info_predict, result_predict = capture_optimizer_info(initial_func_predict)
            operator_context_predict = OperatorContext(OperatorType.PREDICT, function_info)
            dag_node_predict = DagNode(singleton.get_next_op_id(),
                                       BasicCodeLocation(caller_filename, lineno),
                                       operator_context_predict,
                                       DagNodeDetails("SVC", [], optimizer_info_predict),
                                       get_optional_code_info_or_none(optional_code_reference,
                                                                      optional_source_code),
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
            original = gorilla.get_original_attribute(svm.SVC, 'score')
            new_result = original(self, *args, **kwargs)
        return new_result

    @gorilla.name('predict')
    @gorilla.settings(allow_hit=True)
    def patched_predict(self, *args):
        """ Patch for ('sklearn.svm._classes.SVC', 'predict') """

        # pylint: disable=no-method-argument
        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('sklearn.svm._classes.SVC', 'predict')
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

            original_predict = gorilla.get_original_attribute(svm.SVC, 'predict')
            initial_func_predict = partial(original_predict, self, test_data_result)
            optimizer_info_predict, result_predict = capture_optimizer_info(initial_func_predict)
            operator_context_predict = OperatorContext(OperatorType.PREDICT, function_info)
            dag_node_predict = DagNode(singleton.get_next_op_id(),
                                       BasicCodeLocation(caller_filename, lineno),
                                       operator_context_predict,
                                       DagNodeDetails("SVC", [], optimizer_info_predict),
                                       get_optional_code_info_or_none(optional_code_reference,
                                                                      optional_source_code),
                                       processing_func_predict)
            estimator_dag_node = get_dag_node_for_id(self.mlinspect_estimator_node_id)
            function_call_result = FunctionCallResult(result_predict)
            add_dag_node(dag_node_predict, [estimator_dag_node, test_data_node], function_call_result)
            new_result = function_call_result.function_result
            return new_result

        if not call_info_singleton.param_search_active:
            new_result = execute_patched_func_indirect_allowed(execute_inspections)
        else:
            original = gorilla.get_original_attribute(svm.SVC, 'predict')
            new_result = original(self, *args)
        return new_result
