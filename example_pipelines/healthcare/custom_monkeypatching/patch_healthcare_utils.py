"""
Monkey patching for healthcare_utils
"""
from functools import partial

import gorilla

from example_pipelines.healthcare import healthcare_utils
from example_pipelines.healthcare import _gensim_wrapper
from mlwhatif.execution._stat_tracking import capture_optimizer_info
from mlwhatif.instrumentation._operator_types import OperatorContext, FunctionInfo, OperatorType
from mlwhatif.instrumentation._dag_node import DagNode, BasicCodeLocation, DagNodeDetails
from mlwhatif.execution._pipeline_executor import singleton
from mlwhatif.monkeypatching._monkey_patching_utils import add_dag_node, \
    get_input_info, execute_patched_func_no_op_id, get_optional_code_info_or_none, FunctionCallResult, \
    wrap_in_mlinspect_array_if_necessary, get_dag_node_for_id
from mlwhatif.monkeypatching._mlinspect_ndarray import MlinspectNdarray


class SklearnMyW2VTransformerPatching:
    """ Patches for healthcare_utils.MyW2VTransformer"""

    # pylint: disable=too-few-public-methods

    @gorilla.patch(_gensim_wrapper.W2VTransformer, name='__init__', settings=gorilla.Settings(allow_hit=True))
    def patched__init__(self, *, size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, sample=1e-3, seed=1,
                        workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=hash, iter=5,
                        null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000,
                        mlinspect_caller_filename=None, mlinspect_lineno=None,
                        mlinspect_optional_code_reference=None, mlinspect_optional_source_code=None,
                        mlinspect_fit_transform_active=False):
        """ Patch for ('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer') """
        # pylint: disable=no-method-argument, attribute-defined-outside-init, too-many-locals, redefined-builtin,
        # pylint: disable=invalid-name
        original = gorilla.get_original_attribute(_gensim_wrapper.W2VTransformer, '__init__')

        self.mlinspect_caller_filename = mlinspect_caller_filename
        self.mlinspect_lineno = mlinspect_lineno
        self.mlinspect_optional_code_reference = mlinspect_optional_code_reference
        self.mlinspect_optional_source_code = mlinspect_optional_source_code
        self.mlinspect_fit_transform_active = mlinspect_fit_transform_active

        self.mlinspect_non_data_func_args = {'hashfxn': hashfxn, 'size': size, 'alpha': alpha, 'window': window,
                                             'min_count': min_count, 'max_vocab_size': max_vocab_size, 'sample': sample,
                                             'seed': seed, 'workers': workers, 'min_alpha': min_alpha, 'sg': sg,
                                             'hs': hs, 'negative': negative, 'cbow_mean': cbow_mean, 'iter': iter,
                                             'null_word': null_word, 'trim_rule': trim_rule,
                                             'sorted_vocab': sorted_vocab, 'batch_words': batch_words}

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            original(self, **self.mlinspect_non_data_func_args)

            self.mlinspect_caller_filename = caller_filename
            self.mlinspect_lineno = lineno
            self.mlinspect_optional_code_reference = optional_code_reference
            self.mlinspect_optional_source_code = optional_source_code

        return execute_patched_func_no_op_id(original, execute_inspections, self, **self.mlinspect_non_data_func_args)

    @gorilla.patch(healthcare_utils.MyW2VTransformer, name='fit_transform', settings=gorilla.Settings(allow_hit=True))
    def patched_fit_transform(self, *args, **kwargs):
        """ Patch for ('example_pipelines.healthcare.healthcare_utils.MyW2VTransformer', 'fit_transform') """
        # pylint: disable=no-method-argument
        self.mlinspect_fit_transform_active = True  # pylint: disable=attribute-defined-outside-init
        original = gorilla.get_original_attribute(healthcare_utils.MyW2VTransformer, 'fit_transform')
        function_info = FunctionInfo('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer')
        input_info = get_input_info(args[0], self.mlinspect_caller_filename, self.mlinspect_lineno, function_info,
                                    self.mlinspect_optional_code_reference, self.mlinspect_optional_source_code)

        def processing_func(input_df):
            transformer = healthcare_utils.MyW2VTransformer(**self.mlinspect_non_data_func_args)
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
                           DagNodeDetails("Word2Vec: fit_transform", ['array'], optimizer_info),
                           get_optional_code_info_or_none(self.mlinspect_optional_code_reference,
                                                          self.mlinspect_optional_source_code),
                           processing_func)
        function_call_result = FunctionCallResult(result)
        add_dag_node(dag_node, [input_info.dag_node], function_call_result)
        new_result = function_call_result.function_result
        assert isinstance(new_result, MlinspectNdarray)
        self.mlinspect_fit_transform_active = False  # pylint: disable=attribute-defined-outside-init
        return new_result

    @gorilla.patch(healthcare_utils.MyW2VTransformer, name='transform', settings=gorilla.Settings(allow_hit=True))
    def patched_transform(self, *args, **kwargs):
        """ Patch for ('example_pipelines.healthcare.healthcare_utils.MyW2VTransformer', 'transform') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(healthcare_utils.MyW2VTransformer, 'transform')
        if not self.mlinspect_fit_transform_active:
            function_info = FunctionInfo('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer')
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
                               DagNodeDetails("Word2Vec: transform", ['array'], optimizer_info),
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
