"""
Monkey patching for pandas
"""
# pylint: disable=too-many-lines
import operator
import os
from functools import partial

import gorilla
import pandas

from mlwhatif import OperatorType, DagNode, BasicCodeLocation, DagNodeDetails
from mlwhatif.instrumentation._operator_types import OperatorContext, FunctionInfo
from mlwhatif.instrumentation._pipeline_executor import singleton
from mlwhatif.monkeypatching._monkey_patching_utils import execute_patched_func, get_input_info, add_dag_node, \
    get_dag_node_for_id, execute_patched_func_no_op_id, get_optional_code_info_or_none, FunctionCallResult, \
    execute_patched_internal_func_with_depth
from mlwhatif.monkeypatching._patch_sklearn import call_info_singleton


@gorilla.patches(pandas)
class PandasPatching:
    """ Patches for pandas """

    # pylint: disable=too-few-public-methods

    @gorilla.name('read_csv')
    @gorilla.settings(allow_hit=True)
    def patched_read_csv(*args, **kwargs):
        """ Patch for ('pandas.io.parsers', 'read_csv') """
        # pylint: disable=no-method-argument
        original = gorilla.get_original_attribute(pandas, 'read_csv')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.io.parsers', 'read_csv')

            operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
            result = original(*args, **kwargs)
            processing_func = partial(original, *args, **kwargs)

            description = "{}".format(args[0].split(os.path.sep)[-1])
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, list(result.columns)),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                               processing_func)
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [], function_call_result)
            new_result = function_call_result.function_result
            return new_result

        return execute_patched_func(original, execute_inspections, *args, **kwargs)


@gorilla.patches(pandas.DataFrame)
class DataFramePatching:
    """ Patches for 'pandas.core.frame' """

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', 'DataFrame') """
        original = gorilla.get_original_attribute(pandas.DataFrame, '__init__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.core.frame', 'DataFrame')
            operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
            original(self, *args, **kwargs)
            result = self

            columns = list(self.columns)  # pylint: disable=no-member
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(None, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [], function_call_result)

        execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('dropna')
    @gorilla.settings(allow_hit=True)
    def patched_dropna(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', 'dropna') """
        original = gorilla.get_original_attribute(pandas.DataFrame, 'dropna')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.core.frame', 'dropna')

            input_info = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)
            operator_context = OperatorContext(OperatorType.SELECTION, function_info)
            # No input_infos copy needed because it's only a selection and the rows not being removed don't change
            processing_func = lambda df: original(df, *args[1:], **kwargs)
            result = original(input_info.annotated_dfobject.result_data, *args[1:], **kwargs)
            if result is None:
                raise NotImplementedError("TODO: Support inplace dropna")
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("dropna", list(result.columns)),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                               processing_func)
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result

            return new_result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__getitem__')
    @gorilla.settings(allow_hit=True)
    def patched__getitem__(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', '__getitem__') """
        original = gorilla.get_original_attribute(pandas.DataFrame, '__getitem__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('pandas.core.frame', '__getitem__')
            input_info = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)
            dag_parents = [input_info.dag_node]
            processing_func = lambda df: original(df, *args, **kwargs)
            if isinstance(args[0], str):  # Projection to Series
                columns = [args[0]]
                operator_context = OperatorContext(OperatorType.PROJECTION, function_info)
                dag_node = DagNode(op_id,
                                   BasicCodeLocation(caller_filename, lineno),
                                   operator_context,
                                   DagNodeDetails("to {}".format(columns), columns),
                                   get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                                   processing_func)
            elif isinstance(args[0], list) and isinstance(args[0][0], str):  # Projection to DF
                columns = args[0]
                operator_context = OperatorContext(OperatorType.PROJECTION, function_info)
                dag_node = DagNode(op_id,
                                   BasicCodeLocation(caller_filename, lineno),
                                   operator_context,
                                   DagNodeDetails("to {}".format(columns), columns),
                                   get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                                   processing_func)
            elif isinstance(args[0], pandas.Series):  # Selection
                operator_context = OperatorContext(OperatorType.SELECTION, function_info)
                columns = list(self.columns)  # pylint: disable=no-member
                selection_series_input_info = get_input_info(args[0], caller_filename, lineno, function_info,
                                                             optional_code_reference, optional_source_code)
                # FIXME: Add test to make sure that this DAG node is included as a parent correctly
                dag_parents.append(selection_series_input_info.dag_node)
                if optional_source_code:
                    description = "Select by Series: {}".format(optional_source_code)
                else:
                    description = "Select by Series"
                dag_node = DagNode(op_id,
                                   BasicCodeLocation(caller_filename, lineno),
                                   operator_context,
                                   DagNodeDetails(description, columns),
                                   get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                                   processing_func)
            else:
                raise NotImplementedError()
            result = original(input_info.annotated_dfobject.result_data, *args, **kwargs)
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, dag_parents, function_call_result)
            new_result = function_call_result.function_result

            return new_result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('__setitem__')
    @gorilla.settings(allow_hit=True)
    def patched__setitem__(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', '__setitem__') """
        original = gorilla.get_original_attribute(pandas.DataFrame, '__setitem__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('pandas.core.frame', '__setitem__')
            operator_context = OperatorContext(OperatorType.PROJECTION_MODIFY, function_info)

            input_info_self = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                             optional_source_code)
            dag_node_parents = [input_info_self.dag_node]
            if isinstance(args[1], pandas.Series):
                input_info_other = get_input_info(args[1], caller_filename, lineno, function_info, optional_code_reference,
                                                  optional_source_code)
                dag_node_parents.append(input_info_other.dag_node)
                processing_func = lambda df, new_val: original(df, args[0], new_val, *args[2:], **kwargs)
            else:
                processing_func = lambda df: original(df, *args, **kwargs)
            if isinstance(args[0], str):
                result = original(self, *args, **kwargs)
                columns = list(self.columns)  # pylint: disable=no-member
                description = "modifies {}".format([args[0]])
            else:
                raise NotImplementedError("TODO: Handling __setitem__ for key type {}".format(type(args[0])))
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                               processing_func)

            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, dag_node_parents, function_call_result)
            new_result = function_call_result.function_result
            assert hasattr(self, "_mlinspect_dag_node")
            self._mlinspect_dag_node = op_id  # pylint: disable=attribute-defined-outside-init
            return new_result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('replace')
    @gorilla.settings(allow_hit=True)
    def patched_replace(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', 'replace') """
        original = gorilla.get_original_attribute(pandas.DataFrame, 'replace')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.core.frame', 'replace')

            input_info = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)
            operator_context = OperatorContext(OperatorType.PROJECTION_MODIFY, function_info)
            # No input_infos copy needed because it's only a selection and the rows not being removed don't change
            result = original(input_info.annotated_dfobject.result_data, *args, **kwargs)
            if isinstance(args[0], dict):
                raise NotImplementedError("TODO: Add support for replace with dicts")
            description = "Replace '{}' with '{}'".format(args[0], args[1])
            processing_func = lambda df: original(df, *args, **kwargs)
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, list(result.columns)),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                               processing_func)

            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result

            return new_result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('merge')
    @gorilla.settings(allow_hit=True)
    def patched_merge(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', 'merge') """
        original = gorilla.get_original_attribute(pandas.DataFrame, 'merge')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('pandas.core.frame', 'merge')

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
            result = original(input_info_a.annotated_dfobject.result_data,
                              input_info_b.annotated_dfobject.result_data,
                              *args[args_start_index:],
                              **kwargs)
            description = self.get_merge_description(**kwargs)
            processing_func = lambda df_a, df_b: original(df_a, df_b, *args[args_start_index:], **kwargs)
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, list(result.columns)),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code),
                               processing_func)
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [input_info_a.dag_node, input_info_b.dag_node], function_call_result)
            new_result = function_call_result.function_result

            return new_result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @staticmethod
    def get_merge_description(**kwargs):
        """Get the description for a pd.merge call"""
        if 'on' in kwargs:
            description = f"on '{kwargs['on']}'"
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
            description = f"on {left_column} == {right_column}"
        else:
            description = None
        if 'how' in kwargs and description is not None:
            description += f" ({kwargs['how']})"
        elif 'how' in kwargs:
            description = f"({kwargs['how']})"
        return description

    @gorilla.name('groupby')
    @gorilla.settings(allow_hit=True)
    def patched_groupby(self, *args, **kwargs):
        """ Patch for ('pandas.core.frame', 'groupby') """
        original = gorilla.get_original_attribute(pandas.DataFrame, 'groupby')

        def execute_inspections(_, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.core.frame', 'groupby')
            # We ignore groupbys, we only do something with aggs

            input_info = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)
            result = original(self, *args, **kwargs)
            result._mlinspect_dag_node = input_info.dag_node.node_id  # pylint: disable=protected-access

            return result

        return execute_patched_func_no_op_id(original, execute_inspections, self, *args, **kwargs)


@gorilla.patches(pandas.core.groupby.generic.DataFrameGroupBy)
class DataFrameGroupByPatching:
    """ Patches for 'pandas.core.groupby.generic' """

    # pylint: disable=too-few-public-methods

    @gorilla.name('agg')
    @gorilla.settings(allow_hit=True)
    def patched_agg(self, *args, **kwargs):
        """ Patch for ('pandas.core.groupby.generic', 'agg') """
        original = gorilla.get_original_attribute(pandas.core.groupby.generic.DataFrameGroupBy, 'agg')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.core.groupby.generic', 'agg')
            if not hasattr(self, '_mlinspect_dag_node'):
                raise NotImplementedError("TODO: Support agg if groupby happened in external code")
            input_dag_node = get_dag_node_for_id(self._mlinspect_dag_node)  # pylint: disable=no-member

            operator_context = OperatorContext(OperatorType.GROUP_BY_AGG, function_info)

            result = original(self, *args, **kwargs)

            if len(args) > 0:
                description = "Groupby '{}', Aggregate: '{}'".format(result.index.name, args)
            else:
                description = "Groupby '{}', Aggregate: '{}'".format(result.index.name, kwargs)
            columns = [result.index.name] + list(result.columns)
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [input_dag_node], function_call_result)
            new_result = function_call_result.function_result

            return new_result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)


@gorilla.patches(pandas.core.indexing._LocIndexer)  # pylint: disable=protected-access
class LocIndexerPatching:
    """ Patches for 'pandas.core.series' """

    # pylint: disable=too-few-public-methods, too-many-locals

    @gorilla.name('__getitem__')
    @gorilla.settings(allow_hit=True)
    def patched__getitem__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', 'Series') """
        original = gorilla.get_original_attribute(
            pandas.core.indexing._LocIndexer, '__getitem__')  # pylint: disable=protected-access

        if call_info_singleton.column_transformer_active:
            op_id = singleton.get_next_op_id()
            caller_filename = call_info_singleton.transformer_filename
            lineno = call_info_singleton.transformer_lineno
            function_info = call_info_singleton.transformer_function_info
            optional_code_reference = call_info_singleton.transformer_optional_code_reference
            optional_source_code = call_info_singleton.transformer_optional_source_code

            if isinstance(args[0], tuple) and not args[0][0].start and not args[0][0].stop \
                    and isinstance(args[0][1], list) and isinstance(args[0][1][0], str):
                # Projection to one or multiple columns, return value is df
                columns = args[0][1]
            elif isinstance(args[0], tuple) and not args[0][0].start and not args[0][0].stop \
                    and isinstance(args[0][1], str):
                # Projection to one column with str syntax, e.g., for HashingVectorizer
                columns = [args[0][1]]
            else:
                raise NotImplementedError()

            operator_context = OperatorContext(OperatorType.PROJECTION, function_info)
            input_info = get_input_info(self.obj, caller_filename,  # pylint: disable=no-member
                                        lineno, function_info, optional_code_reference, optional_source_code)
            result = original(self, *args, **kwargs)

            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails("to {}".format(columns), columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result
        else:
            new_result = original(self, *args, **kwargs)

        return new_result


@gorilla.patches(pandas.Series)
class SeriesPatching:
    """ Patches for 'pandas.core.series' """

    # pylint: disable=too-few-public-methods

    @gorilla.name('__init__')
    @gorilla.settings(allow_hit=True)
    def patched__init__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', 'Series') """
        original = gorilla.get_original_attribute(pandas.Series, '__init__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.core.series', 'Series')

            operator_context = OperatorContext(OperatorType.DATA_SOURCE, function_info)
            original(self, *args, **kwargs)
            result = self

            if self.name:  # pylint: disable=no-member
                columns = list(self.name)  # pylint: disable=no-member
            else:
                columns = ["_1"]
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(None, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [], function_call_result)

        execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('isin')
    @gorilla.settings(allow_hit=True)
    def patched_isin(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', 'isin') """
        original = gorilla.get_original_attribute(pandas.Series, 'isin')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            function_info = FunctionInfo('pandas.core.series', 'isin')
            input_info = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)
            operator_context = OperatorContext(OperatorType.SUBSCRIPT, function_info)
            description = "isin: {}".format(args[0])
            columns = [self.name]  # pylint: disable=no-member
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))

            result = original(input_info.annotated_dfobject.result_data, *args, **kwargs)
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result

            return new_result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('_cmp_method')
    @gorilla.settings(allow_hit=True)
    def patched_cmp_method(self, other, cmp_op):
        """ Patch for ('pandas.core.series', '_cmp_method') """
        original = gorilla.get_original_attribute(pandas.Series, '_cmp_method')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals

            function_info = FunctionInfo('pandas.core.series', '_cmp_method')
            input_info_self = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                             optional_source_code)
            dag_node_parents = [input_info_self.dag_node]
            operator_context = OperatorContext(OperatorType.SUBSCRIPT, function_info)
            if cmp_op == operator.eq:  # pylint: disable=comparison-with-callable
                description = "="
            elif cmp_op == operator.ne:  # pylint: disable=comparison-with-callable
                description = "!="
            elif cmp_op == operator.gt:  # pylint: disable=comparison-with-callable
                description = ">"
            elif cmp_op == operator.ge:  # pylint: disable=comparison-with-callable
                description = ">="
            elif cmp_op == operator.lt:  # pylint: disable=comparison-with-callable
                description = "<"
            elif cmp_op == operator.le:  # pylint: disable=comparison-with-callable
                description = "<="
            else:
                print(f"Operation {cmp_op} is not supported yet!")
                assert False
            if not isinstance(other, pandas.Series):
                if isinstance(other, str):
                    description += f" '{other}'"
                else:
                    description += f" {other}"
            else:
                input_info_other = get_input_info(other, caller_filename, lineno, function_info,
                                                  optional_code_reference,
                                                  optional_source_code)
                dag_node_parents.append(input_info_other.dag_node)
            # TODO: Pandas uses a function 'get_op_result_name' to construct the new name, this can also be
            #  None sometimes. If these names are actually important for something, revisit this columns code line.
            columns = [self.name]  # pylint: disable=no-member
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))

            result = original(self, other, cmp_op)
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, dag_node_parents, function_call_result)
            new_result = function_call_result.function_result

            return new_result

        return execute_patched_internal_func_with_depth(original, execute_inspections, 4, self, other, cmp_op)

    @gorilla.name('_logical_method')
    @gorilla.settings(allow_hit=True)
    def patched_logical__method(self, other, logical_op):
        """ Patch for ('pandas.core.series', '_logical_method') """
        original = gorilla.get_original_attribute(pandas.Series, '_logical_method')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('pandas.core.series', '_logical_method')
            input_info_self = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                             optional_source_code)
            input_info_other = get_input_info(other, caller_filename, lineno, function_info, optional_code_reference,
                                              optional_source_code)
            operator_context = OperatorContext(OperatorType.SUBSCRIPT, function_info)
            if logical_op == operator.and_:  # pylint: disable=comparison-with-callable
                description = "&"
            elif logical_op == operator.or_:  # pylint: disable=comparison-with-callable
                description = "|"
            elif logical_op == operator.xor:  # pylint: disable=comparison-with-callable
                description = "^"
            else:
                print(f"Operation {logical_op} is not supported yet!")
                assert False

            # TODO: Pandas uses a function 'get_op_result_name' to construct the new name, this can also be
            #  None sometimes. If these names are actually important for something, revisit this columns code line.
            columns = [self.name]  # pylint: disable=no-member
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))

            result = original(self, other, logical_op)
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [input_info_self.dag_node, input_info_other.dag_node], function_call_result)
            new_result = function_call_result.function_result

            return new_result

        return execute_patched_internal_func_with_depth(original, execute_inspections, 4, self, other, logical_op)

    @gorilla.name('__not__')
    @gorilla.settings(allow_hit=True)
    def patched__not__(self, *args, **kwargs):
        """ Patch for ('pandas.core.series', '__not__') """
        original = gorilla.get_original_attribute(pandas.Series, '__not__')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('pandas.core.series', '__not__')
            input_info = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                        optional_source_code)
            operator_context = OperatorContext(OperatorType.SUBSCRIPT, function_info)
            description = "!= {}".format(args[0])
            columns = [self.name]  # pylint: disable=no-member
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))

            result = original(input_info.annotated_dfobject.result_data, *args, **kwargs)
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, [input_info.dag_node], function_call_result)
            new_result = function_call_result.function_result

            return new_result

        return execute_patched_func(original, execute_inspections, self, *args, **kwargs)

    @gorilla.name('_arith_method')
    @gorilla.settings(allow_hit=True)
    def patched__arith_method(self, other, arith_op):
        """ Patch for ('pandas.core.series', '_arith_method') """
        original = gorilla.get_original_attribute(pandas.Series, '_arith_method')

        def execute_inspections(op_id, caller_filename, lineno, optional_code_reference, optional_source_code):
            """ Execute inspections, add DAG node """
            # pylint: disable=too-many-locals
            function_info = FunctionInfo('pandas.core.series', '_arith_method')
            input_info_self = get_input_info(self, caller_filename, lineno, function_info, optional_code_reference,
                                             optional_source_code)
            dag_node_parents = [input_info_self.dag_node]
            operator_context = OperatorContext(OperatorType.SUBSCRIPT, function_info)
            if arith_op == operator.add or arith_op.__name__ == "radd":  # pylint: disable=comparison-with-callable
                description = "+"
            elif arith_op == operator.sub or arith_op.__name__ == "rsub":  # pylint: disable=comparison-with-callable
                description = "-"
            elif arith_op == operator.mul or arith_op.__name__ == "rmul":  # pylint: disable=comparison-with-callable
                description = "*"
            elif arith_op == operator.truediv:  # pylint: disable=comparison-with-callable
                description = "/"
            elif arith_op.__name__ == "rtruediv":
                description = "/"
            else:
                print(f"Operation {arith_op} is not supported yet!")
                assert False
            if not isinstance(other, pandas.Series):
                if isinstance(other, str):
                    description += f" '{other}'"
                else:
                    description += f" {other}"
            else:
                input_info_other = get_input_info(other, caller_filename, lineno, function_info,
                                                  optional_code_reference,
                                                  optional_source_code)
                dag_node_parents.append(input_info_other.dag_node)
            # TODO: Pandas uses a function 'get_op_result_name' to construct the new name, this can also be
            #  None sometimes. If these names are actually important for something, revisit this columns code line.
            columns = [self.name]  # pylint: disable=no-member
            dag_node = DagNode(op_id,
                               BasicCodeLocation(caller_filename, lineno),
                               operator_context,
                               DagNodeDetails(description, columns),
                               get_optional_code_info_or_none(optional_code_reference, optional_source_code))

            result = original(self, other, arith_op)
            function_call_result = FunctionCallResult(result)
            add_dag_node(dag_node, dag_node_parents, function_call_result)
            new_result = function_call_result.function_result

            return new_result

        return execute_patched_internal_func_with_depth(original, execute_inspections, 4, self, other, arith_op)
