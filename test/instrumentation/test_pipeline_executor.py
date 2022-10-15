"""
Tests whether the PipelineExecutor works
"""
import ast
from functools import partial
from inspect import cleandoc
from types import FunctionType

import astunparse
import networkx
from testfixtures import compare, Comparison, RangeComparison

from mlmq import OperatorType, OperatorContext, FunctionInfo
from mlmq.execution import _pipeline_executor
from mlmq.instrumentation._dag_node import CodeReference, DagNode, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo, OptimizerInfo
from mlmq.execution._pipeline_executor import singleton
from mlmq.testing._testing_helper_utils import get_test_code_with_function_def_and_for_loop


def test_func_defs_and_loops():
    """
    Tests whether the monkey patching of pandas function works
    """
    test_code = get_test_code_with_function_def_and_for_loop()

    extracted_dag = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True).original_dag

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 4),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 800), (2, 1),
                                                                             RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(4, 9, 4, 44), "pd.DataFrame([0, 1], columns=['A'])"),
                                   Comparison(partial))
    expected_select_1 = DagNode(1,
                                BasicCodeLocation("<string-source>", 8),
                                OperatorContext(OperatorType.SELECTION, FunctionInfo('pandas.core.frame', 'dropna')),
                                DagNodeDetails('dropna', ['A'], OptimizerInfo(RangeComparison(0, 800), (2, 1),
                                                                              RangeComparison(0, 800))),
                                OptionalCodeInfo(CodeReference(8, 9, 8, 20), 'df.dropna()'),
                                Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_select_1, arg_index=0)
    expected_select_2 = DagNode(2,
                                BasicCodeLocation("<string-source>", 8),
                                OperatorContext(OperatorType.SELECTION, FunctionInfo('pandas.core.frame', 'dropna')),
                                DagNodeDetails('dropna', ['A'], OptimizerInfo(RangeComparison(0, 800), (2, 1),
                                                                              RangeComparison(0, 800))),
                                OptionalCodeInfo(CodeReference(8, 9, 8, 20), 'df.dropna()'),
                                Comparison(FunctionType))
    expected_dag.add_edge(expected_select_1, expected_select_2, arg_index=0)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))


def test_func_defs_and_loops_without_code_reference_tracking():
    """
    Tests whether the monkey patching of pandas function works
    """
    test_code = get_test_code_with_function_def_and_for_loop()

    extracted_dag = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=False).original_dag

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 4),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 800), (2, 1),
                                                                             RangeComparison(0, 800))),
                                   processing_func=Comparison(partial))
    expected_select_1 = DagNode(1,
                                BasicCodeLocation("<string-source>", 8),
                                OperatorContext(OperatorType.SELECTION, FunctionInfo('pandas.core.frame', 'dropna')),
                                DagNodeDetails('dropna', ['A'], OptimizerInfo(RangeComparison(0, 800), (2, 1),
                                                                              RangeComparison(0, 800))),
                                processing_func=Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_select_1, arg_index=0)
    expected_select_2 = DagNode(2,
                                BasicCodeLocation("<string-source>", 8),
                                OperatorContext(OperatorType.SELECTION, FunctionInfo('pandas.core.frame', 'dropna')),
                                DagNodeDetails('dropna', ['A'], OptimizerInfo(RangeComparison(0, 800), (2, 1),
                                                                              RangeComparison(0, 800))),
                                processing_func=Comparison(FunctionType))
    expected_dag.add_edge(expected_select_1, expected_select_2, arg_index=0)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))


def test_annotation_storage():
    """
    Tests whether the monkey patching of pandas function works
    """
    test_code = cleandoc("""
        import pandas
        df = pandas.DataFrame([["x", "y"], ["2", "3"]], columns=["a", "b"])
        assert df._mlinspect_dag_node is not None
        """)

    _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)


def test_black_box_operation():
    """
    Tests whether the monkey patching of pandas function works
    """
    test_code = cleandoc("""
        import pandas
        from mlmq.testing._testing_helper_utils import black_box_df_op
        
        df = black_box_df_op()
        df = df.dropna()
        print("df")
        """)

    extracted_dag = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True).original_dag

    expected_dag = networkx.DiGraph()
    expected_missing_op = DagNode(-1,
                                  BasicCodeLocation("<string-source>", 5),
                                  OperatorContext(OperatorType.MISSING_OP, None),
                                  DagNodeDetails('Warning! Operator <string-source>:5 (df.dropna()) encountered a '
                                                 'DataFrame resulting from an operation without mlmq support!',
                                                 ['A'], OptimizerInfo(None, (5, 1), RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(5, 5, 5, 16), 'df.dropna()'))
    expected_select = DagNode(0,
                              BasicCodeLocation("<string-source>", 5),
                              OperatorContext(OperatorType.SELECTION, FunctionInfo('pandas.core.frame', 'dropna')),
                              DagNodeDetails('dropna', ['A'], OptimizerInfo(RangeComparison(0, 800), (5, 1),
                                                                            RangeComparison(0, 800))),
                              OptionalCodeInfo(CodeReference(5, 5, 5, 16), 'df.dropna()'),
                              Comparison(FunctionType))
    expected_dag.add_edge(expected_missing_op, expected_select, arg_index=0)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))


def test_instrument_pipeline_with_code_reference_tracking():
    """
    Tests whether the instrumentation modifies user code as expected with code reference tracking
    """
    test_code = get_test_code_with_function_def_and_for_loop()
    parsed_ast = ast.parse(test_code)
    parsed_modified_ast = singleton.instrument_pipeline(parsed_ast, True)
    instrumented_code = astunparse.unparse(parsed_modified_ast)
    expected_code = cleandoc("""
            from mlmq.execution._pipeline_executor import set_code_reference_call, set_code_reference_subscript, monkey_patch, undo_monkey_patch
            monkey_patch()
            import pandas as pd
            
            def black_box_df_op():
                df = pd.DataFrame([0, 1], **set_code_reference_call(4, 9, 4, 44, columns=['A']))
                return df
            df = black_box_df_op(**set_code_reference_call(6, 5, 6, 22))
            for _ in range(2, **set_code_reference_call(7, 9, 7, 17)):
                df = df.dropna(**set_code_reference_call(8, 9, 8, 20))
            undo_monkey_patch()
            """)
    compare(cleandoc(instrumented_code), expected_code)


def test_instrument_pipeline_with_code_reference_tracking_comparison():
    """
    Tests whether the instrumentation modifies user code as expected with code reference tracking
    """
    test_code = cleandoc("""
            import pandas as pd
            pd_series = pd.Series([0, 2, 4, None], name='A')
            mask = pd_series > 3
            """)
    parsed_ast = ast.parse(test_code)
    parsed_modified_ast = singleton.instrument_pipeline(parsed_ast, True)
    instrumented_code = astunparse.unparse(parsed_modified_ast)
    expected_code = cleandoc("""
            from mlmq.execution._pipeline_executor import set_code_reference_call, set_code_reference_subscript, monkey_patch, undo_monkey_patch
            monkey_patch()
            import pandas as pd
            pd_series = pd.Series([0, 2, 4, None], **set_code_reference_call(2, 12, 2, 48, name='A'))
            mask = (pd_series > set_code_reference_subscript(3, 7, 3, 20, 3))
            undo_monkey_patch()
            """)
    compare(cleandoc(instrumented_code), expected_code)


def test_instrument_pipeline_with_code_reference_tracking_bin_op():
    """
    Tests whether the instrumentation modifies user code as expected with code reference tracking
    """
    test_code = cleandoc("""
            import pandas as pd
            pd_series = pd.Series([0, 2, 4, None], name='A')
            pd_series = pd_series + 2
            """)
    parsed_ast = ast.parse(test_code)
    parsed_modified_ast = singleton.instrument_pipeline(parsed_ast, True)
    instrumented_code = astunparse.unparse(parsed_modified_ast)
    expected_code = cleandoc("""
            from mlmq.execution._pipeline_executor import set_code_reference_call, set_code_reference_subscript, monkey_patch, undo_monkey_patch
            monkey_patch()
            import pandas as pd
            pd_series = pd.Series([0, 2, 4, None], **set_code_reference_call(2, 12, 2, 48, name='A'))
            pd_series = (pd_series + set_code_reference_subscript(3, 12, 3, 25, 2))
            undo_monkey_patch()
            """)
    compare(cleandoc(instrumented_code), expected_code)


def test_instrument_pipeline_with_code_reference_tracking_bool_op():
    """
    Tests whether the instrumentation modifies user code as expected with code reference tracking
    """
    test_code = cleandoc("""
            import pandas as pd
            mask1 = pd.Series([True, False, True, None], name='A')
            mask2 = pd.Series([True, False, False, None], name='B')
            mask3 = mask1 and mask2
            """)
    parsed_ast = ast.parse(test_code)
    parsed_modified_ast = singleton.instrument_pipeline(parsed_ast, True)
    instrumented_code = astunparse.unparse(parsed_modified_ast)
    expected_code = cleandoc("""
            from mlmq.execution._pipeline_executor import set_code_reference_call, set_code_reference_subscript, monkey_patch, undo_monkey_patch
            monkey_patch()
            import pandas as pd
            mask1 = pd.Series([True, False, True, None], **set_code_reference_call(2, 8, 2, 54, name='A'))
            mask2 = pd.Series([True, False, False, None], **set_code_reference_call(3, 8, 3, 55, name='B'))
            mask3 = (mask1 and set_code_reference_subscript(4, 8, 4, 23, mask2))
            undo_monkey_patch()
            """)
    compare(cleandoc(instrumented_code), expected_code)


def test_instrument_pipeline_without_code_reference_tracking():
    """
    Tests whether the instrumentation modifies user code as expected without code reference tracking
    """
    test_code = get_test_code_with_function_def_and_for_loop()
    parsed_ast = ast.parse(test_code)
    parsed_modified_ast = singleton.instrument_pipeline(parsed_ast, False)
    instrumented_code = astunparse.unparse(parsed_modified_ast)
    expected_code = cleandoc("""
            from mlmq.execution._pipeline_executor import set_code_reference_call, set_code_reference_subscript, monkey_patch, undo_monkey_patch
            monkey_patch()
            import pandas as pd

            def black_box_df_op():
                df = pd.DataFrame([0, 1], columns=['A'])
                return df
            df = black_box_df_op()
            for _ in range(2):
                df = df.dropna()
            undo_monkey_patch()
            """)
    compare(cleandoc(instrumented_code), expected_code)
