"""
Tests whether the monkey patching works for all patched pandas methods
"""
from functools import partial
from inspect import cleandoc
from types import FunctionType

import networkx
import pandas
from testfixtures import compare, StringComparison, Comparison, RangeComparison

from mlwhatif import OperatorContext, FunctionInfo, OperatorType
from mlwhatif.instrumentation import _pipeline_executor
from mlwhatif.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo, OptimizerInfo


def test_read_csv():
    """
    Tests whether the monkey patching of ('pandas.io.parsers', 'read_csv') works
    """
    test_code = cleandoc("""
        import os
        import pandas as pd
        from mlwhatif.utils import get_project_root
        
        train_file = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_train.csv")
        raw_data = pd.read_csv(train_file, na_values='?', index_col=0)
        assert len(raw_data) == 22792
        """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    extracted_node: DagNode = list(inspector_result.dag.nodes)[0]
    expected_node = DagNode(0,
                            BasicCodeLocation("<string-source>", 6),
                            OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.io.parsers', 'read_csv')),
                            DagNodeDetails(StringComparison(r".*\.csv"),
                                           ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                            'marital-status',
                                            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                            'hours-per-week', 'native-country', 'income-per-year'],
                                           OptimizerInfo(RangeComparison(0, 100000), (22792, 15),
                                                         RangeComparison(0, 30000000))),
                            OptionalCodeInfo(CodeReference(6, 11, 6, 62),
                                             "pd.read_csv(train_file, na_values='?', index_col=0)"),
                            Comparison(partial))
    compare(extracted_node, expected_node)

    assert len(extracted_node.processing_func()) == 22792


def test_frame__init__():
    """
    Tests whether the monkey patching of ('pandas.core.frame', 'DataFrame') works
    """
    test_code = cleandoc("""
        import pandas as pd

        df = pd.DataFrame([0, 1, 2], columns=['A'])
        assert len(df) == 3
        """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    extracted_node: DagNode = list(inspector_result.dag.nodes)[0]

    expected_node = DagNode(0,
                            BasicCodeLocation("<string-source>", 3),
                            OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                            DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 100), (3, 1),
                                                                      RangeComparison(0, 400))),
                            OptionalCodeInfo(CodeReference(3, 5, 3, 43), "pd.DataFrame([0, 1, 2], columns=['A'])"),
                            Comparison(partial))
    compare(extracted_node, expected_node)

    df_created_with_extracted_func = extracted_node.processing_func()
    df_expected = pandas.DataFrame([0, 1, 2], columns=['A'])
    pandas.testing.assert_frame_equal(df_created_with_extracted_func, df_expected)


def test_frame_dropna():
    """
    Tests whether the monkey patching of ('pandas.core.frame', 'dropna') works
    """
    test_code = cleandoc("""
        import pandas as pd
        
        df = pd.DataFrame([0, 2, 4, 5, None], columns=['A'])
        assert len(df) == 5
        df = df.dropna()
        assert len(df) == 4
        """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 3),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(3, 5, 3, 52),
                                                    "pd.DataFrame([0, 2, 4, 5, None], columns=['A'])"),
                                   Comparison(partial))
    expected_select = DagNode(1,
                              BasicCodeLocation("<string-source>", 5),
                              OperatorContext(OperatorType.SELECTION, FunctionInfo('pandas.core.frame', 'dropna')),
                              DagNodeDetails('dropna', ['A']),
                              OptionalCodeInfo(CodeReference(5, 5, 5, 16), 'df.dropna()'),
                              Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_select, arg_index=0)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    extracted_dropna = list(inspector_result.dag.nodes)[1]
    pandas_df = pandas.DataFrame([0, 2, None, None, 4, None], columns=['A'])
    filtered_df = extracted_dropna.processing_func(pandas_df)
    assert len(filtered_df) == 3


def test_frame__getitem__series():
    """
    Tests whether the monkey patching of ('pandas.core.frame', '__getitem__') works for a single string argument
    """
    test_code = cleandoc("""
            import pandas as pd

            df = pd.DataFrame([0, 2, 4, 8, None], columns=['A'])
            a = df['A']
            pd.testing.assert_series_equal(a, pd.Series([0, 2, 4, 8, None], name='A'))
            """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[2])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 3),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(3, 5, 3, 52),
                                                    "pd.DataFrame([0, 2, 4, 8, None], columns=['A'])"),
                                   Comparison(partial))
    expected_project = DagNode(1,
                               BasicCodeLocation("<string-source>", 4),
                               OperatorContext(OperatorType.PROJECTION,
                                               FunctionInfo('pandas.core.frame', '__getitem__')),
                               DagNodeDetails("to ['A']", ['A']),
                               OptionalCodeInfo(CodeReference(4, 4, 4, 11), "df['A']"),
                               Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_project, arg_index=0)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    extracted_getitem = list(inspector_result.dag.nodes)[1]
    pandas_df = pandas.DataFrame({'A': [0, 2, 5], 'B': [1, 3, 5]})
    projected_df = extracted_getitem.processing_func(pandas_df)
    pandas.testing.assert_series_equal(projected_df, pandas.Series([0, 2, 5], name='A'))


def test_frame__getitem__frame():
    """
    Tests whether the monkey patching of ('pandas.core.frame', '__getitem__') works for multiple string arguments
    """
    test_code = cleandoc("""
                import pandas as pd

                df = pd.DataFrame([[0, None, 2], [1, 2, 3], [4, None, 2], [9, 2, 3], [6, 1, 2], [1, 2, 3]], 
                    columns=['A', 'B', 'C'])
                df_projection = df[['A', 'C']]
                df_expected = pd.DataFrame([[0, 2], [1, 3], [4, 2], [9, 3], [6, 2], [1, 3]], columns=['A', 'C'])
                pd.testing.assert_frame_equal(df_projection, df_expected)
                """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[2])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 3),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A', 'B', 'C']),
                                   OptionalCodeInfo(CodeReference(3, 5, 4, 28),
                                                    "pd.DataFrame([[0, None, 2], [1, 2, 3], [4, None, 2], "
                                                    "[9, 2, 3], [6, 1, 2], [1, 2, 3]], \n"
                                                    "    columns=['A', 'B', 'C'])"),
                                   Comparison(partial))
    expected_project = DagNode(1,
                               BasicCodeLocation("<string-source>", 5),
                               OperatorContext(OperatorType.PROJECTION,
                                               FunctionInfo('pandas.core.frame', '__getitem__')),
                               DagNodeDetails("to ['A', 'C']", ['A', 'C']),
                               OptionalCodeInfo(CodeReference(5, 16, 5, 30), "df[['A', 'C']]"),
                               Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_project, arg_index=0)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    extracted_getitem = list(inspector_result.dag.nodes)[1]
    pandas_df = pandas.DataFrame({'A': [0, 2, 5, 7, 1], 'B': [0, 2, 5, 1, 1], 'C': [1, 3, 5, None, None]})
    projected_df = extracted_getitem.processing_func(pandas_df)
    df_expected = pandas.DataFrame({'A': [0, 2, 5, 7, 1], 'C': [1, 3, 5, None, None]})
    pandas.testing.assert_frame_equal(projected_df, df_expected)


def test_frame__getitem__selection():
    """
    Tests whether the monkey patching of ('pandas.core.frame', '__getitem__') works for filtering
    """
    test_code = cleandoc("""
                import pandas as pd

                df = pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 5, 4, 11, None]})
                df_selection = df[df['A'] > 3]
                df_expected = pd.DataFrame({'A': [4, 8, 5], 'B': [4, 11, None]})
                pd.testing.assert_frame_equal(df_selection.reset_index(drop=True), df_expected.reset_index(drop=True))
                """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[4])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 3),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A', 'B']),
                                   OptionalCodeInfo(CodeReference(3, 5, 3, 67),
                                                    "pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 5, 4, 11, None]})"),
                                   Comparison(partial))
    expected_projection = DagNode(1,
                                  BasicCodeLocation("<string-source>", 4),
                                  OperatorContext(OperatorType.PROJECTION,
                                                  FunctionInfo('pandas.core.frame', '__getitem__')),
                                  DagNodeDetails("to ['A']", ['A']),
                                  OptionalCodeInfo(CodeReference(4, 18, 4, 25), "df['A']"),
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_projection, arg_index=0)
    expected_subscript = DagNode(2,
                                 BasicCodeLocation('<string-source>', 4),
                                 OperatorContext(OperatorType.SUBSCRIPT,
                                                 FunctionInfo('pandas.core.series', '_cmp_method')),
                                 DagNodeDetails('> 3', ['A']),
                                 OptionalCodeInfo(CodeReference(4, 18, 4, 29), "df['A'] > 3"),
                                 Comparison(FunctionType))
    expected_dag.add_edge(expected_projection, expected_subscript, arg_index=0)
    expected_selection = DagNode(3,
                                 BasicCodeLocation("<string-source>", 4),
                                 OperatorContext(OperatorType.SELECTION,
                                                 FunctionInfo('pandas.core.frame', '__getitem__')),
                                 DagNodeDetails("Select by Series: df[df['A'] > 3]", ['A', 'B']),
                                 OptionalCodeInfo(CodeReference(4, 15, 4, 30), "df[df['A'] > 3]"),
                                 Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_selection, arg_index=0)
    expected_dag.add_edge(expected_subscript, expected_selection, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    extracted_getitem = list(inspector_result.dag.nodes)[3]
    pandas_df = pandas.DataFrame({'col_a': [0, 2, 4, 8, 5], 'col_b': [1, 5, 4, 11, None]})
    df_selection = pandas_df['col_b'] > 1
    filtered_df = extracted_getitem.processing_func(pandas_df, df_selection)
    df_expected = pandas.DataFrame({'col_a': [2, 4, 8], 'col_b': [5, 4, 11.]})
    pandas.testing.assert_frame_equal(filtered_df.reset_index(drop=True), df_expected.reset_index(drop=True))


def test_frame__setitem__():
    """
    Tests whether the monkey patching of ('pandas.core.frame', '__setitem__') works
    """
    test_code = cleandoc("""
                import pandas as pd

                pandas_df = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                              'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                              'baz': [1, 2, 3, 4, 5, 6],
                              'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
                pandas_df['baz'] = pandas_df['baz'] + 1
                df_expected = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two', 'two'],
                              'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                              'baz': [2, 3, 4, 5, 6, 7],
                              'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
                pd.testing.assert_frame_equal(pandas_df, df_expected)
                """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[4])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 3),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['foo', 'bar', 'baz', 'zoo']),
                                   OptionalCodeInfo(CodeReference(3, 12, 6, 53),
                                                    "pd.DataFrame({'foo': ['one', 'one', 'one', 'two', "
                                                    "'two', 'two'],\n"
                                                    "              'bar': ['A', 'B', 'C', 'A', 'B', 'C'],\n"
                                                    "              'baz': [1, 2, 3, 4, 5, 6],\n"
                                                    "              'zoo': ['x', 'y', 'z', 'q', 'w', 't']})"),
                                   Comparison(partial))
    expected_project = DagNode(1,
                               BasicCodeLocation("<string-source>", 7),
                               OperatorContext(OperatorType.PROJECTION,
                                               FunctionInfo('pandas.core.frame', '__getitem__')),
                               DagNodeDetails("to ['baz']", ['baz']),
                               OptionalCodeInfo(CodeReference(7, 19, 7, 35), "pandas_df['baz']"),
                               Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_project, arg_index=0)
    expected_subscript = DagNode(2,
                                 BasicCodeLocation('<string-source>', 7),
                                 OperatorContext(OperatorType.SUBSCRIPT,
                                                 FunctionInfo('pandas.core.series', '_arith_method')),
                                 DagNodeDetails('+ 1', ['baz']),
                                 OptionalCodeInfo(CodeReference(7, 19, 7, 39), "pandas_df['baz'] + 1"),
                                 Comparison(FunctionType))
    expected_dag.add_edge(expected_project, expected_subscript, arg_index=0)
    expected_project_modify = DagNode(3,
                                      BasicCodeLocation("<string-source>", 7),
                                      OperatorContext(OperatorType.PROJECTION_MODIFY,
                                                      FunctionInfo('pandas.core.frame', '__setitem__')),
                                      DagNodeDetails("modifies ['baz']", ['foo', 'bar', 'baz', 'zoo']),
                                      OptionalCodeInfo(CodeReference(7, 0, 7, 39),
                                                       "pandas_df['baz'] = pandas_df['baz'] + 1"),
                                      Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_project_modify, arg_index=0)
    expected_dag.add_edge(expected_subscript, expected_project_modify, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    extracted_setitem = list(inspector_result.dag.nodes)[3]
    pandas_df = pandas.DataFrame({'foo': ['one', 'two', 'two', 'two', 'two', 'two'],
                                  'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                                  'baz': [1, 2, 3, 4, 5, 6],
                                  'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
    new_values = pandas_df['baz'] * 2
    extracted_setitem.processing_func(pandas_df, new_values)
    df_expected = pandas.DataFrame({'foo': ['one', 'two', 'two', 'two', 'two', 'two'],
                                    'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                                    'baz': [2, 4, 6, 8, 10, 12],
                                    'zoo': ['x', 'y', 'z', 'q', 'w', 't']})
    pandas.testing.assert_frame_equal(pandas_df, df_expected)


def test_frame_replace():
    """
    Tests whether the monkey patching of ('pandas.core.frame', 'replace') works
    """
    test_code = cleandoc("""
        import pandas as pd

        df = pd.DataFrame(['Low', 'Medium', 'Low', 'High', None], columns=['A'])
        df_replace = df.replace('Medium', 'Low')
        df_expected = pd.DataFrame(['Low', 'Low', 'Low', 'High', None], columns=['A'])
        pd.testing.assert_frame_equal(df_replace.reset_index(drop=True), df_expected.reset_index(drop=True))
        """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[2])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 3),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(3, 5, 3, 72),
                                                    "pd.DataFrame(['Low', 'Medium', 'Low', 'High', None], "
                                                    "columns=['A'])"),
                                   Comparison(partial))
    expected_modify = DagNode(1,
                              BasicCodeLocation("<string-source>", 4),
                              OperatorContext(OperatorType.PROJECTION_MODIFY,
                                              FunctionInfo('pandas.core.frame', 'replace')),
                              DagNodeDetails("Replace 'Medium' with 'Low'", ['A']),
                              OptionalCodeInfo(CodeReference(4, 13, 4, 40), "df.replace('Medium', 'Low')"),
                              Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_modify, arg_index=0)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    extracted_replace = list(inspector_result.dag.nodes)[1]
    pandas_df = pandas.DataFrame(['Medium', 'High', 'Medium', 'Low', None], columns=['C'])
    df_replace = extracted_replace.processing_func(pandas_df)
    df_expected = pandas.DataFrame(['Low', 'High', 'Low', 'Low', None], columns=['C'])
    pandas.testing.assert_frame_equal(df_replace.reset_index(drop=True), df_expected.reset_index(drop=True))


def test_frame_merge_on():
    """
    Tests whether the monkey patching of ('pandas.core.frame', 'merge') works
    """
    test_code = cleandoc("""
        import pandas as pd

        df_a = pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 2, 4, 5, 7]})
        df_b = pd.DataFrame({'B': [1, 2, 3, 4, 5], 'C': [1, 5, 4, 11, None]})
        df_merged = df_a.merge(df_b, on='B')
        df_expected = pd.DataFrame({'A': [0, 2, 4, 8], 'B': [1, 2, 4, 5], 'C': [1, 5, 11, None]})
        pd.testing.assert_frame_equal(df_merged.reset_index(drop=True), df_expected.reset_index(drop=True))
        """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[3])

    expected_dag = networkx.DiGraph()
    expected_a = DagNode(0,
                         BasicCodeLocation("<string-source>", 3),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['A', 'B']),
                         OptionalCodeInfo(CodeReference(3, 7, 3, 65),
                                          "pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 2, 4, 5, 7]})"),
                         Comparison(partial))
    expected_b = DagNode(1,
                         BasicCodeLocation("<string-source>", 4),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['B', 'C']),
                         OptionalCodeInfo(CodeReference(4, 7, 4, 69),
                                          "pd.DataFrame({'B': [1, 2, 3, 4, 5], 'C': [1, 5, 4, 11, None]})"),
                         Comparison(partial))
    expected_join = DagNode(2,
                            BasicCodeLocation("<string-source>", 5),
                            OperatorContext(OperatorType.JOIN, FunctionInfo('pandas.core.frame', 'merge')),
                            DagNodeDetails("on 'B'", ['A', 'B', 'C']),
                            OptionalCodeInfo(CodeReference(5, 12, 5, 36), "df_a.merge(df_b, on='B')"),
                            Comparison(FunctionType))
    expected_dag.add_edge(expected_a, expected_join, arg_index=0)
    expected_dag.add_edge(expected_b, expected_join, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    extracted_merge = list(inspector_result.dag.nodes)[2]
    df_a = pandas.DataFrame({'col_a': [0, 20, 4, 8, 5], 'B': [1, 2, 4, 5, 7]})
    df_b = pandas.DataFrame({'B': [10, 2, 30, 4, 5], 'col_c': [10, 5, 4, 11, None]})
    df_merged = extracted_merge.processing_func(df_a, df_b)
    df_expected = pandas.DataFrame({'col_a': [20, 4, 8], 'B': [2, 4, 5], 'col_c': [5, 11, None]})
    pandas.testing.assert_frame_equal(df_merged.reset_index(drop=True), df_expected.reset_index(drop=True))


def test_frame_merge_left_right_on():
    """
    Tests whether the monkey patching of ('pandas.core.frame', 'merge') works
    """
    test_code = cleandoc("""
        import pandas as pd

        df_a = pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 2, 4, 5, 7]})
        df_b = pd.DataFrame({'C': [1, 2, 3, 4, 5], 'D': [1, 5, 4, 11, None]})
        df_merged = df_a.merge(df_b, left_on='B', right_on='C')
        df_expected = pd.DataFrame({'A': [0, 2, 4, 8], 'B': [1, 2, 4, 5],  'C': [1, 2, 4, 5], 'D': [1, 5, 11, None]})
        pd.testing.assert_frame_equal(df_merged.reset_index(drop=True), df_expected.reset_index(drop=True))
        """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[3])

    expected_dag = networkx.DiGraph()
    expected_a = DagNode(0,
                         BasicCodeLocation("<string-source>", 3),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['A', 'B']),
                         OptionalCodeInfo(CodeReference(3, 7, 3, 65),
                                          "pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 2, 4, 5, 7]})"),
                         Comparison(partial))
    expected_b = DagNode(1,
                         BasicCodeLocation("<string-source>", 4),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['C', 'D']),
                         OptionalCodeInfo(CodeReference(4, 7, 4, 69),
                                          "pd.DataFrame({'C': [1, 2, 3, 4, 5], 'D': [1, 5, 4, 11, None]})"),
                         Comparison(partial))
    expected_join = DagNode(2,
                            BasicCodeLocation("<string-source>", 5),
                            OperatorContext(OperatorType.JOIN, FunctionInfo('pandas.core.frame', 'merge')),
                            DagNodeDetails("on 'B' == 'C'", ['A', 'B', 'C', 'D']),
                            OptionalCodeInfo(CodeReference(5, 12, 5, 55),
                                             "df_a.merge(df_b, left_on='B', right_on='C')"),
                            Comparison(FunctionType))
    expected_dag.add_edge(expected_a, expected_join, arg_index=0)
    expected_dag.add_edge(expected_b, expected_join, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    extracted_merge = list(inspector_result.dag.nodes)[2]
    df_a = pandas.DataFrame({'col_a': [0, 20, 4, 8, 5], 'B': [1, 2, 4, 5, 7]})
    df_b = pandas.DataFrame({'C': [10, 2, 30, 4, 5], 'col_d': [10, 5, 4, 11, None]})
    df_merged = extracted_merge.processing_func(df_a, df_b)
    df_expected = pandas.DataFrame({'col_a': [20, 4, 8], 'B': [2, 4, 5], 'C': [2, 4, 5], 'col_d': [5, 11, None]})
    pandas.testing.assert_frame_equal(df_merged.reset_index(drop=True), df_expected.reset_index(drop=True))


def test_frame_merge_index():
    """
    Tests whether the monkey patching of ('pandas.core.frame', 'merge') works
    """
    test_code = cleandoc("""
        import pandas as pd

        df_a = pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 2, 4, 5, 7]})
        df_b = pd.DataFrame({'C': [1, 2, 3, 4], 'D': [1, 5, 4, 11]})
        df_merged = df_a.merge(right=df_b, left_index=True, right_index=True, how='outer')
        df_expected = pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 2, 4, 5, 7],  'C': [1., 2., 3., 4., None], 
            'D': [1., 5., 4., 11., None]})
        pd.testing.assert_frame_equal(df_merged.reset_index(drop=True), df_expected.reset_index(drop=True))
        """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[3])

    expected_dag = networkx.DiGraph()
    expected_a = DagNode(0,
                         BasicCodeLocation("<string-source>", 3),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['A', 'B']),
                         OptionalCodeInfo(CodeReference(3, 7, 3, 65),
                                          "pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [1, 2, 4, 5, 7]})"),
                         Comparison(partial))
    expected_b = DagNode(1,
                         BasicCodeLocation("<string-source>", 4),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['C', 'D']),
                         OptionalCodeInfo(CodeReference(4, 7, 4, 60),
                                          "pd.DataFrame({'C': [1, 2, 3, 4], 'D': [1, 5, 4, 11]})"),
                         Comparison(partial))
    expected_join = DagNode(2,
                            BasicCodeLocation("<string-source>", 5),
                            OperatorContext(OperatorType.JOIN, FunctionInfo('pandas.core.frame', 'merge')),
                            DagNodeDetails('on left_index == right_index (outer)', ['A', 'B', 'C', 'D']),
                            OptionalCodeInfo(CodeReference(5, 12, 5, 82),
                                             "df_a.merge(right=df_b, left_index=True, right_index=True, how='outer')"),
                            Comparison(FunctionType))
    expected_dag.add_edge(expected_a, expected_join, arg_index=0)
    expected_dag.add_edge(expected_b, expected_join, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    extracted_merge = list(inspector_result.dag.nodes)[2]
    df_a = pandas.DataFrame({'col_a': [10, 2, 4, 8, 5], 'col_b': [1, 2, 4, 5, 7]})
    df_b = pandas.DataFrame({'col_c': [1, 2, 3, 4], 'col_d': [1, 5, 4, 11]})
    df_merged = extracted_merge.processing_func(df_a, df_b)
    df_expected = pandas.DataFrame(
        {'col_a': [10, 2, 4, 8, 5], 'col_b': [1, 2, 4, 5, 7], 'col_c': [1., 2., 3., 4., None],
         'col_d': [1., 5., 4., 11., None]})
    pandas.testing.assert_frame_equal(df_merged.reset_index(drop=True), df_expected.reset_index(drop=True))


def test_frame_merge_sorted():
    """
    Tests whether the monkey patching of ('pandas.core.frame', 'merge') works if the sort option is set to True
    """
    test_code = cleandoc("""
        import pandas as pd

        df_a = pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [7, 5, 4, 2, 1]})
        df_b = pd.DataFrame({'B': [1, 4, 3, 2, 5], 'C': [1, 5, 4, 11, None]})
        df_merged = df_a.merge(df_b, on='B', sort=True)
        df_expected = pd.DataFrame({'A': [5, 8, 4, 2], 'B': [1, 2, 4, 5], 'C': [1, 11, 5, None]})
        pd.testing.assert_frame_equal(df_merged, df_expected)
        """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[3])

    expected_dag = networkx.DiGraph()
    expected_a = DagNode(0,
                         BasicCodeLocation("<string-source>", 3),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['A', 'B']),
                         OptionalCodeInfo(CodeReference(3, 7, 3, 65),
                                          "pd.DataFrame({'A': [0, 2, 4, 8, 5], 'B': [7, 5, 4, 2, 1]})"),
                         Comparison(partial))
    expected_b = DagNode(1,
                         BasicCodeLocation("<string-source>", 4),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['B', 'C']),
                         OptionalCodeInfo(CodeReference(4, 7, 4, 69),
                                          "pd.DataFrame({'B': [1, 4, 3, 2, 5], 'C': [1, 5, 4, 11, None]})"),
                         Comparison(partial))
    expected_join = DagNode(2,
                            BasicCodeLocation("<string-source>", 5),
                            OperatorContext(OperatorType.JOIN, FunctionInfo('pandas.core.frame', 'merge')),
                            DagNodeDetails("on 'B'", ['A', 'B', 'C']),
                            OptionalCodeInfo(CodeReference(5, 12, 5, 47), "df_a.merge(df_b, on='B', sort=True)"),
                            Comparison(FunctionType))
    expected_dag.add_edge(expected_a, expected_join, arg_index=0)
    expected_dag.add_edge(expected_b, expected_join, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    extracted_merge = list(inspector_result.dag.nodes)[2]
    df_a = pandas.DataFrame({'col_a': [0, 20, 4, 8, 5], 'B': [1, 2, 4, 5, 7]})
    df_b = pandas.DataFrame({'B': [10, 2, 30, 5, 4], 'col_c': [10, 5, 4, 11, None]})
    df_merged = extracted_merge.processing_func(df_a, df_b)
    df_expected = pandas.DataFrame({'col_a': [20, 4, 8], 'B': [2, 4, 5], 'col_c': [5, None, 11]})
    pandas.testing.assert_frame_equal(df_merged.reset_index(drop=True), df_expected.reset_index(drop=True))


def test_groupby_agg():
    """
    Tests whether the monkey patching of ('pandas.core.frame', 'groupby') and ('pandas.core.groupbygeneric', 'agg')
    works.
    """
    test_code = cleandoc("""
        import pandas as pd

        df = pd.DataFrame({'group': ['A', 'B', 'A', 'C', 'B'], 'value': [1, 2, 1, 3, 4]})
        df_groupby_agg = df.groupby('group').agg(mean_value=('value', 'mean'))
        
        df_expected = pd.DataFrame({'group': ['A', 'B', 'C'], 'mean_value': [1, 3, 3]})
        pd.testing.assert_frame_equal(df_groupby_agg.reset_index(drop=False), df_expected.reset_index(drop=True))
        """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[2])

    expected_dag = networkx.DiGraph()
    expected_data = DagNode(0,
                            BasicCodeLocation("<string-source>", 3),
                            OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                            DagNodeDetails(None, ['group', 'value']),
                            OptionalCodeInfo(CodeReference(3, 5, 3, 81),
                                             "pd.DataFrame({'group': ['A', 'B', 'A', 'C', 'B'], "
                                             "'value': [1, 2, 1, 3, 4]})"),
                            Comparison(partial))
    expected_groupby_agg = DagNode(1,
                                   BasicCodeLocation("<string-source>", 4),
                                   OperatorContext(OperatorType.GROUP_BY_AGG,
                                                   FunctionInfo('pandas.core.groupby.generic', 'agg')),
                                   DagNodeDetails("Groupby 'group', Aggregate: '{'mean_value': ('value', 'mean')}'",
                                                  ['group', 'mean_value']),
                                   OptionalCodeInfo(CodeReference(4, 17, 4, 70),
                                                    "df.groupby('group').agg(mean_value=('value', 'mean'))"),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_data, expected_groupby_agg, arg_index=0)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    pandas_df = pandas.DataFrame({'group': ['A', 'B', 'A', 'B'], 'value': [1, 2, 7, 4]})
    extracted_node_groupby_agg = list(inspector_result.dag.nodes)[1]
    df_groupby_agg = extracted_node_groupby_agg.processing_func(pandas_df)
    df_expected = pandas.DataFrame({'group': ['A', 'B'], 'mean_value': [4, 3]})
    pandas.testing.assert_frame_equal(df_groupby_agg.reset_index(drop=False), df_expected.reset_index(drop=True))


def test_series__init__():
    """
    Tests whether the monkey patching of ('pandas.core.series', 'Series') works
    """
    test_code = cleandoc("""
        import pandas as pd

        pd_series = pd.Series([0, 2, 4, None], name='A')
        assert len(pd_series) == 4
        """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    extracted_node: DagNode = list(inspector_result.dag.nodes)[0]

    expected_node = DagNode(0,
                            BasicCodeLocation("<string-source>", 3),
                            OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.series', 'Series')),
                            DagNodeDetails(None, ['A']),
                            OptionalCodeInfo(CodeReference(3, 12, 3, 48), "pd.Series([0, 2, 4, None], name='A')"),
                            Comparison(partial))
    compare(extracted_node, expected_node)

    extracted_func_result = extracted_node.processing_func()
    expected = pandas.Series([0, 2, 4, None], name='A')
    pandas.testing.assert_series_equal(extracted_func_result, expected)


def test_series_isin():
    """
    Tests whether the monkey patching of ('pandas.core.series', 'isin') works
    """
    test_code = cleandoc("""
        import pandas as pd

        pd_series = pd.Series([0, 2, 4, None], name='A')
        filtered = pd_series.isin([2, 4])
        expected = pd.Series([False, True, True, False], name='A')
        pd.testing.assert_series_equal(filtered.reset_index(drop=True), expected.reset_index(drop=True))
        """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    extracted_dag = inspector_result.dag
    extracted_dag.remove_node(list(extracted_dag.nodes)[2])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 3),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.series', 'Series')),
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(3, 12, 3, 48),
                                                    "pd.Series([0, 2, 4, None], name='A')"),
                                   Comparison(partial))
    expected_isin = DagNode(1,
                            BasicCodeLocation("<string-source>", 4),
                            OperatorContext(OperatorType.SUBSCRIPT,
                                            FunctionInfo('pandas.core.series', 'isin')),
                            DagNodeDetails('isin: [2, 4]', ['A']),
                            OptionalCodeInfo(CodeReference(4, 11, 4, 33),
                                             'pd_series.isin([2, 4])'),
                            Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_isin, arg_index=0)

    compare(extracted_dag, expected_dag)

    extracted_node = list(extracted_dag.nodes)[1]
    pd_series = pandas.Series([2, 2, 4, 0], name='A')
    extracted_func_result = extracted_node.processing_func(pd_series)
    expected = pandas.Series([True, True, True, False], name='A')
    pandas.testing.assert_series_equal(extracted_func_result.reset_index(drop=True), expected.reset_index(drop=True))


def test_series__cmp_method():
    """
    Tests whether the monkey patching of ('pandas.core.series', '_cmp_method') works
    """
    test_code = cleandoc("""
                import pandas as pd

                pd_series = pd.Series([0, 2, 4, None], name='A')
                mask = pd_series > 3
                pd.testing.assert_series_equal(mask, pd.Series([False, False, True, False], name='A'))
                """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[2])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 3),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.series', 'Series')),
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(3, 12, 3, 48),
                                                    "pd.Series([0, 2, 4, None], name='A')"),
                                   Comparison(partial))
    expected_projection = DagNode(1,
                                  BasicCodeLocation("<string-source>", 4),
                                  OperatorContext(OperatorType.SUBSCRIPT,
                                                  FunctionInfo('pandas.core.series', '_cmp_method')),
                                  DagNodeDetails('> 3', ['A']),
                                  OptionalCodeInfo(CodeReference(4, 7, 4, 20), 'pd_series > 3'),
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_projection, arg_index=0)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    extracted_node = list(inspector_result.dag.nodes)[1]
    pd_series = pandas.Series([4, 2, 4, None], name='B')
    extracted_func_result = extracted_node.processing_func(pd_series)
    expected = pandas.Series([True, False, True, False], name='B')
    pandas.testing.assert_series_equal(extracted_func_result.reset_index(drop=True), expected.reset_index(drop=True))


def test_series__arith_method():
    """
    Tests whether the monkey patching of ('pandas.core.series', '_arith_method') works
    """
    test_code = cleandoc("""
                import pandas as pd
                pd_series = pd.Series([0, 2, 4, None], name='A')
                pd_series = pd_series + 2
                pd.testing.assert_series_equal(pd_series, pd.Series([2, 4, 6, None], name='A'))
                """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[2])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 2),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.series', 'Series')),
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(2, 12, 2, 48),
                                                    "pd.Series([0, 2, 4, None], name='A')"),
                                   Comparison(partial))
    expected_projection = DagNode(1,
                                  BasicCodeLocation("<string-source>", 3),
                                  OperatorContext(OperatorType.SUBSCRIPT,
                                                  FunctionInfo('pandas.core.series', '_arith_method')),
                                  DagNodeDetails('+ 2', ['A']),
                                  OptionalCodeInfo(CodeReference(3, 12, 3, 25), 'pd_series + 2'),
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_projection, arg_index=0)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    extracted_node = list(inspector_result.dag.nodes)[1]
    pd_series = pandas.Series([0, 2, 8, None], name='B')
    extracted_func_result = extracted_node.processing_func(pd_series)
    expected = pandas.Series([2, 4, 10, None], name='B')
    pandas.testing.assert_series_equal(extracted_func_result.reset_index(drop=True), expected.reset_index(drop=True))


def test_series__logical_method():
    """
    Tests whether the monkey patching of ('pandas.core.series', 'test_series__logical_method') works
    """
    test_code = cleandoc("""
                import pandas as pd
                mask1 = pd.Series([True, False, True, True], name='A')
                mask2 = pd.Series([True, False, False, True], name='B')
                mask3 = mask1 & mask2
                pd.testing.assert_series_equal(mask3, pd.Series([True, False, False, True], name=None))
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[3])

    expected_dag = networkx.DiGraph()
    expected_data_source1 = DagNode(0,
                                    BasicCodeLocation("<string-source>", 2),
                                    OperatorContext(OperatorType.DATA_SOURCE,
                                                    FunctionInfo('pandas.core.series', 'Series')),
                                    DagNodeDetails(None, ['A']),
                                    OptionalCodeInfo(CodeReference(2, 8, 2, 54),
                                                     "pd.Series([True, False, True, True], name='A')"),
                                    Comparison(partial))
    expected_data_source2 = DagNode(1,
                                    BasicCodeLocation("<string-source>", 3),
                                    OperatorContext(OperatorType.DATA_SOURCE,
                                                    FunctionInfo('pandas.core.series', 'Series')),
                                    DagNodeDetails(None, ['B']),
                                    OptionalCodeInfo(CodeReference(3, 8, 3, 55),
                                                     "pd.Series([True, False, False, True], name='B')"),
                                    Comparison(partial))
    expected_subscript = DagNode(2,
                                 BasicCodeLocation("<string-source>", 4),
                                 OperatorContext(OperatorType.SUBSCRIPT,
                                                 FunctionInfo('pandas.core.series', '_logical_method')),
                                 DagNodeDetails('&', ['A']),
                                 OptionalCodeInfo(CodeReference(4, 8, 4, 21), 'mask1 & mask2'),
                                 Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source1, expected_subscript, arg_index=0)
    expected_dag.add_edge(expected_data_source2, expected_subscript, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    extracted_node = list(inspector_result.dag.nodes)[2]
    pd_series1 = pandas.Series([True, False, True, True], name='C')
    pd_series2 = pandas.Series([False, False, False, True], name='D')
    extracted_func_result = extracted_node.processing_func(pd_series1, pd_series2)
    expected = pandas.Series([False, False, False, True], name=None)
    pandas.testing.assert_series_equal(extracted_func_result.reset_index(drop=True), expected.reset_index(drop=True))
