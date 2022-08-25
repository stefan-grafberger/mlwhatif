"""
Tests whether the monkey patching works for all patched pandas methods
"""
from functools import partial
from inspect import cleandoc
from types import FunctionType

import networkx
import pandas
from testfixtures import compare, Comparison, RangeComparison

from mlwhatif import OperatorContext, FunctionInfo, OperatorType
from mlwhatif.instrumentation import _pipeline_executor
from mlwhatif.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo, OptimizerInfo


def test_frame_fuzzy_merge_on():
    """
    Tests whether the monkey patching of ('pandas.core.frame', 'merge') works
    """
    test_code = cleandoc("""
        import pandas as pd
        import fuzzy_pandas as fpd

        df_a = pd.DataFrame({'A': [0, 2, 4, 8, 5], 'name': ['George Smiley', 'Oliver Lacon', 'Toby Esterhase', 
            'Jim Prideaux', 'Peter Guillam']})
        df_b = pd.DataFrame({'B': [1, 2, 3, 4, 5], 'person_name': ['Peter Guillam', 'Oliver LACON', 'George SMILEY', 
            'Claus Kretzschmar', 'Konny Saks']})
        df_merged = fpd.fuzzy_merge(df_a, df_b, left_on='name', right_on='person_name', method='levenshtein')
        df_expected = pd.DataFrame({'A': [0, 2, 5], 'name': ['George Smiley', 'Oliver Lacon', 'Peter Guillam'], 
            'B': [3, 2, 1], 'person_name': ['George SMILEY', 'Oliver LACON', 'Peter Guillam']})
        pd.testing.assert_frame_equal(df_merged.reset_index(drop=True), df_expected.reset_index(drop=True))
        """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.original_dag.remove_node(list(inspector_result.original_dag.nodes)[3])

    expected_dag = networkx.DiGraph()
    expected_a = DagNode(0,
                         BasicCodeLocation("<string-source>", 4),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['A', 'name'], OptimizerInfo(RangeComparison(0, 100), (5, 2),
                                                                           RangeComparison(0, 500))),
                         OptionalCodeInfo(CodeReference(4, 7, 5, 38),
                                          "pd.DataFrame({'A': [0, 2, 4, 8, 5], 'name': ['George Smiley', "
                                          "'Oliver Lacon', 'Toby Esterhase', \n    'Jim Prideaux', 'Peter Guillam']})"),
                         Comparison(partial))
    expected_b = DagNode(1,
                         BasicCodeLocation("<string-source>", 6),
                         OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                         DagNodeDetails(None, ['B', 'person_name'], OptimizerInfo(RangeComparison(0, 100), (5, 2),
                                                                                  RangeComparison(0, 500))),
                         OptionalCodeInfo(CodeReference(6, 7, 7, 40),
                                          "pd.DataFrame({'B': [1, 2, 3, 4, 5], 'person_name': ['Peter Guillam', "
                                          "'Oliver LACON', 'George SMILEY', \n    "
                                          "'Claus Kretzschmar', 'Konny Saks']})"),
                         Comparison(partial))
    expected_join = DagNode(2,
                            BasicCodeLocation("<string-source>", 8),
                            OperatorContext(OperatorType.JOIN, FunctionInfo('fuzzy_pandas.fuzzy_merge', 'fuzzy_merge')),
                            DagNodeDetails("on 'name' ~~ 'person_name'", ['A', 'name', 'B', 'person_name'],
                                           OptimizerInfo(RangeComparison(0, 200), (3, 4), RangeComparison(0, 800))),
                            OptionalCodeInfo(CodeReference(8, 12, 8, 101),
                                             "fpd.fuzzy_merge(df_a, df_b, left_on='name', right_on='person_name', "
                                             "method='levenshtein')"),
                            Comparison(FunctionType))
    expected_dag.add_edge(expected_a, expected_join, arg_index=0)
    expected_dag.add_edge(expected_b, expected_join, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    extracted_merge = list(inspector_result.original_dag.nodes)[2]
    df_a = pandas.DataFrame({'A': [0, 2, 4], 'name': ['George Smiley', 'Oliver Lacon', 'Toby Esterhase']})
    df_b = pandas.DataFrame({'B': [1, 2, 3, 4], 'person_name': ['Peter Guillam', 'Oliver LACON', 'George SMILEY',
                                                                'Claus Kretzschmar']})
    df_merged = extracted_merge.processing_func(df_a, df_b)
    df_expected = pandas.DataFrame({'A': [0, 2], 'name': ['George Smiley', 'Oliver Lacon'],
                                    'B': [3, 2], 'person_name': ['George SMILEY', 'Oliver LACON']})
    pandas.testing.assert_frame_equal(df_merged.reset_index(drop=True), df_expected.reset_index(drop=True))
