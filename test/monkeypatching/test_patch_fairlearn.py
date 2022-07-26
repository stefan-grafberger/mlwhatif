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
from test.monkeypatching.test_patch_sklearn import filter_dag_for_nodes_with_ids


def test_metric_frame___init__():
    """
    Tests whether the monkey patching of ('pandas.core.series', 'test_series__logical_method') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from fairlearn.metrics import MetricFrame, false_negative_rate
                
                predictions = pd.Series([True, False, True, True], name='A')
                labels = pd.Series([True, False, False, True], name='B')
                sensitive_features = pd.DataFrame(['cat_a', 'cat_a', 'cat_b', 'cat_a'], columns=['cat_col'])
                
                fnr_by_group = MetricFrame(metrics=false_negative_rate,
                                           y_pred=predictions,
                                           y_true=labels,
                                           sensitive_features=sensitive_features)
                fnr_by_group = fnr_by_group.by_group.reset_index(drop=False)
                expected = pd.DataFrame({'cat_col': ['cat_a', 'cat_b'], 'false_negative_rate': [0.0, 0.0]})
                expected['false_negative_rate'] = expected['false_negative_rate'].astype(object)
                pd.testing.assert_frame_equal(fnr_by_group, expected, atol=1.0)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    filter_dag_for_nodes_with_ids(inspector_result, {0, 1, 2, 3, 4}, 9)

    expected_dag = networkx.DiGraph()
    expected_data_source1 = DagNode(0,
                                    BasicCodeLocation("<string-source>", 4),
                                    OperatorContext(OperatorType.DATA_SOURCE,
                                                    FunctionInfo('pandas.core.series', 'Series')),
                                    DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                              RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(4, 14, 4, 60),
                                                     "pd.Series([True, False, True, True], name='A')"),
                                    Comparison(partial))
    expected_data_source2 = DagNode(1,
                                    BasicCodeLocation("<string-source>", 5),
                                    OperatorContext(OperatorType.DATA_SOURCE,
                                                    FunctionInfo('pandas.core.series', 'Series')),
                                    DagNodeDetails(None, ['B'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                              RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(5, 9, 5, 56),
                                                     "pd.Series([True, False, False, True], name='B')"),
                                    Comparison(partial))
    expected_data_source3 = DagNode(2,
                                    BasicCodeLocation("<string-source>", 6),
                                    OperatorContext(OperatorType.DATA_SOURCE,
                                                    FunctionInfo('pandas.core.frame', 'DataFrame')),
                                    DagNodeDetails(None, ['cat_col'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                    RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(6, 21, 6, 92),
                                                     "pd.DataFrame(['cat_a', 'cat_a', 'cat_b', 'cat_a'], "
                                                     "columns=['cat_col'])"),
                                    Comparison(partial))
    expected_test_labels = DagNode(3,
                                   BasicCodeLocation("<string-source>", 8),
                                   OperatorContext(OperatorType.TEST_LABELS,
                                                   FunctionInfo('fairlearn.metrics._metric_frame', 'MetricFrame')),
                                   DagNodeDetails(None, ['B'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                             RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(8, 15, 11, 65),
                                                    'MetricFrame(metrics=false_negative_rate,\n'
                                                    '                           y_pred=predictions,\n'
                                                    '                           y_true=labels,\n'
                                                    '                           sensitive_features=sensitive_features)'),
                                   Comparison(FunctionType))
    expected_score = DagNode(4,
                             BasicCodeLocation("<string-source>", 8),
                             OperatorContext(OperatorType.SCORE,
                                             FunctionInfo('fairlearn.metrics._metric_frame', 'MetricFrame')),
                             DagNodeDetails('false_negative_rate', [], OptimizerInfo(RangeComparison(0, 200), (1, 1),
                                                                                     RangeComparison(0, 800))),
                             OptionalCodeInfo(CodeReference(8, 15, 11, 65),
                                              'MetricFrame(metrics=false_negative_rate,\n'
                                              '                           y_pred=predictions,\n'
                                              '                           y_true=labels,\n'
                                              '                           sensitive_features=sensitive_features)'),
                             Comparison(partial))
    expected_dag.add_edge(expected_data_source2, expected_test_labels, arg_index=0)
    expected_dag.add_edge(expected_data_source1, expected_score, arg_index=0)
    expected_dag.add_edge(expected_test_labels, expected_score, arg_index=1)
    expected_dag.add_edge(expected_data_source3, expected_score, arg_index=2)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    extracted_node = list(inspector_result.dag.nodes)[4]
    pd_series1 = pandas.Series([True, False, True, True], name='C')
    pd_series2 = pandas.Series([False, False, False, True], name='D')
    pd_series3 = pandas.Series(['cat_a', 'cat_a', 'cat_a', 'cat_c'], name='sensitive')
    extracted_func_result = extracted_node.processing_func(pd_series1, pd_series2, pd_series3)
    actual_fnr_by_group = extracted_func_result.by_group.reset_index(drop=False)
    expected = pandas.DataFrame({'sensitive': ['cat_a', 'cat_c'], 'false_negative_rate': [0.0, 0.0]})
    expected['false_negative_rate'] = expected['false_negative_rate'].astype(object)
    pandas.testing.assert_frame_equal(actual_fnr_by_group, expected, atol=1.0)


def test_equalized_odds_difference():
    """
    Tests whether the monkey patching of ('pandas.core.series', 'test_series__logical_method') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from fairlearn.metrics import equalized_odds_difference

                predictions = pd.Series([True, False, True, True], name='A')
                labels = pd.Series([True, False, False, True], name='B')
                sensitive_features = pd.DataFrame(['cat_a', 'cat_a', 'cat_b', 'cat_a'], columns=['cat_col'])

                metric = equalized_odds_difference(y_pred=predictions, y_true=labels, 
                                                   sensitive_features=sensitive_features)
                assert 0.0 <= metric <= 1.0
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    expected_dag = networkx.DiGraph()
    expected_data_source1 = DagNode(0,
                                    BasicCodeLocation("<string-source>", 4),
                                    OperatorContext(OperatorType.DATA_SOURCE,
                                                    FunctionInfo('pandas.core.series', 'Series')),
                                    DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                              RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(4, 14, 4, 60),
                                                     "pd.Series([True, False, True, True], name='A')"),
                                    Comparison(partial))
    expected_data_source2 = DagNode(1,
                                    BasicCodeLocation("<string-source>", 5),
                                    OperatorContext(OperatorType.DATA_SOURCE,
                                                    FunctionInfo('pandas.core.series', 'Series')),
                                    DagNodeDetails(None, ['B'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                              RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(5, 9, 5, 56),
                                                     "pd.Series([True, False, False, True], name='B')"),
                                    Comparison(partial))
    expected_data_source3 = DagNode(2,
                                    BasicCodeLocation("<string-source>", 6),
                                    OperatorContext(OperatorType.DATA_SOURCE,
                                                    FunctionInfo('pandas.core.frame', 'DataFrame')),
                                    DagNodeDetails(None, ['cat_col'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                    RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(6, 21, 6, 92),
                                                     "pd.DataFrame(['cat_a', 'cat_a', 'cat_b', 'cat_a'], "
                                                     "columns=['cat_col'])"),
                                    Comparison(partial))
    expected_test_labels = DagNode(3,
                                   BasicCodeLocation("<string-source>", 8),
                                   OperatorContext(OperatorType.TEST_LABELS,
                                                   FunctionInfo('fairlearn.metrics._disparities',
                                                                'equalized_odds_difference')),
                                   DagNodeDetails(None, ['B'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                             RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(8, 9, 9, 73),
                                                    'equalized_odds_difference(y_pred=predictions, y_true=labels, \n'
                                                    '                                   sensitive_features='
                                                    'sensitive_features)'),
                                   Comparison(FunctionType))
    expected_score = DagNode(4,
                             BasicCodeLocation("<string-source>", 8),
                             OperatorContext(OperatorType.SCORE,
                                             FunctionInfo('fairlearn.metrics._disparities',
                                                          'equalized_odds_difference')),
                             DagNodeDetails('equalized_odds_difference', [],
                                            OptimizerInfo(RangeComparison(0, 200), (1, 1),
                                                          RangeComparison(0, 800))),
                             OptionalCodeInfo(CodeReference(8, 9, 9, 73),
                                              'equalized_odds_difference(y_pred=predictions, y_true=labels, \n'
                                              '                                   sensitive_features='
                                              'sensitive_features)'),
                             Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source2, expected_test_labels, arg_index=0)
    expected_dag.add_edge(expected_data_source1, expected_score, arg_index=0)
    expected_dag.add_edge(expected_test_labels, expected_score, arg_index=1)
    expected_dag.add_edge(expected_data_source3, expected_score, arg_index=2)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    extracted_node = list(inspector_result.dag.nodes)[4]
    pd_series1 = pandas.Series([True, False, True, True], name='C')
    pd_series2 = pandas.Series([False, False, False, True], name='D')
    pd_series3 = pandas.Series(['cat_a', 'cat_a', 'cat_a', 'cat_c'], name='sensitive')
    extracted_func_result = extracted_node.processing_func(pd_series1, pd_series2, pd_series3)
    assert 0.0 <= extracted_func_result <= 1.0
