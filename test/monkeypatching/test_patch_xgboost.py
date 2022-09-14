"""
Tests whether the monkey patching works for all patched sklearn methods
"""
# pylint: disable=too-many-lines
from functools import partial
from inspect import cleandoc
from types import FunctionType

import networkx
import pandas
from sklearn.preprocessing import label_binarize
from testfixtures import compare, Comparison, RangeComparison
from xgboost import XGBClassifier

from mlwhatif import OperatorType, OperatorContext, FunctionInfo
from mlwhatif.execution import _pipeline_executor
from mlwhatif.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo, OptimizerInfo
from test.monkeypatching.test_patch_sklearn import filter_dag_for_nodes_with_ids


def test_xgbclassifier():
    """
    Tests whether the monkey patching of ('sklearn.tree._classes', 'DecisionTreeClassifier') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from xgboost import XGBClassifier
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = XGBClassifier(max_depth=12, tree_method='hist')
                clf = clf.fit(train, target)

                test_predict = clf.predict(pd.DataFrame([[0., 0.], [0.6, 0.6]], columns=['A', 'B']))
                print(test_predict)
                expected = np.array([0., 0.])
                assert np.allclose(test_predict, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.original_dag.remove_node(list(inspector_result.original_dag.nodes)[10])
    inspector_result.original_dag.remove_node(list(inspector_result.original_dag.nodes)[9])
    inspector_result.original_dag.remove_node(list(inspector_result.original_dag.nodes)[8])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A', 'B', 'target'],
                                                  OptimizerInfo(RangeComparison(0, 200), (4, 3),
                                                                RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(6, 5, 6, 95),
                                                    "pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], "
                                                    "'target': ['no', 'no', 'yes', 'yes']})"),
                                   Comparison(partial))
    expected_data_projection = DagNode(1,
                                       BasicCodeLocation("<string-source>", 8),
                                       OperatorContext(OperatorType.PROJECTION,
                                                       FunctionInfo('pandas.core.frame', '__getitem__')),
                                       DagNodeDetails("to ['A', 'B']", ['A', 'B'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(8, 39, 8, 53), "df[['A', 'B']]"),
                                       Comparison(FunctionType))
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 8),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                       DagNodeDetails('Standard Scaler: fit_transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                    RangeComparison(0, 10000))),
                                       OptionalCodeInfo(CodeReference(8, 8, 8, 24), 'StandardScaler()'),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_data_projection, arg_index=0)
    expected_dag.add_edge(expected_data_projection, expected_standard_scaler, arg_index=0)
    expected_label_projection = DagNode(3,
                                        BasicCodeLocation("<string-source>", 9),
                                        OperatorContext(OperatorType.PROJECTION,
                                                        FunctionInfo('pandas.core.frame', '__getitem__')),
                                        DagNodeDetails("to ['target']", ['target'],
                                                       OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                     RangeComparison(0, 800))),
                                        OptionalCodeInfo(CodeReference(9, 24, 9, 36), "df['target']"),
                                        Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_label_projection, arg_index=0)
    expected_label_encode = DagNode(4,
                                    BasicCodeLocation("<string-source>", 9),
                                    OperatorContext(OperatorType.PROJECTION_MODIFY,
                                                    FunctionInfo('sklearn.preprocessing._label', 'label_binarize')),
                                    DagNodeDetails("label_binarize, classes: ['no', 'yes']", ['array'],
                                                   OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                 RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(9, 9, 9, 60),
                                                     "label_binarize(df['target'], classes=['no', 'yes'])"),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_label_projection, expected_label_encode, arg_index=0)
    expected_train_data = DagNode(5,
                                  BasicCodeLocation("<string-source>", 11),
                                  OperatorContext(OperatorType.TRAIN_DATA,
                                                  FunctionInfo('xgboost.sklearn', 'XGBClassifier')),
                                  DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                                RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 53),
                                                   "XGBClassifier(max_depth=12, tree_method='hist')"),
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_standard_scaler, expected_train_data, arg_index=0)
    expected_train_labels = DagNode(6,
                                    BasicCodeLocation("<string-source>", 11),
                                    OperatorContext(OperatorType.TRAIN_LABELS,
                                                    FunctionInfo('xgboost.sklearn', 'XGBClassifier')),
                                    DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                  RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(11, 6, 11, 53),
                                                     "XGBClassifier(max_depth=12, tree_method='hist')"),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_train_labels, arg_index=0)
    expected_decision_tree = DagNode(7,
                                     BasicCodeLocation("<string-source>", 11),
                                     OperatorContext(OperatorType.ESTIMATOR,
                                                     FunctionInfo('xgboost.sklearn', 'XGBClassifier')),
                                     DagNodeDetails('XGB Classifier', [],
                                                    OptimizerInfo(RangeComparison(0, 1000), None,
                                                                  RangeComparison(0, 10000))),
                                     OptionalCodeInfo(CodeReference(11, 6, 11, 53),
                                                      "XGBClassifier(max_depth=12, tree_method='hist')"),
                                     Comparison(FunctionType),
                                     Comparison(partial))
    expected_dag.add_edge(expected_train_data, expected_decision_tree, arg_index=0)
    expected_dag.add_edge(expected_train_labels, expected_decision_tree, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.original_dag.nodes)[7]
    train_data_node = list(inspector_result.original_dag.nodes)[5]
    train_label_node = list(inspector_result.original_dag.nodes)[6]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_data = train_data_node.processing_func(train_df[['C', 'D']])
    train_labels = label_binarize(train_df['target'], classes=['no', 'yes'])
    train_labels = train_label_node.processing_func(train_labels)
    fitted_estimator = fit_node.processing_func(train_data, train_labels)
    assert isinstance(fitted_estimator, XGBClassifier)
    assert isinstance(fit_node.make_classifier_func(), XGBClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_score = fitted_estimator.score(test_df[['C', 'D']], test_labels)
    assert test_score == 0.5


def test_xgbclassifier_score():
    """
    Tests whether the monkey patching of ('sklearn.tree._classes.DecisionTreeClassifier', 'score') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from xgboost import XGBClassifier
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = XGBClassifier(max_depth=12, tree_method='hist')
                clf = clf.fit(train, target)

                test_df = pd.DataFrame({'A': [0., 0.6], 'B':  [0., 0.6], 'target': ['no', 'yes']})
                test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
                test_score = clf.score(test_df[['A', 'B']], test_labels)
                assert test_score == 0.5
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    filter_dag_for_nodes_with_ids(inspector_result, {7, 10, 11, 12, 13, 14, 15}, 16)

    expected_dag = networkx.DiGraph()
    expected_data_projection = DagNode(11,
                                       BasicCodeLocation("<string-source>", 16),
                                       OperatorContext(OperatorType.PROJECTION,
                                                       FunctionInfo('pandas.core.frame', '__getitem__')),
                                       DagNodeDetails("to ['A', 'B']", ['A', 'B'],
                                                      OptimizerInfo(RangeComparison(0, 200), (2, 2),
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(16, 23, 16, 42), "test_df[['A', 'B']]"),
                                       Comparison(FunctionType))
    expected_test_data = DagNode(12,
                                 BasicCodeLocation("<string-source>", 16),
                                 OperatorContext(OperatorType.TEST_DATA,
                                                 FunctionInfo('xgboost.sklearn.XGBClassifier', 'score')),
                                 DagNodeDetails(None, ['A', 'B'], OptimizerInfo(RangeComparison(0, 200), (2, 2),
                                                                                RangeComparison(0, 800))),
                                 OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                                  "clf.score(test_df[['A', 'B']], test_labels)"),
                                 Comparison(FunctionType))
    expected_dag.add_edge(expected_data_projection, expected_test_data, arg_index=0)
    expected_label_encode = DagNode(10,
                                    BasicCodeLocation("<string-source>", 15),
                                    OperatorContext(OperatorType.PROJECTION_MODIFY,
                                                    FunctionInfo('sklearn.preprocessing._label', 'label_binarize')),
                                    DagNodeDetails("label_binarize, classes: ['no', 'yes']", ['array'],
                                                   OptimizerInfo(RangeComparison(0, 200), (2, 1),
                                                                 RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(15, 14, 15, 70),
                                                     "label_binarize(test_df['target'], classes=['no', 'yes'])"),
                                    Comparison(FunctionType))
    expected_test_labels = DagNode(13,
                                   BasicCodeLocation("<string-source>", 16),
                                   OperatorContext(OperatorType.TEST_LABELS,
                                                   FunctionInfo('xgboost.sklearn.XGBClassifier', 'score')),
                                   DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (2, 1),
                                                                                 RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                                    "clf.score(test_df[['A', 'B']], test_labels)"),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_test_labels, arg_index=0)
    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 11),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('xgboost.sklearn', 'XGBClassifier')),
                                  DagNodeDetails('XGB Classifier', [], OptimizerInfo(RangeComparison(0, 1000), None,
                                                                                     RangeComparison(0, 10000))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 53),
                                                   "XGBClassifier(max_depth=12, tree_method='hist')"),
                                  Comparison(FunctionType),
                                  Comparison(partial))
    expected_predict = DagNode(14,
                               BasicCodeLocation("<string-source>", 16),
                               OperatorContext(OperatorType.PREDICT,
                                               FunctionInfo('xgboost.sklearn.XGBClassifier', 'score')),
                               DagNodeDetails('XGB Classifier', [], OptimizerInfo(RangeComparison(0, 1000), (2, 1),
                                                                                  RangeComparison(0, 800))),
                               OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                                "clf.score(test_df[['A', 'B']], test_labels)"),
                               Comparison(FunctionType))
    expected_dag.add_edge(expected_classifier, expected_predict, arg_index=0)
    expected_dag.add_edge(expected_test_data, expected_predict, arg_index=1)

    expected_score = DagNode(15,
                             BasicCodeLocation("<string-source>", 16),
                             OperatorContext(OperatorType.SCORE,
                                             FunctionInfo('xgboost.sklearn.XGBClassifier', 'score')),
                             DagNodeDetails('Accuracy', [], OptimizerInfo(RangeComparison(0, 1000), (1, 1),
                                                                          RangeComparison(0, 800))),
                             OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                              "clf.score(test_df[['A', 'B']], test_labels)"),
                             Comparison(FunctionType))
    expected_dag.add_edge(expected_predict, expected_score, arg_index=0)
    expected_dag.add_edge(expected_test_labels, expected_score, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.original_dag.nodes)[0]
    predict_node = list(inspector_result.original_dag.nodes)[5]
    score_node = list(inspector_result.original_dag.nodes)[6]
    test_data_node = list(inspector_result.original_dag.nodes)[3]
    test_label_node = list(inspector_result.original_dag.nodes)[4]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_labels = label_binarize(train_df['target'], classes=['no', 'yes'])
    fitted_estimator = fit_node.processing_func(train_df[['C', 'D']], train_labels)
    assert isinstance(fitted_estimator, XGBClassifier)
    assert isinstance(fit_node.make_classifier_func(), XGBClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_labels = test_label_node.processing_func(test_labels)
    test_predictions = predict_node.processing_func(fitted_estimator, test_data)
    test_score = score_node.processing_func(test_predictions, test_labels)
    assert test_score == 0.5


def test_xgbclassifier_predict():
    """
    Tests whether the monkey patching of ('sklearn.tree._classes.DecisionTreeClassifier', 'predict') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from xgboost import XGBClassifier
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = XGBClassifier(max_depth=12, tree_method='hist')
                clf = clf.fit(train, target)

                test_df = pd.DataFrame({'A': [0., 0.6], 'B':  [0., 0.6], 'target': ['no', 'yes']})
                predictions = clf.predict(test_df[['A', 'B']])
                assert len(predictions) == 2
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    filter_dag_for_nodes_with_ids(inspector_result, {7, 9, 10, 11}, 12)

    expected_dag = networkx.DiGraph()
    expected_data_projection = DagNode(9,
                                       BasicCodeLocation("<string-source>", 15),
                                       OperatorContext(OperatorType.PROJECTION,
                                                       FunctionInfo('pandas.core.frame', '__getitem__')),
                                       DagNodeDetails("to ['A', 'B']", ['A', 'B'],
                                                      OptimizerInfo(RangeComparison(0, 200), (2, 2),
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(15, 26, 15, 45), "test_df[['A', 'B']]"),
                                       Comparison(FunctionType))
    expected_test_data = DagNode(10,
                                 BasicCodeLocation("<string-source>", 15),
                                 OperatorContext(OperatorType.TEST_DATA,
                                                 FunctionInfo('xgboost.sklearn.XGBClassifier',
                                                              'predict')),
                                 DagNodeDetails(None, ['A', 'B'], OptimizerInfo(RangeComparison(0, 200), (2, 2),
                                                                                RangeComparison(0, 800))),
                                 OptionalCodeInfo(CodeReference(15, 14, 15, 46),
                                                  "clf.predict(test_df[['A', 'B']])"),
                                 Comparison(FunctionType))
    expected_dag.add_edge(expected_data_projection, expected_test_data, arg_index=0)
    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 11),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('xgboost.sklearn', 'XGBClassifier')),
                                  DagNodeDetails('XGB Classifier', [], OptimizerInfo(RangeComparison(0, 1000), None,
                                                                                     RangeComparison(0, 10000))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 53),
                                                   "XGBClassifier(max_depth=12, tree_method='hist')"),
                                  Comparison(FunctionType),
                                  Comparison(partial))
    expected_predict = DagNode(11,
                               BasicCodeLocation("<string-source>", 15),
                               OperatorContext(OperatorType.PREDICT,
                                               FunctionInfo('xgboost.sklearn.XGBClassifier', 'predict')),
                               DagNodeDetails('XGB Classifier', [], OptimizerInfo(RangeComparison(0, 1000), (2, 1),
                                                                                  RangeComparison(0, 800))),
                               OptionalCodeInfo(CodeReference(15, 14, 15, 46),
                                                "clf.predict(test_df[['A', 'B']])"),
                               Comparison(FunctionType))
    expected_dag.add_edge(expected_classifier, expected_predict, arg_index=0)
    expected_dag.add_edge(expected_test_data, expected_predict, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.original_dag.nodes)[0]
    predict_node = list(inspector_result.original_dag.nodes)[3]
    test_data_node = list(inspector_result.original_dag.nodes)[2]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_labels = label_binarize(train_df['target'], classes=['no', 'yes'])
    fitted_estimator = fit_node.processing_func(train_df[['C', 'D']], train_labels)
    assert isinstance(fitted_estimator, XGBClassifier)
    assert isinstance(fit_node.make_classifier_func(), XGBClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_predict = predict_node.processing_func(fitted_estimator, test_data)
    assert len(test_predict) == 2
