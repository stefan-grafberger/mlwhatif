"""
Tests whether the monkey patching works for all patched sklearn methods
"""
# pylint: disable=too-many-lines
from functools import partial
from inspect import cleandoc
from types import FunctionType

import networkx
import numpy
import pandas
from scipy.sparse import csr_matrix
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize, OneHotEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier  # pylint: disable=no-name-in-module
from testfixtures import compare, Comparison, RangeComparison

from mlmq import OperatorType, OperatorContext, FunctionInfo
from mlmq.execution import _pipeline_executor
from mlmq.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo, OptimizerInfo
from mlmq.monkeypatching._patch_sklearn import TrainTestSplitResult


def test_label_binarize():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._label', 'label_binarize') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize
                import numpy as np

                pd_series = pd.Series(['yes', 'no', 'no', 'yes'], name='A')
                binarized = label_binarize(pd_series, classes=['no', 'yes'])
                expected = np.array([[1], [0], [0], [1]])
                assert np.array_equal(binarized, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 5),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.series', 'Series')),
                                   DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                             RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(5, 12, 5, 59),
                                                    "pd.Series(['yes', 'no', 'no', 'yes'], name='A')"),
                                   Comparison(partial))
    expected_binarize = DagNode(1,
                                BasicCodeLocation("<string-source>", 6),
                                OperatorContext(OperatorType.PROJECTION_MODIFY,
                                                FunctionInfo('sklearn.preprocessing._label', 'label_binarize')),
                                DagNodeDetails("label_binarize, classes: ['no', 'yes']", ['array'],
                                               OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                             RangeComparison(0, 800))),
                                OptionalCodeInfo(CodeReference(6, 12, 6, 60),
                                                 "label_binarize(pd_series, classes=['no', 'yes'])"),
                                Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_binarize, arg_index=0)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    label_binarize_node = list(inspector_result.original_dag.nodes)[1]
    pd_series = pandas.Series(['no', 'yes', 'no', 'yes'], name='A')
    binarize_result = label_binarize_node.processing_func(pd_series)
    expected = numpy.array([[0], [1], [0], [1]])
    assert numpy.array_equal(binarize_result, expected)


def test_train_test_split():
    """
    Tests whether the monkey patching of ('sklearn.model_selection._split', 'train_test_split') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.model_selection import train_test_split

                pandas_df = pd.DataFrame({'A': [1, 2, 10, 5]})
                train_data, test_data = train_test_split(pandas_df, random_state=0)
                
                expected_train = pd.DataFrame({'A': [5, 2, 1]})
                expected_test = pd.DataFrame({'A': [10]})
                
                pd.testing.assert_frame_equal(train_data.reset_index(drop=True), expected_train.reset_index(drop=True))
                pd.testing.assert_frame_equal(test_data.reset_index(drop=True), expected_test.reset_index(drop=True))
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.original_dag.remove_node(list(inspector_result.original_dag.nodes)[5])
    inspector_result.original_dag.remove_node(list(inspector_result.original_dag.nodes)[4])

    expected_dag = networkx.DiGraph()
    expected_source = DagNode(0,
                              BasicCodeLocation("<string-source>", 4),
                              OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('pandas.core.frame', 'DataFrame')),
                              DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                        RangeComparison(0, 800))),
                              OptionalCodeInfo(CodeReference(4, 12, 4, 46), "pd.DataFrame({'A': [1, 2, 10, 5]})"),
                              Comparison(partial))
    expected_split = DagNode(1,
                             BasicCodeLocation("<string-source>", 5),
                             OperatorContext(OperatorType.TRAIN_TEST_SPLIT,
                                             FunctionInfo('sklearn.model_selection._split', 'train_test_split')),
                             DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                       RangeComparison(0, 800))),
                             OptionalCodeInfo(CodeReference(5, 24, 5, 67),
                                              'train_test_split(pandas_df, random_state=0)'),
                             Comparison(FunctionType))
    expected_dag.add_edge(expected_source, expected_split, arg_index=0)
    expected_train = DagNode(2,
                             BasicCodeLocation("<string-source>", 5),
                             OperatorContext(OperatorType.TRAIN_TEST_SPLIT,
                                             FunctionInfo('sklearn.model_selection._split', 'train_test_split')),
                             DagNodeDetails('(Train Data)', ['A'], OptimizerInfo(0, (3, 1), RangeComparison(0, 800))),
                             OptionalCodeInfo(CodeReference(5, 24, 5, 67),
                                              'train_test_split(pandas_df, random_state=0)'),
                             Comparison(FunctionType))
    expected_dag.add_edge(expected_split, expected_train, arg_index=0)
    expected_test = DagNode(3,
                            BasicCodeLocation("<string-source>", 5),
                            OperatorContext(OperatorType.TRAIN_TEST_SPLIT,
                                            FunctionInfo('sklearn.model_selection._split', 'train_test_split')),
                            DagNodeDetails('(Test Data)', ['A'], OptimizerInfo(0, (1, 1), RangeComparison(0, 800))),
                            OptionalCodeInfo(CodeReference(5, 24, 5, 67),
                                             'train_test_split(pandas_df, random_state=0)'),
                            Comparison(FunctionType))
    expected_dag.add_edge(expected_split, expected_test, arg_index=0)

    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    split_node = list(inspector_result.original_dag.nodes)[1]
    train_node = list(inspector_result.original_dag.nodes)[2]
    test_node = list(inspector_result.original_dag.nodes)[3]
    pandas_df = pandas.DataFrame({'A': ['a', 'c', 'e', 'f']})
    split_result = split_node.processing_func(pandas_df)
    assert isinstance(split_result, TrainTestSplitResult)
    train_data = train_node.processing_func(split_result)
    test_data = test_node.processing_func(split_result)

    expected_train = pandas.DataFrame({'A': ['f', 'c', 'a']})
    expected_test = pandas.DataFrame({'A': ['e']})

    pandas.testing.assert_frame_equal(train_data.reset_index(drop=True), expected_train.reset_index(drop=True))
    pandas.testing.assert_frame_equal(test_data.reset_index(drop=True), expected_test.reset_index(drop=True))


def test_standard_scaler():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._data', 'StandardScaler') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import StandardScaler
                import numpy as np

                df = pd.DataFrame({'A': [1, 2, 10, 5]})
                standard_scaler = StandardScaler()
                encoded_data = standard_scaler.fit_transform(df)
                test_df = pd.DataFrame({'A': [1, 2, 10, 5]})
                encoded_data = standard_scaler.transform(test_df)
                expected = np.array([[-1.], [-0.71428571], [1.57142857], [0.14285714]])
                assert np.allclose(encoded_data, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 5),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                             RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(5, 5, 5, 39), "pd.DataFrame({'A': [1, 2, 10, 5]})"),
                                   Comparison(partial))
    expected_transformer = DagNode(1,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                   DagNodeDetails('Standard Scaler: fit_transform', ['array'],
                                                  OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                RangeComparison(0, 4000))),
                                   OptionalCodeInfo(CodeReference(6, 18, 6, 34), 'StandardScaler()'),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_transformer, arg_index=0)
    expected_data_source_two = DagNode(2,
                                       BasicCodeLocation("<string-source>", 8),
                                       OperatorContext(OperatorType.DATA_SOURCE,
                                                       FunctionInfo('pandas.core.frame', 'DataFrame')),
                                       DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                 RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(8, 10, 8, 44),
                                                        "pd.DataFrame({'A': [1, 2, 10, 5]})"),
                                       Comparison(partial))
    expected_transformer_two = DagNode(3,
                                       BasicCodeLocation("<string-source>", 6),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                       DagNodeDetails('Standard Scaler: transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(6, 18, 6, 34), 'StandardScaler()'),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_transformer, expected_transformer_two, arg_index=0)
    expected_dag.add_edge(expected_data_source_two, expected_transformer_two, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.original_dag.nodes)[1]
    transform_node = list(inspector_result.original_dag.nodes)[3]
    pandas_df = pandas.DataFrame({'A': [5, 1, 100, 2]})
    fit_transformed_result = fit_transform_node.processing_func(pandas_df)
    expected_fit_transform_data = numpy.array([[-0.52166986], [-0.61651893], [1.73099545], [-0.59280666]])
    assert numpy.allclose(fit_transformed_result, expected_fit_transform_data)

    test_df = pandas.DataFrame({'A': [50, 2, 10, 1]})
    encoded_data = transform_node.processing_func(fit_transformed_result, test_df)
    expected = numpy.array([[0.54538213], [-0.59280666], [-0.40310853], [-0.61651893]])
    assert numpy.allclose(encoded_data, expected)


def test_robust_scaler():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._data', 'RobustScaler') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import RobustScaler
                import numpy as np

                df = pd.DataFrame({'A': [1, 2, 10, 5]})
                standard_scaler = RobustScaler()
                encoded_data = standard_scaler.fit_transform(df)
                test_df = pd.DataFrame({'A': [1, 2, 10, 5]})
                encoded_data = standard_scaler.transform(test_df)
                print(encoded_data)
                expected = np.array([[-0.55555556], [-0.33333333], [ 1.44444444], [ 0.33333333]])
                assert np.allclose(encoded_data, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 5),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                             RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(5, 5, 5, 39), "pd.DataFrame({'A': [1, 2, 10, 5]})"),
                                   Comparison(partial))
    expected_transformer = DagNode(1,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('sklearn.preprocessing._data', 'RobustScaler')),
                                   DagNodeDetails('Robust Scaler: fit_transform', ['array'],
                                                  OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                RangeComparison(0, 4000))),
                                   OptionalCodeInfo(CodeReference(6, 18, 6, 32), 'RobustScaler()'),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_transformer, arg_index=0)
    expected_data_source_two = DagNode(2,
                                       BasicCodeLocation("<string-source>", 8),
                                       OperatorContext(OperatorType.DATA_SOURCE,
                                                       FunctionInfo('pandas.core.frame', 'DataFrame')),
                                       DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                 RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(8, 10, 8, 44),
                                                        "pd.DataFrame({'A': [1, 2, 10, 5]})"),
                                       Comparison(partial))
    expected_transformer_two = DagNode(3,
                                       BasicCodeLocation("<string-source>", 6),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data', 'RobustScaler')),
                                       DagNodeDetails('Robust Scaler: transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(6, 18, 6, 32), 'RobustScaler()'),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_transformer, expected_transformer_two, arg_index=0)
    expected_dag.add_edge(expected_data_source_two, expected_transformer_two, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.original_dag.nodes)[1]
    transform_node = list(inspector_result.original_dag.nodes)[3]
    pandas_df = pandas.DataFrame({'A': [5, 1, 100, 2]})
    fit_transformed_result = fit_transform_node.processing_func(pandas_df)
    expected_fit_transform_data = numpy.array([[0.05555556], [-0.09259259], [3.57407407], [-0.05555556]])
    assert numpy.allclose(fit_transformed_result, expected_fit_transform_data)

    test_df = pandas.DataFrame({'A': [50, 2, 10, 1]})
    encoded_data = transform_node.processing_func(fit_transformed_result, test_df)
    expected = numpy.array([[1.72222222], [-0.05555556], [0.24074074], [-0.09259259]])
    assert numpy.allclose(encoded_data, expected)


def test_pca():
    """
    Tests whether the monkey patching of ('sklearn.compose._column_transformer', 'ColumnTransformer') works with
    multiple transformers with dense output
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder
                from sklearn.compose import ColumnTransformer
                from sklearn.decomposition import PCA
                from scipy.sparse import csr_matrix
                import numpy
                
                df = pd.DataFrame({'A': [1, 2, 10, 5], 'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                column_transformer = ColumnTransformer(transformers=[
                    ('numeric', StandardScaler(), ['A']),
                    ('categorical', OneHotEncoder(sparse=False), ['B'])
                ])
                encoded_data = column_transformer.fit_transform(df)
                transformer = PCA(n_components=2, random_state=42)
                reduced_data = transformer.fit_transform(encoded_data)
                test_transform_function = transformer.transform(encoded_data)
                print(reduced_data)
                expected = numpy.array([[-0.80795996, -0.75406407],
                                        [-0.953227,    0.23384447],
                                        [ 1.64711991, -0.29071836],
                                        [ 0.11406705,  0.81093796]])
                assert numpy.allclose(reduced_data, expected)
                assert numpy.allclose(test_transform_function, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    filter_dag_for_nodes_with_ids(inspector_result, [5, 6, 7], 8)

    expected_dag = networkx.DiGraph()
    expected_concat = DagNode(5,
                              BasicCodeLocation("<string-source>", 9),
                              OperatorContext(OperatorType.CONCATENATION,
                                              FunctionInfo('sklearn.compose._column_transformer', 'ColumnTransformer')),
                              DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 4),
                                                                            RangeComparison(0, 2000))),
                              OptionalCodeInfo(CodeReference(9, 21, 12, 2),
                                               "ColumnTransformer(transformers=[\n"
                                               "    ('numeric', StandardScaler(), ['A']),\n"
                                               "    ('categorical', OneHotEncoder(sparse=False), ['B'])\n])"),
                              Comparison(FunctionType))
    expected_transformer = DagNode(6,
                                   BasicCodeLocation("<string-source>", 14),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('sklearn.decomposition._pca',
                                                                'PCA')),
                                   DagNodeDetails('PCA: fit_transform', ['array'],
                                                  OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                RangeComparison(0, 2000))),
                                   OptionalCodeInfo(CodeReference(14, 14, 14, 50),
                                                    'PCA(n_components=2, random_state=42)'),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_concat, expected_transformer, arg_index=0)
    expected_transform_test = DagNode(7,
                                      BasicCodeLocation("<string-source>", 14),
                                      OperatorContext(OperatorType.TRANSFORMER,
                                                      FunctionInfo('sklearn.decomposition._pca',
                                                                   'PCA')),
                                      DagNodeDetails('PCA: transform', ['array'],
                                                     OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                   RangeComparison(0, 2000))),
                                      OptionalCodeInfo(CodeReference(14, 14, 14, 50),
                                                       'PCA(n_components=2, random_state=42)'),
                                      Comparison(FunctionType))
    expected_dag.add_edge(expected_transformer, expected_transform_test, arg_index=0)
    expected_dag.add_edge(expected_concat, expected_transform_test, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.original_dag.nodes)[1]
    transform_node = list(inspector_result.original_dag.nodes)[2]
    pandas_df = pandas.DataFrame({'A': [5, 1, 100, 2], 'B': [5, 1, 100, 2]})
    fit_transformed_result = fit_transform_node.processing_func(pandas_df)
    expected_fit_transform_data = numpy.array([[-3.11126984e+01, 1.17350380e-14], [-3.67695526e+01, -2.15521568e-15],
                                               [1.03237590e+02, 1.99309735e-15], [-3.53553391e+01, -2.26556491e-15]])
    assert numpy.allclose(fit_transformed_result, expected_fit_transform_data)

    test_df = pandas.DataFrame({'A': [50, 2, 10, 1], 'B': [50, 2, 10, 1]})
    encoded_data = transform_node.processing_func(fit_transformed_result, test_df)
    expected = numpy.array([[3.25269119e+01, 4.88498131e-15], [-3.53553391e+01, -5.77315973e-15],
                            [-2.40416306e+01, -3.99680289e-15], [-3.67695526e+01, -4.44089210e-15]])
    assert numpy.allclose(encoded_data, expected)


def test_function_transformer():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing_function_transformer', 'FunctionTransformer') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import FunctionTransformer
                import numpy as np
                
                def safe_log(x):
                    return np.log(x, out=np.zeros_like(x), where=(x!=0))

                df = pd.DataFrame({'A': [1, 2, 10, 5]})
                function_transformer = FunctionTransformer(lambda x: safe_log(x))
                encoded_data = function_transformer.fit_transform(df)
                test_df = pd.DataFrame({'A': [1, 2, 10, 5]})
                encoded_data = function_transformer.transform(test_df)
                expected = np.array([[0.000000], [0.693147], [2.302585], [1.609438]])
                assert np.allclose(encoded_data, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 8),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                             RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(8, 5, 8, 39), "pd.DataFrame({'A': [1, 2, 10, 5]})"),
                                   Comparison(partial))
    expected_transformer = DagNode(1,
                                   BasicCodeLocation("<string-source>", 9),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('sklearn.preprocessing_function_transformer',
                                                                'FunctionTransformer')),
                                   DagNodeDetails('Function Transformer: fit_transform', ['A'],
                                                  OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                RangeComparison(0, 4000))),
                                   OptionalCodeInfo(CodeReference(9, 23, 9, 65),
                                                    'FunctionTransformer(lambda x: safe_log(x))'),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_transformer, arg_index=0)
    expected_data_source_two = DagNode(2,
                                       BasicCodeLocation("<string-source>", 11),
                                       OperatorContext(OperatorType.DATA_SOURCE,
                                                       FunctionInfo('pandas.core.frame', 'DataFrame')),
                                       DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                 RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(11, 10, 11, 44),
                                                        "pd.DataFrame({'A': [1, 2, 10, 5]})"),
                                       Comparison(partial))
    expected_transformer_two = DagNode(3,
                                       BasicCodeLocation("<string-source>", 9),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing_function_transformer',
                                                                    'FunctionTransformer')),
                                       DagNodeDetails('Function Transformer: transform', ['A'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                    RangeComparison(0, 10000))),
                                       OptionalCodeInfo(CodeReference(9, 23, 9, 65),
                                                        'FunctionTransformer(lambda x: safe_log(x))'),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_transformer, expected_transformer_two, arg_index=0)
    expected_dag.add_edge(expected_data_source_two, expected_transformer_two, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.original_dag.nodes)[1]
    transform_node = list(inspector_result.original_dag.nodes)[3]
    pandas_df = pandas.DataFrame({'A': [5, 1, 100, 2]})
    fit_transformed_result = fit_transform_node.processing_func(pandas_df)
    expected_fit_transform_data = numpy.array([[1.609438], [0.000000], [4.605170], [0.693147]])
    assert numpy.allclose(fit_transformed_result, expected_fit_transform_data)

    test_df = pandas.DataFrame({'A': [5, 1, ]})
    encoded_data = transform_node.processing_func(fit_transformed_result, test_df)
    expected = numpy.array([[1.609438], [0.000000]])
    assert numpy.allclose(encoded_data, expected)


def test_kbins_discretizer():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._discretization', 'KBinsDiscretizer') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import KBinsDiscretizer
                import numpy as np

                df = pd.DataFrame({'A': [1, 2, 10, 5]})
                discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
                encoded_data = discretizer.fit_transform(df)
                test_df = pd.DataFrame({'A': [1, 2, 10, 5]})
                encoded_data = discretizer.transform(test_df)
                expected = np.array([[0.], [0.], [2.], [1.]])
                assert np.allclose(encoded_data, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 5),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                             RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(5, 5, 5, 39), "pd.DataFrame({'A': [1, 2, 10, 5]})"),
                                   Comparison(partial))
    expected_transformer = DagNode(1,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('sklearn.preprocessing._discretization',
                                                                'KBinsDiscretizer')),
                                   DagNodeDetails('K-Bins Discretizer: fit_transform', ['array'],
                                                  OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                RangeComparison(0, 4000))),
                                   OptionalCodeInfo(CodeReference(6, 14, 6, 78),
                                                    "KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')"),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_transformer, arg_index=0)
    expected_data_source_two = DagNode(2,
                                       BasicCodeLocation("<string-source>", 8),
                                       OperatorContext(OperatorType.DATA_SOURCE,
                                                       FunctionInfo('pandas.core.frame', 'DataFrame')),
                                       DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                 RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(8, 10, 8, 44),
                                                        "pd.DataFrame({'A': [1, 2, 10, 5]})"),
                                       Comparison(partial))
    expected_transformer_two = DagNode(3,
                                       BasicCodeLocation("<string-source>", 6),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._discretization',
                                                                    'KBinsDiscretizer')),
                                       DagNodeDetails('K-Bins Discretizer: transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(6, 14, 6, 78),
                                                        "KBinsDiscretizer(n_bins=3, encode='ordinal', "
                                                        "strategy='uniform')"),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_transformer, expected_transformer_two, arg_index=0)
    expected_dag.add_edge(expected_data_source_two, expected_transformer_two, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.original_dag.nodes)[1]
    transform_node = list(inspector_result.original_dag.nodes)[3]
    pandas_df = pandas.DataFrame({'A': [5, 1, 100, 2]})
    fit_transformed_result = fit_transform_node.processing_func(pandas_df)
    expected_fit_transform_data = numpy.array([[0.], [0.], [2.], [0.]])
    assert numpy.allclose(fit_transformed_result, expected_fit_transform_data)

    test_df = pandas.DataFrame({'A': [50, 2, 10, 1]})
    encoded_data = transform_node.processing_func(fit_transformed_result, test_df)
    expected = numpy.array([[1.], [0.], [0.], [0.]])
    assert numpy.allclose(encoded_data, expected)


def test_simple_imputer():
    """
    Tests whether the monkey patching of ('sklearn.impute._base’, 'SimpleImputer') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.impute import SimpleImputer
                import numpy as np

                df = pd.DataFrame({'A': ['cat_a', np.nan, 'cat_a', 'cat_c']})
                imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
                imputed_data = imputer.fit_transform(df)
                test_df = pd.DataFrame({'A': ['cat_a', np.nan, 'cat_a', 'cat_c']})
                imputed_data = imputer.transform(test_df)
                expected = np.array([['cat_a'], ['cat_a'], ['cat_a'], ['cat_c']])
                assert np.array_equal(imputed_data, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 5),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                             RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(5, 5, 5, 61),
                                                    "pd.DataFrame({'A': ['cat_a', np.nan, 'cat_a', 'cat_c']})"),
                                   Comparison(partial))
    expected_transformer = DagNode(1,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('sklearn.impute._base', 'SimpleImputer')),
                                   DagNodeDetails('Simple Imputer: fit_transform', ['A'],
                                                  OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                RangeComparison(0, 4000))),
                                   OptionalCodeInfo(CodeReference(6, 10, 6, 72),
                                                    "SimpleImputer(missing_values=np.nan, strategy='most_frequent')"),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_transformer, arg_index=0)
    expected_data_source_two = DagNode(2,
                                       BasicCodeLocation("<string-source>", 8),
                                       OperatorContext(OperatorType.DATA_SOURCE,
                                                       FunctionInfo('pandas.core.frame', 'DataFrame')),
                                       DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                 RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(8, 10, 8, 66),
                                                        "pd.DataFrame({'A': ['cat_a', np.nan, 'cat_a', 'cat_c']})"),
                                       Comparison(partial))
    expected_transformer_two = DagNode(3,
                                       BasicCodeLocation("<string-source>", 6),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.impute._base', 'SimpleImputer')),
                                       DagNodeDetails('Simple Imputer: transform', ['A'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(6, 10, 6, 72),
                                                        "SimpleImputer(missing_values=np.nan, strategy='most_frequent')"),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_transformer, expected_transformer_two, arg_index=0)
    expected_dag.add_edge(expected_data_source_two, expected_transformer_two, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.original_dag.nodes)[1]
    transform_node = list(inspector_result.original_dag.nodes)[3]
    pandas_df = pandas.DataFrame({'A': ['cat_d', 'cat_h', 'cat_d', numpy.nan]})
    fit_transformed_result = fit_transform_node.processing_func(pandas_df)
    expected_fit_transform_data = numpy.array([['cat_d'], ['cat_h'], ['cat_d'], ['cat_d']])
    assert numpy.array_equal(fit_transformed_result, expected_fit_transform_data)

    test_df = pandas.DataFrame({'A': [numpy.nan, 'cat_c']})
    encoded_data = transform_node.processing_func(fit_transformed_result, test_df)
    expected = numpy.array([['cat_d'], ['cat_c']])
    assert numpy.array_equal(encoded_data, expected)


def test_one_hot_encoder_not_sparse():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._encoders', 'OneHotEncoder') with dense output
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, OneHotEncoder
                import numpy as np

                df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                one_hot_encoder = OneHotEncoder(sparse=False)
                encoded_data = one_hot_encoder.fit_transform(df)
                expected = np.array([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
                print(encoded_data)
                assert np.allclose(encoded_data, expected)
                test_df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                encoded_data = one_hot_encoder.transform(test_df)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 5),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                             RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(5, 5, 5, 62),
                                                    "pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})"),
                                   Comparison(partial))
    expected_transformer = DagNode(1,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')),
                                   DagNodeDetails('One-Hot Encoder: fit_transform', ['array'],
                                                  OptimizerInfo(RangeComparison(0, 200), (4, 3),
                                                                RangeComparison(0, 4000))),
                                   OptionalCodeInfo(CodeReference(6, 18, 6, 45), 'OneHotEncoder(sparse=False)'),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_transformer, arg_index=0)
    expected_data_source_two = DagNode(2,
                                       BasicCodeLocation("<string-source>", 11),
                                       OperatorContext(OperatorType.DATA_SOURCE,
                                                       FunctionInfo('pandas.core.frame', 'DataFrame')),
                                       DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                 RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(11, 10, 11, 67),
                                                        "pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})"),
                                       Comparison(partial))
    expected_transformer_two = DagNode(3,
                                       BasicCodeLocation("<string-source>", 6),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._encoders',
                                                                    'OneHotEncoder')),
                                       DagNodeDetails('One-Hot Encoder: transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 3),
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(6, 18, 6, 45), 'OneHotEncoder(sparse=False)'),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_transformer, expected_transformer_two, arg_index=0)
    expected_dag.add_edge(expected_data_source_two, expected_transformer_two, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.original_dag.nodes)[1]
    transform_node = list(inspector_result.original_dag.nodes)[3]
    pandas_df = pandas.DataFrame({'A': ['cat_d', 'cat_h', 'cat_d', 'cat_c']})
    fit_transformed_result = fit_transform_node.processing_func(pandas_df)
    expected_fit_transform_data = numpy.array([[0., 1., 0.], [0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])
    assert numpy.allclose(fit_transformed_result, expected_fit_transform_data)

    test_df = pandas.DataFrame({'A': ['cat_h', 'cat_c']})
    encoded_data = transform_node.processing_func(fit_transformed_result, test_df)
    expected = numpy.array([[0., 0., 1.], [1., 0., 0.]])
    assert numpy.allclose(encoded_data, expected)


def test_one_hot_encoder_sparse():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._encoders', 'OneHotEncoder') works for sparse output
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, OneHotEncoder
                from scipy.sparse import csr_matrix
                import numpy

                df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                one_hot_encoder = OneHotEncoder()
                encoded_data = one_hot_encoder.fit_transform(df)
                expected = csr_matrix([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
                assert numpy.allclose(encoded_data.A, expected.A) and isinstance(encoded_data, csr_matrix)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                             RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(6, 5, 6, 62),
                                                    "pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})"),
                                   Comparison(partial))
    expected_transformer = DagNode(1,
                                   BasicCodeLocation("<string-source>", 7),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')),
                                   DagNodeDetails('One-Hot Encoder: fit_transform', ['array'],
                                                  OptimizerInfo(RangeComparison(0, 200), (4, 3),
                                                                RangeComparison(0, 4000))),
                                   OptionalCodeInfo(CodeReference(7, 18, 7, 33), 'OneHotEncoder()'),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_transformer, arg_index=0)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.original_dag.nodes)[1]
    pandas_df = pandas.DataFrame({'A': ['cat_d', 'cat_h', 'cat_d', 'cat_c']})
    fit_transformed_result = fit_transform_node.processing_func(pandas_df)
    expected_fit_transform_data = csr_matrix([[0., 1., 0.], [0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])
    assert numpy.allclose(fit_transformed_result.A, expected_fit_transform_data.A)


def test_hashing_vectorizer():
    """
    Tests whether the monkey patching of ('sklearn.feature_extraction.text', 'HashingVectorizer') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.feature_extraction.text import HashingVectorizer
                from scipy.sparse import csr_matrix
                import numpy as np

                df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                vectorizer = HashingVectorizer(ngram_range=(1, 3), n_features=2**2)
                encoded_data = vectorizer.fit_transform(df['A'])
                expected = csr_matrix([[-0., 0., 0., -1.], [0., -1., -0., 0.], [0., 0., 0., -1.], [0., 0., 0., -1.]])
                assert np.allclose(encoded_data.A, expected.A)
                test_df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                encoded_data = vectorizer.transform(test_df['A'])
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.original_dag.remove_node(list(inspector_result.original_dag)[0])
    inspector_result.original_dag.remove_node(list(inspector_result.original_dag)[2])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(1,
                                   BasicCodeLocation("<string-source>", 8),
                                   OperatorContext(OperatorType.PROJECTION,
                                                   FunctionInfo('pandas.core.frame', '__getitem__')),
                                   DagNodeDetails("to ['A']", ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                   RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(8, 40, 8, 47), "df['A']"),
                                   Comparison(FunctionType))
    expected_transformer = DagNode(2,
                                   BasicCodeLocation("<string-source>", 7),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('sklearn.feature_extraction.text',
                                                                'HashingVectorizer')),
                                   DagNodeDetails('Hashing Vectorizer: fit_transform', ['array'],
                                                  OptimizerInfo(RangeComparison(0, 200), (4, 4),
                                                                RangeComparison(0, 4000))),
                                   OptionalCodeInfo(CodeReference(7, 13, 7, 67),
                                                    'HashingVectorizer(ngram_range=(1, 3), n_features=2**2)'),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_transformer, arg_index=0)
    expected_data_source_two = DagNode(4,
                                       BasicCodeLocation("<string-source>", 12),
                                       OperatorContext(OperatorType.PROJECTION,
                                                       FunctionInfo('pandas.core.frame', '__getitem__')),
                                       DagNodeDetails("to ['A']", ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                       RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(12, 36, 12, 48), "test_df['A']"),
                                       Comparison(FunctionType))
    expected_transformer_two = DagNode(5,
                                       BasicCodeLocation("<string-source>", 7),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.feature_extraction.text',
                                                                    'HashingVectorizer')),
                                       DagNodeDetails('Hashing Vectorizer: transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 4),
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(7, 13, 7, 67),
                                                        'HashingVectorizer(ngram_range=(1, 3), n_features=2**2)'),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_transformer, expected_transformer_two, arg_index=0)
    expected_dag.add_edge(expected_data_source_two, expected_transformer_two, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.original_dag.nodes)[1]
    transform_node = list(inspector_result.original_dag.nodes)[3]
    pandas_df = pandas.DataFrame({'A': ['aaa', 'bbb', 'bbb', 'aaa']})
    fit_transformed_result = fit_transform_node.processing_func(pandas_df['A'])
    expected = csr_matrix([[0., -1., 0., 0.], [0., 0., 0., -1.], [0., 0., 0., -1.], [0., -1., 0., 0.]])
    assert numpy.allclose(fit_transformed_result.A, expected.A)

    test_df = pandas.DataFrame({'A': ['bbb', 'ccc', 'bbb', 'bbb']})
    encoded_data = transform_node.processing_func(fit_transformed_result, test_df['A'])
    expected = csr_matrix([[0., 0., 0., -1.], [1., 0., 0., 0.], [0., 0., 0., -1.], [0., 0., 0., -1.]])
    assert numpy.allclose(encoded_data.A, expected.A)


def test_column_transformer_one_transformer():
    """
    Tests whether the monkey patching of ('sklearn.compose._column_transformer', 'ColumnTransformer') works with
    one transformer
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.compose import ColumnTransformer
                from scipy.sparse import csr_matrix
                import numpy

                df = pd.DataFrame({'A': [1, 2, 10, 5], 'B': [1, 2, 10, 5]})
                column_transformer = ColumnTransformer(transformers=[
                    ('numeric', StandardScaler(), ['A', 'B'])
                ])
                encoded_data = column_transformer.fit_transform(df)
                expected = numpy.array([[-1.], [-0.71428571], [1.57142857], [0.14285714]])
                assert numpy.allclose(encoded_data, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 7),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A', 'B'],
                                                  OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(7, 5, 7, 59),
                                                    "pd.DataFrame({'A': [1, 2, 10, 5], 'B': [1, 2, 10, 5]})"),
                                   Comparison(partial))
    expected_projection = DagNode(1,
                                  BasicCodeLocation("<string-source>", 8),
                                  OperatorContext(OperatorType.PROJECTION,
                                                  FunctionInfo('sklearn.compose._column_transformer',
                                                               'ColumnTransformer')),
                                  DagNodeDetails("to ['A', 'B']", ['A', 'B'],
                                                 OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                               RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(8, 21, 10, 2),
                                                   "ColumnTransformer(transformers=[\n"
                                                   "    ('numeric', StandardScaler(), ['A', 'B'])\n])"),
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_projection, arg_index=0)
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 9),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                       DagNodeDetails('Standard Scaler: fit_transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                    RangeComparison(0, 4000))),
                                       OptionalCodeInfo(CodeReference(9, 16, 9, 32), 'StandardScaler()'),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_projection, expected_standard_scaler, arg_index=0)
    expected_concat = DagNode(3,
                              BasicCodeLocation("<string-source>", 8),
                              OperatorContext(OperatorType.CONCATENATION,
                                              FunctionInfo('sklearn.compose._column_transformer', 'ColumnTransformer')),
                              DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                            RangeComparison(0, 800))),
                              OptionalCodeInfo(CodeReference(8, 21, 10, 2),
                                               "ColumnTransformer(transformers=[\n"
                                               "    ('numeric', StandardScaler(), ['A', 'B'])\n])"),
                              Comparison(FunctionType))
    expected_dag.add_edge(expected_standard_scaler, expected_concat, arg_index=0)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    project_node = list(inspector_result.original_dag.nodes)[1]
    transformer_node = list(inspector_result.original_dag.nodes)[2]
    concat_node = list(inspector_result.original_dag.nodes)[3]
    pandas_df = pandas.DataFrame({'A': [1, 2, 10, 5], 'B': [1, 2, 10, 5]})
    projected_data = project_node.processing_func(pandas_df)
    transformed_data = transformer_node.processing_func(projected_data)
    concatenated_data = concat_node.processing_func(transformed_data)
    expected = numpy.array([[-1.], [-0.71428571], [1.57142857], [0.14285714]])
    assert numpy.allclose(concatenated_data, expected)


def test_column_transformer_one_transformer_single_column_projection():
    """
    Tests whether the monkey patching of ('sklearn.compose._column_transformer', 'ColumnTransformer') works with
    one transformer
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.feature_extraction.text import HashingVectorizer
                from scipy.sparse import csr_matrix
                from sklearn.compose import ColumnTransformer
                import numpy as np

                df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c'], 'B': [1, 2, 10, 5]})
                column_transformer = ColumnTransformer(transformers=[
                    ('hashing', HashingVectorizer(ngram_range=(1, 3), n_features=2**2), 'A')
                ])
                encoded_data = column_transformer.fit_transform(df)
                expected = csr_matrix([[-0., 0., 0., -1.], [0., -1., -0., 0.], [0., 0., 0., -1.], [0., 0., 0., -1.]])
                assert np.allclose(encoded_data.A, expected.A)
                test_df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c'],  'B': [1, 2, 10, 5]})
                encoded_data = column_transformer.transform(test_df)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    filter_dag_for_nodes_with_ids(inspector_result, {1, 2, 3, 6}, 8)

    expected_dag = networkx.DiGraph()
    expected_projection = DagNode(1,
                                  BasicCodeLocation("<string-source>", 8),
                                  OperatorContext(OperatorType.PROJECTION,
                                                  FunctionInfo('sklearn.compose._column_transformer',
                                                               'ColumnTransformer')),
                                  DagNodeDetails("to ['A']", ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                  RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(8, 21, 10, 2),
                                                   "ColumnTransformer(transformers=[\n"
                                                   "    ('hashing', HashingVectorizer(ngram_range=(1, 3), "
                                                   "n_features=2**2), 'A')\n])"),
                                  Comparison(FunctionType))
    expected_vectorizer = DagNode(2,
                                  BasicCodeLocation("<string-source>", 9),
                                  OperatorContext(OperatorType.TRANSFORMER,
                                                  FunctionInfo('sklearn.feature_extraction.text', 'HashingVectorizer')),
                                  DagNodeDetails('Hashing Vectorizer: fit_transform', ['array'],
                                                 OptimizerInfo(RangeComparison(0, 200), (4, 4),
                                                               RangeComparison(0, 4000))),
                                  OptionalCodeInfo(CodeReference(9, 16, 9, 70), 'HashingVectorizer(ngram_range=(1, 3), '
                                                                                'n_features=2**2)'),
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_projection, expected_vectorizer, arg_index=0)
    expected_concat = DagNode(3,
                              BasicCodeLocation("<string-source>", 8),
                              OperatorContext(OperatorType.CONCATENATION,
                                              FunctionInfo('sklearn.compose._column_transformer', 'ColumnTransformer')),
                              DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 4),
                                                                            RangeComparison(0, 800))),
                              OptionalCodeInfo(CodeReference(8, 21, 10, 2),
                                               "ColumnTransformer(transformers=[\n"
                                               "    ('hashing', HashingVectorizer(ngram_range=(1, 3), "
                                               "n_features=2**2), 'A')\n])"),
                              Comparison(FunctionType))
    expected_dag.add_edge(expected_vectorizer, expected_concat, arg_index=0)

    expected_transform = DagNode(6,
                                 BasicCodeLocation("<string-source>", 9),
                                 OperatorContext(OperatorType.TRANSFORMER,
                                                 FunctionInfo('sklearn.feature_extraction.text', 'HashingVectorizer')),
                                 DagNodeDetails('Hashing Vectorizer: transform', ['array'],
                                                OptimizerInfo(RangeComparison(0, 200), (4, 4),
                                                              RangeComparison(0, 800))),
                                 OptionalCodeInfo(CodeReference(9, 16, 9, 70), 'HashingVectorizer(ngram_range=(1, 3), '
                                                                               'n_features=2**2)'),
                                 Comparison(FunctionType))
    expected_dag.add_edge(expected_vectorizer, expected_transform, arg_index=0)

    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    project_node = list(inspector_result.original_dag.nodes)[0]
    transformer_node = list(inspector_result.original_dag.nodes)[1]
    concat_node = list(inspector_result.original_dag.nodes)[2]
    pandas_df = pandas.DataFrame({'A': ['cat_a', 'cat_a', 'cat_b', 'cat_c'], 'B': [1, 2, 10, 5]})
    projected_data = project_node.processing_func(pandas_df)
    transformed_data = transformer_node.processing_func(projected_data)
    concatenated_data = concat_node.processing_func(transformed_data)
    expected = csr_matrix([[-0., 0., 0., -1.], [-0., 0., 0., -1.], [0., -1., 0., 0.], [0., 0., 0., -1.]])
    assert numpy.allclose(concatenated_data.A, expected.A)


def test_column_transformer_multiple_transformers_all_dense():
    """
    Tests whether the monkey patching of ('sklearn.compose._column_transformer', 'ColumnTransformer') works with
    multiple transformers with dense output
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder
                from sklearn.compose import ColumnTransformer
                from scipy.sparse import csr_matrix
                import numpy

                df = pd.DataFrame({'A': [1, 2, 10, 5], 'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                column_transformer = ColumnTransformer(transformers=[
                    ('numeric', StandardScaler(), ['A']),
                    ('categorical', OneHotEncoder(sparse=False), ['B'])
                ])
                encoded_data = column_transformer.fit_transform(df)
                expected = numpy.array([[-1., 1., 0., 0.], [-0.71428571, 0., 1., 0.], [ 1.57142857, 1., 0., 0.], 
                    [0.14285714, 0., 0., 1.]])
                print(encoded_data)
                assert numpy.allclose(encoded_data, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 7),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A', 'B'], OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                                  RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(7, 5, 7, 82),
                                                    "pd.DataFrame({'A': [1, 2, 10, 5], "
                                                    "'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})"),
                                   Comparison(partial))
    expected_projection_1 = DagNode(1,
                                    BasicCodeLocation("<string-source>", 8),
                                    OperatorContext(OperatorType.PROJECTION,
                                                    FunctionInfo('sklearn.compose._column_transformer',
                                                                 'ColumnTransformer')),
                                    DagNodeDetails("to ['A']", ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                    RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(8, 21, 11, 2),
                                                     "ColumnTransformer(transformers=[\n"
                                                     "    ('numeric', StandardScaler(), ['A']),\n"
                                                     "    ('categorical', OneHotEncoder(sparse=False), ['B'])\n])"),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_projection_1, arg_index=0)
    expected_projection_2 = DagNode(3,
                                    BasicCodeLocation("<string-source>", 8),
                                    OperatorContext(OperatorType.PROJECTION,
                                                    FunctionInfo('sklearn.compose._column_transformer',
                                                                 'ColumnTransformer')),
                                    DagNodeDetails("to ['B']", ['B'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                    RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(8, 21, 11, 2),
                                                     "ColumnTransformer(transformers=[\n"
                                                     "    ('numeric', StandardScaler(), ['A']),\n"
                                                     "    ('categorical', OneHotEncoder(sparse=False), ['B'])\n])"),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_projection_2, arg_index=0)
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 9),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                       DagNodeDetails('Standard Scaler: fit_transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                    RangeComparison(0, 4000))),
                                       OptionalCodeInfo(CodeReference(9, 16, 9, 32), 'StandardScaler()'),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_projection_1, expected_standard_scaler, arg_index=0)
    expected_one_hot = DagNode(4,
                               BasicCodeLocation("<string-source>", 10),
                               OperatorContext(OperatorType.TRANSFORMER,
                                               FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')),
                               DagNodeDetails('One-Hot Encoder: fit_transform', ['array'],
                                              OptimizerInfo(RangeComparison(0, 200), (4, 3),
                                                            RangeComparison(0, 4000))),
                               OptionalCodeInfo(CodeReference(10, 20, 10, 47), 'OneHotEncoder(sparse=False)'),
                               Comparison(FunctionType))
    expected_dag.add_edge(expected_projection_2, expected_one_hot, arg_index=0)
    expected_concat = DagNode(5,
                              BasicCodeLocation("<string-source>", 8),
                              OperatorContext(OperatorType.CONCATENATION,
                                              FunctionInfo('sklearn.compose._column_transformer', 'ColumnTransformer')),
                              DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 4),
                                                                            RangeComparison(0, 800))),
                              OptionalCodeInfo(CodeReference(8, 21, 11, 2),
                                               "ColumnTransformer(transformers=[\n"
                                               "    ('numeric', StandardScaler(), ['A']),\n"
                                               "    ('categorical', OneHotEncoder(sparse=False), ['B'])\n])"),
                              Comparison(FunctionType))
    expected_dag.add_edge(expected_standard_scaler, expected_concat, arg_index=0)
    expected_dag.add_edge(expected_one_hot, expected_concat, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    project1_node = list(inspector_result.original_dag.nodes)[1]
    project2_node = list(inspector_result.original_dag.nodes)[3]
    transformer1_node = list(inspector_result.original_dag.nodes)[2]
    transformer2_node = list(inspector_result.original_dag.nodes)[4]
    concat_node = list(inspector_result.original_dag.nodes)[5]
    pandas_df = pandas.DataFrame({'A': [1, 2, 10, 5], 'B': ['cat_a', 'cat_b', 'cat_b', 'cat_c']})
    projected_data1 = project1_node.processing_func(pandas_df)
    projected_data2 = project2_node.processing_func(pandas_df)
    transformed_data1 = transformer1_node.processing_func(projected_data1)
    transformed_data2 = transformer2_node.processing_func(projected_data2)
    concatenated_data = concat_node.processing_func(transformed_data1, transformed_data2)
    expected = numpy.array([[-1., 1., 0., 0.], [-0.71428571, 0., 1., 0.], [1.57142857, 0., 1., 0.],
                            [0.14285714, 0., 0., 1.]])
    assert numpy.allclose(concatenated_data, expected)


def test_column_transformer_multiple_transformers_sparse_dense():
    """
    Tests whether the monkey patching of ('sklearn.compose._column_transformer', 'ColumnTransformer') works with
    multiple transformers with sparse and dense mixed output
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder
                from sklearn.compose import ColumnTransformer
                from scipy.sparse import csr_matrix
                import numpy

                df = pd.DataFrame({'A': [1, 2, 10, 5], 'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                column_transformer = ColumnTransformer(transformers=[
                    ('numeric', StandardScaler(), ['A']),
                    ('categorical', OneHotEncoder(sparse=True), ['B'])
                ])
                encoded_data = column_transformer.fit_transform(df)
                expected = numpy.array([[-1., 1., 0., 0.], [-0.71428571, 0., 1., 0.], [ 1.57142857, 1., 0., 0.], 
                    [0.14285714, 0., 0., 1.]])
                print(encoded_data)
                assert numpy.allclose(encoded_data, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 7),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A', 'B'], OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                                  RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(7, 5, 7, 82),
                                                    "pd.DataFrame({'A': [1, 2, 10, 5], "
                                                    "'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})"),
                                   Comparison(partial))
    expected_projection_1 = DagNode(1,
                                    BasicCodeLocation("<string-source>", 8),
                                    OperatorContext(OperatorType.PROJECTION,
                                                    FunctionInfo('sklearn.compose._column_transformer',
                                                                 'ColumnTransformer')),
                                    DagNodeDetails("to ['A']", ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                    RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(8, 21, 11, 2),
                                                     "ColumnTransformer(transformers=[\n"
                                                     "    ('numeric', StandardScaler(), ['A']),\n"
                                                     "    ('categorical', OneHotEncoder(sparse=True), ['B'])\n])"),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_projection_1, arg_index=0)
    expected_projection_2 = DagNode(3,
                                    BasicCodeLocation("<string-source>", 8),
                                    OperatorContext(OperatorType.PROJECTION,
                                                    FunctionInfo('sklearn.compose._column_transformer',
                                                                 'ColumnTransformer')),
                                    DagNodeDetails("to ['B']", ['B'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                    RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(8, 21, 11, 2),
                                                     "ColumnTransformer(transformers=[\n"
                                                     "    ('numeric', StandardScaler(), ['A']),\n"
                                                     "    ('categorical', OneHotEncoder(sparse=True), ['B'])\n])"),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_projection_2, arg_index=0)
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 9),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                       DagNodeDetails('Standard Scaler: fit_transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                    RangeComparison(0, 4000))),
                                       OptionalCodeInfo(CodeReference(9, 16, 9, 32), 'StandardScaler()'),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_projection_1, expected_standard_scaler, arg_index=0)
    expected_one_hot = DagNode(4,
                               BasicCodeLocation("<string-source>", 10),
                               OperatorContext(OperatorType.TRANSFORMER,
                                               FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')),
                               DagNodeDetails('One-Hot Encoder: fit_transform', ['array'],
                                              OptimizerInfo(RangeComparison(0, 200), (4, 3),
                                                            RangeComparison(0, 4000))),
                               OptionalCodeInfo(CodeReference(10, 20, 10, 46), 'OneHotEncoder(sparse=True)'),
                               Comparison(FunctionType))
    expected_dag.add_edge(expected_projection_2, expected_one_hot, arg_index=0)
    expected_concat = DagNode(5,
                              BasicCodeLocation("<string-source>", 8),
                              OperatorContext(OperatorType.CONCATENATION,
                                              FunctionInfo('sklearn.compose._column_transformer', 'ColumnTransformer')),
                              DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 4),
                                                                            RangeComparison(0, 800))),
                              OptionalCodeInfo(CodeReference(8, 21, 11, 2),
                                               "ColumnTransformer(transformers=[\n"
                                               "    ('numeric', StandardScaler(), ['A']),\n"
                                               "    ('categorical', OneHotEncoder(sparse=True), ['B'])\n])"),
                              Comparison(FunctionType))
    expected_dag.add_edge(expected_standard_scaler, expected_concat, arg_index=0)
    expected_dag.add_edge(expected_one_hot, expected_concat, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    project1_node = list(inspector_result.original_dag.nodes)[1]
    project2_node = list(inspector_result.original_dag.nodes)[3]
    transformer1_node = list(inspector_result.original_dag.nodes)[2]
    transformer2_node = list(inspector_result.original_dag.nodes)[4]
    concat_node = list(inspector_result.original_dag.nodes)[5]
    pandas_df = pandas.DataFrame({'A': [1, 2, 10, 5], 'B': ['cat_a', 'cat_b', 'cat_b', 'cat_c']})
    projected_data1 = project1_node.processing_func(pandas_df)
    projected_data2 = project2_node.processing_func(pandas_df)
    transformed_data1 = transformer1_node.processing_func(projected_data1)
    transformed_data2 = transformer2_node.processing_func(projected_data2)
    concatenated_data = concat_node.processing_func(transformed_data1, transformed_data2)
    expected = numpy.array([[-1., 1., 0., 0.], [-0.71428571, 0., 1., 0.], [1.57142857, 0., 1., 0.],
                            [0.14285714, 0., 0., 1.]])
    assert numpy.allclose(concatenated_data, expected)


def test_column_transformer_transform_after_fit_transform():
    """
    Tests whether the monkey patching of ('sklearn.compose._column_transformer', 'ColumnTransformer') works with
    multiple transformers with sparse and dense mixed output
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder
                from sklearn.compose import ColumnTransformer
                from scipy.sparse import csr_matrix
                import numpy

                df = pd.DataFrame({'A': [1, 2, 10, 5], 'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                column_transformer = ColumnTransformer(transformers=[
                    ('numeric', StandardScaler(), ['A']),
                    ('categorical', OneHotEncoder(sparse=True), ['B'])
                ])
                encoded_data = column_transformer.fit_transform(df)
                test_df = pd.DataFrame({'A': [1, 2, 10, 5], 'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                encoded_data = column_transformer.transform(test_df)
                expected = numpy.array([[-1., 1., 0., 0.], [-0.71428571, 0., 1., 0.], [ 1.57142857, 1., 0., 0.], 
                    [0.14285714, 0., 0., 1.]])
                print(encoded_data)
                assert numpy.allclose(encoded_data, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    fit_func_transformer1 = list(inspector_result.original_dag.nodes)[2].processing_func
    fit_func_transformer2 = list(inspector_result.original_dag.nodes)[4].processing_func
    filter_dag_for_nodes_with_ids(inspector_result, {6, 7, 8, 9, 10, 11}, 12)

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(6,
                                   BasicCodeLocation("<string-source>", 13),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A', 'B'], OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                                  RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(13, 10, 13, 87),
                                                    "pd.DataFrame({'A': [1, 2, 10, 5], "
                                                    "'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})"),
                                   Comparison(partial))
    expected_projection_1 = DagNode(7,
                                    BasicCodeLocation("<string-source>", 8),
                                    OperatorContext(OperatorType.PROJECTION,
                                                    FunctionInfo('sklearn.compose._column_transformer',
                                                                 'ColumnTransformer')),
                                    DagNodeDetails("to ['A']", ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                    RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(8, 21, 11, 2),
                                                     "ColumnTransformer(transformers=[\n"
                                                     "    ('numeric', StandardScaler(), ['A']),\n"
                                                     "    ('categorical', OneHotEncoder(sparse=True), ['B'])\n])"),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_projection_1, arg_index=0)
    expected_projection_2 = DagNode(9,
                                    BasicCodeLocation("<string-source>", 8),
                                    OperatorContext(OperatorType.PROJECTION,
                                                    FunctionInfo('sklearn.compose._column_transformer',
                                                                 'ColumnTransformer')),
                                    DagNodeDetails("to ['B']", ['B'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                    RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(8, 21, 11, 2),
                                                     "ColumnTransformer(transformers=[\n"
                                                     "    ('numeric', StandardScaler(), ['A']),\n"
                                                     "    ('categorical', OneHotEncoder(sparse=True), ['B'])\n])"),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_projection_2, arg_index=0)
    expected_standard_scaler = DagNode(8,
                                       BasicCodeLocation("<string-source>", 9),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                       DagNodeDetails('Standard Scaler: transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(9, 16, 9, 32), 'StandardScaler()'),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_projection_1, expected_standard_scaler, arg_index=1)
    expected_one_hot = DagNode(10,
                               BasicCodeLocation("<string-source>", 10),
                               OperatorContext(OperatorType.TRANSFORMER,
                                               FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')),
                               DagNodeDetails('One-Hot Encoder: transform', ['array'],
                                              OptimizerInfo(RangeComparison(0, 200), (4, 3),
                                                            RangeComparison(0, 800))),
                               OptionalCodeInfo(CodeReference(10, 20, 10, 46), 'OneHotEncoder(sparse=True)'),
                               Comparison(FunctionType))
    expected_dag.add_edge(expected_projection_2, expected_one_hot, arg_index=1)
    expected_concat = DagNode(11,
                              BasicCodeLocation("<string-source>", 8),
                              OperatorContext(OperatorType.CONCATENATION,
                                              FunctionInfo('sklearn.compose._column_transformer', 'ColumnTransformer')),
                              DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 4),
                                                                            RangeComparison(0, 800))),
                              OptionalCodeInfo(CodeReference(8, 21, 11, 2),
                                               "ColumnTransformer(transformers=[\n"
                                               "    ('numeric', StandardScaler(), ['A']),\n"
                                               "    ('categorical', OneHotEncoder(sparse=True), ['B'])\n])"),
                              Comparison(FunctionType))
    expected_dag.add_edge(expected_standard_scaler, expected_concat, arg_index=0)
    expected_dag.add_edge(expected_one_hot, expected_concat, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    project1_node = list(inspector_result.original_dag.nodes)[1]
    project2_node = list(inspector_result.original_dag.nodes)[3]
    transformer1_node = list(inspector_result.original_dag.nodes)[2]
    transformer2_node = list(inspector_result.original_dag.nodes)[4]
    concat_node = list(inspector_result.original_dag.nodes)[5]
    pandas_df = pandas.DataFrame({'A': [1, 2, 10, 5], 'B': ['cat_a', 'cat_b', 'cat_b', 'cat_c']})
    projected_data1 = project1_node.processing_func(pandas_df)
    projected_data2 = project2_node.processing_func(pandas_df)
    fitted_transformer1 = fit_func_transformer1(projected_data1)
    fitted_transformer2 = fit_func_transformer2(projected_data2)
    transformed_data1 = transformer1_node.processing_func(fitted_transformer1, projected_data1)
    transformed_data2 = transformer2_node.processing_func(fitted_transformer2, projected_data2)
    concatenated_data = concat_node.processing_func(transformed_data1, transformed_data2)
    expected = numpy.array([[-1., 1., 0., 0.], [-0.71428571, 0., 1., 0.], [1.57142857, 0., 1., 0.],
                            [0.14285714, 0., 0., 1.]])
    assert numpy.allclose(concatenated_data, expected)


def test_decision_tree():
    """
    Tests whether the monkey patching of ('sklearn.tree._classes', 'DecisionTreeClassifier') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.tree import DecisionTreeClassifier
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
                
                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = DecisionTreeClassifier()
                clf = clf.fit(train, target)
                
                test_predict = clf.predict(pd.DataFrame([[0., 0.], [0.6, 0.6]], columns=['A', 'B']))
                expected = np.array([0., 1.])
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
                                                  FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')),
                                  DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                                RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 30), 'DecisionTreeClassifier()'),
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_standard_scaler, expected_train_data, arg_index=0)
    expected_train_labels = DagNode(6,
                                    BasicCodeLocation("<string-source>", 11),
                                    OperatorContext(OperatorType.TRAIN_LABELS,
                                                    FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')),
                                    DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                  RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(11, 6, 11, 30), 'DecisionTreeClassifier()'),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_train_labels, arg_index=0)
    expected_decision_tree = DagNode(7,
                                     BasicCodeLocation("<string-source>", 11),
                                     OperatorContext(OperatorType.ESTIMATOR,
                                                     FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')),
                                     DagNodeDetails('Decision Tree', [], OptimizerInfo(RangeComparison(0, 1000), None,
                                                                                       RangeComparison(0, 10000))),
                                     OptionalCodeInfo(CodeReference(11, 6, 11, 30), 'DecisionTreeClassifier()'),
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
    assert isinstance(fitted_estimator, DecisionTreeClassifier)
    assert isinstance(fit_node.make_classifier_func(), DecisionTreeClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_score = fitted_estimator.score(test_df[['C', 'D']], test_labels)
    assert test_score == 0.5


def test_decision_tree_score():
    """
    Tests whether the monkey patching of ('sklearn.tree._classes.DecisionTreeClassifier', 'score') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.tree import DecisionTreeClassifier
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = DecisionTreeClassifier()
                clf = clf.fit(train, target)

                test_df = pd.DataFrame({'A': [0., 0.6], 'B':  [0., 0.6], 'target': ['no', 'yes']})
                test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
                test_score = clf.score(test_df[['A', 'B']], test_labels)
                assert test_score == 1.0
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
                                                 FunctionInfo('sklearn.tree._classes.DecisionTreeClassifier', 'score')),
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
                                                   FunctionInfo('sklearn.tree._classes.DecisionTreeClassifier',
                                                                'score')),
                                   DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (2, 1),
                                                                                 RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                                    "clf.score(test_df[['A', 'B']], test_labels)"),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_test_labels, arg_index=0)
    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 11),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')),
                                  DagNodeDetails('Decision Tree', [], OptimizerInfo(RangeComparison(0, 1000), None,
                                                                                    RangeComparison(0, 10000))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 30),
                                                   'DecisionTreeClassifier()'),
                                  Comparison(FunctionType),
                                  Comparison(partial))
    expected_predict = DagNode(14,
                               BasicCodeLocation("<string-source>", 16),
                               OperatorContext(OperatorType.PREDICT,
                                               FunctionInfo('sklearn.tree._classes.DecisionTreeClassifier', 'score')),
                               DagNodeDetails('Decision Tree', [], OptimizerInfo(RangeComparison(0, 1000), (2, 1),
                                                                                 RangeComparison(0, 800))),
                               OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                                "clf.score(test_df[['A', 'B']], test_labels)"),
                               Comparison(FunctionType))
    expected_dag.add_edge(expected_classifier, expected_predict, arg_index=0)
    expected_dag.add_edge(expected_test_data, expected_predict, arg_index=1)

    expected_score = DagNode(15,
                             BasicCodeLocation("<string-source>", 16),
                             OperatorContext(OperatorType.SCORE,
                                             FunctionInfo('sklearn.tree._classes.DecisionTreeClassifier', 'score')),
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
    assert isinstance(fitted_estimator, DecisionTreeClassifier)
    assert isinstance(fit_node.make_classifier_func(), DecisionTreeClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_labels = test_label_node.processing_func(test_labels)
    test_predictions = predict_node.processing_func(fitted_estimator, test_data)
    test_score = score_node.processing_func(test_predictions, test_labels)
    assert test_score == 0.5


def test_decision_tree_predict():
    """
    Tests whether the monkey patching of ('sklearn.tree._classes.DecisionTreeClassifier', 'predict') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.tree import DecisionTreeClassifier
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = DecisionTreeClassifier()
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
                                                 FunctionInfo('sklearn.tree._classes.DecisionTreeClassifier',
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
                                                  FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')),
                                  DagNodeDetails('Decision Tree', [], OptimizerInfo(RangeComparison(0, 1000), None,
                                                                                    RangeComparison(0, 10000))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 30),
                                                   'DecisionTreeClassifier()'),
                                  Comparison(FunctionType),
                                  Comparison(partial))
    expected_predict = DagNode(11,
                               BasicCodeLocation("<string-source>", 15),
                               OperatorContext(OperatorType.PREDICT,
                                               FunctionInfo('sklearn.tree._classes.DecisionTreeClassifier',
                                                            'predict')),
                               DagNodeDetails('Decision Tree', [], OptimizerInfo(RangeComparison(0, 1000), (2, 1),
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
    assert isinstance(fitted_estimator, DecisionTreeClassifier)
    assert isinstance(fit_node.make_classifier_func(), DecisionTreeClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_predict = predict_node.processing_func(fitted_estimator, test_data)
    assert len(test_predict) == 2


def test_sgd_classifier():
    """
    Tests whether the monkey patching of ('sklearn.linear_model._stochastic_gradient', 'SGDClassifier') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.linear_model import SGDClassifier
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = SGDClassifier(loss='log', random_state=42)
                clf = clf.fit(train, target)

                test_predict = clf.predict(pd.DataFrame([[0., 0.], [0.6, 0.6]], columns=['A', 'B']))
                expected = np.array([0., 1.])
                assert np.allclose(test_predict, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    filter_dag_for_nodes_with_ids(inspector_result, {2, 5, 4, 6, 7}, 11)

    expected_dag = networkx.DiGraph()
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 8),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                       DagNodeDetails('Standard Scaler: fit_transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                    RangeComparison(0, 4000))),
                                       OptionalCodeInfo(CodeReference(8, 8, 8, 24), 'StandardScaler()'),
                                       Comparison(FunctionType))
    expected_train_data = DagNode(5,
                                  BasicCodeLocation("<string-source>", 11),
                                  OperatorContext(OperatorType.TRAIN_DATA,
                                                  FunctionInfo('sklearn.linear_model._stochastic_gradient',
                                                               'SGDClassifier')),
                                  DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                                RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 48),
                                                   "SGDClassifier(loss='log', random_state=42)"),
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_standard_scaler, expected_train_data, arg_index=0)
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
    expected_train_labels = DagNode(6,
                                    BasicCodeLocation("<string-source>", 11),
                                    OperatorContext(OperatorType.TRAIN_LABELS,
                                                    FunctionInfo('sklearn.linear_model._stochastic_gradient',
                                                                 'SGDClassifier')),
                                    DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                  RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(11, 6, 11, 48),
                                                     "SGDClassifier(loss='log', random_state=42)"),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_train_labels, arg_index=0)
    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 11),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('sklearn.linear_model._stochastic_gradient',
                                                               'SGDClassifier')),
                                  DagNodeDetails('SGD Classifier', [], OptimizerInfo(RangeComparison(0, 1000), None,
                                                                                     RangeComparison(0, 10000))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 48),
                                                   "SGDClassifier(loss='log', random_state=42)"),
                                  Comparison(FunctionType),
                                  Comparison(partial))
    expected_dag.add_edge(expected_train_data, expected_classifier, arg_index=0)
    expected_dag.add_edge(expected_train_labels, expected_classifier, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.original_dag.nodes)[4]
    train_data_node = list(inspector_result.original_dag.nodes)[2]
    train_label_node = list(inspector_result.original_dag.nodes)[3]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_data = train_data_node.processing_func(train_df[['C', 'D']])
    train_labels = label_binarize(train_df['target'], classes=['no', 'yes'])
    train_labels = train_label_node.processing_func(train_labels)
    fitted_estimator = fit_node.processing_func(train_data, train_labels)
    assert isinstance(fitted_estimator, SGDClassifier)
    assert isinstance(fit_node.make_classifier_func(), SGDClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_score = fitted_estimator.score(test_df[['C', 'D']], test_labels)
    assert test_score == 0.5


def filter_dag_for_nodes_with_ids(inspector_result, node_ids, total_expected_node_num):
    """
    Filter for DAG Nodes relevant for this test
    """
    assert len(inspector_result.original_dag.nodes) == total_expected_node_num
    dag_nodes_irrelevant__for_test = [dag_node for dag_node in list(inspector_result.original_dag.nodes)
                                      if dag_node.node_id not in node_ids]
    inspector_result.original_dag.remove_nodes_from(dag_nodes_irrelevant__for_test)
    assert len(inspector_result.original_dag.nodes) == len(node_ids)


def test_sgd_classifier_score():
    """
    Tests whether the monkey patching of ('sklearn.linear_model._stochastic_gradient.SGDClassifier', 'score') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.linear_model import SGDClassifier
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = SGDClassifier(loss='log', random_state=42)
                clf = clf.fit(train, target)

                test_df = pd.DataFrame({'A': [0., 0.6], 'B':  [0., 0.6], 'target': ['no', 'yes']})
                test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
                test_score = clf.score(test_df[['A', 'B']], test_labels)
                assert test_score == 1.0
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
                                                 FunctionInfo('sklearn.linear_model._stochastic_gradient.'
                                                              'SGDClassifier', 'score')),
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
                                                   FunctionInfo('sklearn.linear_model._stochastic_gradient.'
                                                                'SGDClassifier', 'score')),
                                   DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (2, 1),
                                                                                 RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                                    "clf.score(test_df[['A', 'B']], test_labels)"),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_test_labels, arg_index=0)
    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 11),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('sklearn.linear_model._stochastic_gradient',
                                                               'SGDClassifier')),
                                  DagNodeDetails('SGD Classifier', [], OptimizerInfo(RangeComparison(0, 1000), None,
                                                                                     RangeComparison(0, 10000))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 48),
                                                   "SGDClassifier(loss='log', random_state=42)"),
                                  Comparison(FunctionType),
                                  Comparison(partial))
    expected_predict = DagNode(14,
                               BasicCodeLocation("<string-source>", 16),
                               OperatorContext(OperatorType.PREDICT,
                                               FunctionInfo('sklearn.linear_model._stochastic_gradient.SGDClassifier',
                                                            'score')),
                               DagNodeDetails('SGD Classifier', [], OptimizerInfo(RangeComparison(0, 1000), (2, 1),
                                                                                  RangeComparison(0, 800))),
                               OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                                "clf.score(test_df[['A', 'B']], test_labels)"),
                               Comparison(FunctionType))
    expected_dag.add_edge(expected_classifier, expected_predict, arg_index=0)
    expected_dag.add_edge(expected_test_data, expected_predict, arg_index=1)

    expected_score = DagNode(15,
                             BasicCodeLocation("<string-source>", 16),
                             OperatorContext(OperatorType.SCORE,
                                             FunctionInfo('sklearn.linear_model._stochastic_gradient.SGDClassifier',
                                                          'score')),
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
    assert isinstance(fitted_estimator, SGDClassifier)
    assert isinstance(fit_node.make_classifier_func(), SGDClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_labels = test_label_node.processing_func(test_labels)
    test_predict = predict_node.processing_func(fitted_estimator, test_data)
    test_score = score_node.processing_func(test_predict, test_labels)
    assert test_score == 0.5


def test_sgd_classifier_predict():
    """
    Tests whether the monkey patching of ('sklearn.linear_model._stochastic_gradient.SGDClassifier', 'predict') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.linear_model import SGDClassifier
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = SGDClassifier(loss='log', random_state=42)
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
                                                 FunctionInfo('sklearn.linear_model._stochastic_gradient.SGDClassifier',
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
                                                  FunctionInfo('sklearn.linear_model._stochastic_gradient',
                                                               'SGDClassifier')),
                                  DagNodeDetails('SGD Classifier', [], OptimizerInfo(RangeComparison(0, 1000), None,
                                                                                     RangeComparison(0, 10000))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 48),
                                                   "SGDClassifier(loss='log', random_state=42)"),
                                  Comparison(FunctionType),
                                  Comparison(partial))
    expected_predict = DagNode(11,
                               BasicCodeLocation("<string-source>", 15),
                               OperatorContext(OperatorType.PREDICT,
                                               FunctionInfo('sklearn.linear_model._stochastic_gradient.SGDClassifier',
                                                            'predict')),
                               DagNodeDetails('SGD Classifier', [], OptimizerInfo(RangeComparison(0, 1000), (2, 1),
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
    assert isinstance(fitted_estimator, SGDClassifier)
    assert isinstance(fit_node.make_classifier_func(), SGDClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_predict = predict_node.processing_func(fitted_estimator, test_data)
    assert len(test_predict) == 2


def test_logistic_regression():
    """
    Tests whether the monkey patching of ('sklearn.linear_model._logistic', 'LogisticRegression') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.linear_model import LogisticRegression
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = LogisticRegression()
                clf = clf.fit(train, target)

                test_predict = clf.predict(pd.DataFrame([[0., 0.], [0.6, 0.6]], columns=['A', 'B']))
                expected = np.array([0., 1.])
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
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 8),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                       DagNodeDetails('Standard Scaler: fit_transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                    RangeComparison(0, 10000))),
                                       OptionalCodeInfo(CodeReference(8, 8, 8, 24), 'StandardScaler()'),
                                       Comparison(FunctionType))
    expected_data_projection = DagNode(1,
                                       BasicCodeLocation("<string-source>", 8),
                                       OperatorContext(OperatorType.PROJECTION,
                                                       FunctionInfo('pandas.core.frame', '__getitem__')),
                                       DagNodeDetails("to ['A', 'B']", ['A', 'B'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(8, 39, 8, 53), "df[['A', 'B']]"),
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
                                                  FunctionInfo('sklearn.linear_model._logistic', 'LogisticRegression')),
                                  DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                                RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 26), 'LogisticRegression()'),
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_standard_scaler, expected_train_data, arg_index=0)
    expected_train_labels = DagNode(6,
                                    BasicCodeLocation("<string-source>", 11),
                                    OperatorContext(OperatorType.TRAIN_LABELS,
                                                    FunctionInfo('sklearn.linear_model._logistic',
                                                                 'LogisticRegression')),
                                    DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                  RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(11, 6, 11, 26), 'LogisticRegression()'),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_train_labels, arg_index=0)
    expected_estimator = DagNode(7,
                                 BasicCodeLocation("<string-source>", 11),
                                 OperatorContext(OperatorType.ESTIMATOR,
                                                 FunctionInfo('sklearn.linear_model._logistic',
                                                              'LogisticRegression')),
                                 DagNodeDetails('Logistic Regression', [],
                                                OptimizerInfo(RangeComparison(0, 1000), None,
                                                              RangeComparison(0, 10000))),
                                 OptionalCodeInfo(CodeReference(11, 6, 11, 26), 'LogisticRegression()'),
                                 Comparison(FunctionType),
                                 Comparison(partial))
    expected_dag.add_edge(expected_train_data, expected_estimator, arg_index=0)
    expected_dag.add_edge(expected_train_labels, expected_estimator, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.original_dag.nodes)[7]
    train_data_node = list(inspector_result.original_dag.nodes)[5]
    train_label_node = list(inspector_result.original_dag.nodes)[6]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_data = train_data_node.processing_func(train_df[['C', 'D']])
    train_labels = label_binarize(train_df['target'], classes=['no', 'yes'])
    train_labels = train_label_node.processing_func(train_labels)
    fitted_estimator = fit_node.processing_func(train_data, train_labels)
    assert isinstance(fitted_estimator, LogisticRegression)
    assert isinstance(fit_node.make_classifier_func(), LogisticRegression)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_score = fitted_estimator.score(test_df[['C', 'D']], test_labels)
    assert test_score == 0.5


def test_logistic_regression_score():
    """
    Tests whether the monkey patching of ('sklearn.linear_model._logistic.LogisticRegression', 'score') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.linear_model import LogisticRegression
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = LogisticRegression()
                clf = clf.fit(train, target)

                test_df = pd.DataFrame({'A': [0., 0.6], 'B':  [0., 0.6], 'target': ['no', 'yes']})
                test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
                test_score = clf.score(test_df[['A', 'B']], test_labels)
                assert test_score == 1.0
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
                                                 FunctionInfo('sklearn.linear_model._logistic.LogisticRegression',
                                                              'score')),
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
                                                   FunctionInfo('sklearn.linear_model._logistic.LogisticRegression',
                                                                'score')),
                                   DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (2, 1),
                                                                                 RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                                    "clf.score(test_df[['A', 'B']], test_labels)"),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_test_labels, arg_index=0)
    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 11),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('sklearn.linear_model._logistic', 'LogisticRegression')),
                                  DagNodeDetails('Logistic Regression', [],
                                                 OptimizerInfo(RangeComparison(0, 1000), None,
                                                               RangeComparison(0, 10000))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 26),
                                                   'LogisticRegression()'),
                                  Comparison(FunctionType),
                                  Comparison(partial))
    expected_predict = DagNode(14,
                               BasicCodeLocation("<string-source>", 16),
                               OperatorContext(OperatorType.PREDICT,
                                               FunctionInfo('sklearn.linear_model._logistic.LogisticRegression',
                                                            'score')),
                               DagNodeDetails('Logistic Regression', [], OptimizerInfo(RangeComparison(0, 1000), (2, 1),
                                                                                       RangeComparison(0, 800))),
                               OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                                "clf.score(test_df[['A', 'B']], test_labels)"),
                               Comparison(FunctionType))
    expected_dag.add_edge(expected_classifier, expected_predict, arg_index=0)
    expected_dag.add_edge(expected_test_data, expected_predict, arg_index=1)

    expected_score = DagNode(15,
                             BasicCodeLocation("<string-source>", 16),
                             OperatorContext(OperatorType.SCORE,
                                             FunctionInfo('sklearn.linear_model._logistic.LogisticRegression',
                                                          'score')),
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
    assert isinstance(fitted_estimator, LogisticRegression)
    assert isinstance(fit_node.make_classifier_func(), LogisticRegression)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_labels = test_label_node.processing_func(test_labels)
    test_predict = predict_node.processing_func(fitted_estimator, test_data)
    test_score = score_node.processing_func(test_predict, test_labels)
    assert test_score == 0.5


def test_logistic_regression_predict():
    """
    Tests whether the monkey patching of ('sklearn.linear_model._logistic.LogisticRegression', 'score') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.linear_model import LogisticRegression
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = LogisticRegression()
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
                                                 FunctionInfo('sklearn.linear_model._logistic.LogisticRegression',
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
                                                  FunctionInfo('sklearn.linear_model._logistic', 'LogisticRegression')),
                                  DagNodeDetails('Logistic Regression', [],
                                                 OptimizerInfo(RangeComparison(0, 1000), None,
                                                               RangeComparison(0, 10000))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 26),
                                                   'LogisticRegression()'),
                                  Comparison(FunctionType),
                                  Comparison(partial))
    expected_predict = DagNode(11,
                               BasicCodeLocation("<string-source>", 15),
                               OperatorContext(OperatorType.PREDICT,
                                               FunctionInfo('sklearn.linear_model._logistic.LogisticRegression',
                                                            'predict')),
                               DagNodeDetails('Logistic Regression', [], OptimizerInfo(RangeComparison(0, 1000), (2, 1),
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
    assert isinstance(fitted_estimator, LogisticRegression)
    assert isinstance(fit_node.make_classifier_func(), LogisticRegression)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_predict = predict_node.processing_func(fitted_estimator, test_data)
    assert len(test_predict) == 2


def test_keras_wrapper():
    """
    Tests whether the monkey patching of ('tensorflow.python.keras.wrappers.scikit_learn', 'KerasClassifier') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import StandardScaler, OneHotEncoder
                from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
                from tensorflow.keras.layers import Dense
                from tensorflow.keras.models import Sequential
                from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = OneHotEncoder(sparse=False).fit_transform(df[['target']])
                
                def create_model(input_dim):
                    clf = Sequential()
                    clf.add(Dense(9, activation='relu', input_dim=input_dim))
                    clf.add(Dense(9, activation='relu'))
                    clf.add(Dense(2, activation='softmax'))
                    clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
                    return clf

                clf = KerasClassifier(build_fn=create_model, epochs=2, batch_size=1, verbose=0, input_dim=2)
                clf.fit(train, target)

                test_predict = clf.predict(pd.DataFrame([[0., 0.], [0.6, 0.6]], columns=['A', 'B']))
                assert test_predict.shape == (2,)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.original_dag.remove_node(list(inspector_result.original_dag.nodes)[10])
    inspector_result.original_dag.remove_node(list(inspector_result.original_dag.nodes)[9])
    inspector_result.original_dag.remove_node(list(inspector_result.original_dag.nodes)[8])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 9),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A', 'B', 'target'],
                                                  OptimizerInfo(RangeComparison(0, 200), (4, 3),
                                                                RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(9, 5, 9, 95),
                                                    "pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], "
                                                    "'target': ['no', 'no', 'yes', 'yes']})"),
                                   Comparison(partial))
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 11),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                       DagNodeDetails('Standard Scaler: fit_transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                    RangeComparison(0, 10000))),
                                       OptionalCodeInfo(CodeReference(11, 8, 11, 24), 'StandardScaler()'),
                                       Comparison(FunctionType))
    expected_data_projection = DagNode(1,
                                       BasicCodeLocation("<string-source>", 11),
                                       OperatorContext(OperatorType.PROJECTION,
                                                       FunctionInfo('pandas.core.frame', '__getitem__')),
                                       DagNodeDetails("to ['A', 'B']", ['A', 'B'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(11, 39, 11, 53), "df[['A', 'B']]"),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_data_projection, arg_index=0)
    expected_dag.add_edge(expected_data_projection, expected_standard_scaler, arg_index=0)
    expected_label_projection = DagNode(3,
                                        BasicCodeLocation("<string-source>", 12),
                                        OperatorContext(OperatorType.PROJECTION,
                                                        FunctionInfo('pandas.core.frame', '__getitem__')),
                                        DagNodeDetails("to ['target']", ['target'],
                                                       OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                     RangeComparison(0, 800))),
                                        OptionalCodeInfo(CodeReference(12, 51, 12, 65), "df[['target']]"),
                                        Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_label_projection, arg_index=0)
    expected_label_encode = DagNode(4,
                                    BasicCodeLocation("<string-source>", 12),
                                    OperatorContext(OperatorType.TRANSFORMER,
                                                    FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')),
                                    DagNodeDetails('One-Hot Encoder: fit_transform', ['array'],
                                                   OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                 RangeComparison(0, 10000))),
                                    OptionalCodeInfo(CodeReference(12, 9, 12, 36), 'OneHotEncoder(sparse=False)'),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_label_projection, expected_label_encode, arg_index=0)
    expected_train_data = DagNode(5,
                                  BasicCodeLocation("<string-source>", 22),
                                  OperatorContext(OperatorType.TRAIN_DATA,
                                                  FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn',
                                                               'KerasClassifier')),
                                  DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                                RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(22, 6, 22, 92),
                                                   'KerasClassifier(build_fn=create_model, epochs=2, '
                                                   'batch_size=1, verbose=0, input_dim=2)'),
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_standard_scaler, expected_train_data, arg_index=0)
    expected_train_labels = DagNode(6,
                                    BasicCodeLocation("<string-source>", 22),
                                    OperatorContext(OperatorType.TRAIN_LABELS,
                                                    FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn',
                                                                 'KerasClassifier')),
                                    DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                                  RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(22, 6, 22, 92),
                                                     'KerasClassifier(build_fn=create_model, epochs=2, '
                                                     'batch_size=1, verbose=0, input_dim=2)'),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_train_labels, arg_index=0)
    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 22),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn',
                                                               'KerasClassifier')),
                                  DagNodeDetails('Neural Network', [], OptimizerInfo(RangeComparison(0, 1000), None,
                                                                                     RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(22, 6, 22, 92),
                                                   'KerasClassifier(build_fn=create_model, epochs=2, '
                                                   'batch_size=1, verbose=0, input_dim=2)'),
                                  Comparison(FunctionType),
                                  Comparison(partial))
    expected_dag.add_edge(expected_train_data, expected_classifier, arg_index=0)
    expected_dag.add_edge(expected_train_labels, expected_classifier, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.original_dag.nodes)[7]
    train_data_node = list(inspector_result.original_dag.nodes)[5]
    train_label_node = list(inspector_result.original_dag.nodes)[6]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_data = train_data_node.processing_func(train_df[['C', 'D']])
    train_labels = OneHotEncoder(sparse=False).fit_transform(train_df[['target']])
    train_labels = train_label_node.processing_func(train_labels)
    fitted_estimator = fit_node.processing_func(train_data, train_labels)
    assert isinstance(fitted_estimator, KerasClassifier)
    assert isinstance(fit_node.make_classifier_func(), KerasClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_labels = OneHotEncoder(sparse=False).fit_transform(test_df[['target']])
    test_score = fitted_estimator.score(test_df[['C', 'D']], test_labels)
    assert test_score >= 0.0


def test_keras_wrapper_score():
    """
    Tests whether the monkey patching of ('tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier', 'score')
     works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import StandardScaler, OneHotEncoder
                from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
                from tensorflow.keras.layers import Dense
                from tensorflow.keras.models import Sequential
                from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
                import tensorflow as tf
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = OneHotEncoder(sparse=False).fit_transform(df[['target']])
                
                def create_model(input_dim):
                    clf = Sequential()
                    clf.add(Dense(2, activation='relu', input_dim=input_dim))
                    clf.add(Dense(2, activation='relu'))
                    clf.add(Dense(2, activation='softmax'))
                    clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
                    return clf

                np.random.seed(42)
                tf.random.set_seed(42)
                clf = KerasClassifier(build_fn=create_model, epochs=15, batch_size=1, verbose=0, input_dim=2)
                clf = clf.fit(train, target)

                test_df = pd.DataFrame({'A': [0., 0.8], 'B':  [0., 0.8], 'target': ['no', 'yes']})
                test_labels = OneHotEncoder(sparse=False).fit_transform(test_df[['target']])
                test_score = clf.score(test_df[['A', 'B']], test_labels)
                assert test_score == 1.0
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    filter_dag_for_nodes_with_ids(inspector_result, {7, 10, 11, 12, 13, 14, 15}, 16)

    expected_dag = networkx.DiGraph()
    expected_data_projection = DagNode(11,
                                       BasicCodeLocation("<string-source>", 30),
                                       OperatorContext(OperatorType.PROJECTION,
                                                       FunctionInfo('pandas.core.frame', '__getitem__')),
                                       DagNodeDetails("to ['A', 'B']", ['A', 'B'],
                                                      OptimizerInfo(RangeComparison(0, 200), (2, 2),
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(30, 23, 30, 42), "test_df[['A', 'B']]"),
                                       Comparison(FunctionType))
    expected_test_data = DagNode(12,
                                 BasicCodeLocation("<string-source>", 30),
                                 OperatorContext(OperatorType.TEST_DATA,
                                                 FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn.'
                                                              'KerasClassifier', 'score')),
                                 DagNodeDetails(None, ['A', 'B'], OptimizerInfo(RangeComparison(0, 200), (2, 2),
                                                                                RangeComparison(0, 800))),
                                 OptionalCodeInfo(CodeReference(30, 13, 30, 56),
                                                  "clf.score(test_df[['A', 'B']], test_labels)"),
                                 Comparison(FunctionType))
    expected_dag.add_edge(expected_data_projection, expected_test_data, arg_index=0)
    expected_label_encode = DagNode(10,
                                    BasicCodeLocation("<string-source>", 29),
                                    OperatorContext(OperatorType.TRANSFORMER,
                                                    FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')),
                                    DagNodeDetails('One-Hot Encoder: fit_transform', ['array'],
                                                   OptimizerInfo(RangeComparison(0, 200), (2, 2),
                                                                 RangeComparison(0, 10000))),
                                    OptionalCodeInfo(CodeReference(29, 14, 29, 41), 'OneHotEncoder(sparse=False)'),
                                    Comparison(FunctionType))
    expected_test_labels = DagNode(13,
                                   BasicCodeLocation("<string-source>", 30),
                                   OperatorContext(OperatorType.TEST_LABELS,
                                                   FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn.'
                                                                'KerasClassifier', 'score')),
                                   DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (2, 2),
                                                                                 RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(30, 13, 30, 56),
                                                    "clf.score(test_df[['A', 'B']], test_labels)"),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_test_labels, arg_index=0)
    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 25),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn',
                                                               'KerasClassifier')),
                                  DagNodeDetails('Neural Network', [], OptimizerInfo(RangeComparison(0, 1000), None,
                                                                                     RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(25, 6, 25, 93),
                                                   'KerasClassifier(build_fn=create_model, epochs=15, batch_size=1, '
                                                   'verbose=0, input_dim=2)'),
                                  Comparison(FunctionType),
                                  Comparison(partial))
    expected_predict = DagNode(14,
                               BasicCodeLocation("<string-source>", 30),
                               OperatorContext(OperatorType.PREDICT,
                                               FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn.'
                                                            'KerasClassifier', 'score')),
                               DagNodeDetails('Neural Network', [], OptimizerInfo(RangeComparison(0, 1000), (2, 1),
                                                                                  RangeComparison(0, 800))),
                               OptionalCodeInfo(CodeReference(30, 13, 30, 56),
                                                "clf.score(test_df[['A', 'B']], test_labels)"),
                               Comparison(FunctionType))
    expected_dag.add_edge(expected_classifier, expected_predict, arg_index=0)
    expected_dag.add_edge(expected_test_data, expected_predict, arg_index=1)

    expected_score = DagNode(15,
                             BasicCodeLocation("<string-source>", 30),
                             OperatorContext(OperatorType.SCORE,
                                             FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn.'
                                                          'KerasClassifier', 'score')),
                             DagNodeDetails('Accuracy', [], OptimizerInfo(RangeComparison(0, 1000), (1, 1),
                                                                          RangeComparison(0, 800))),
                             OptionalCodeInfo(CodeReference(30, 13, 30, 56),
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
    train_labels = OneHotEncoder(sparse=False).fit_transform(train_df[['target']])
    fitted_estimator = fit_node.processing_func(train_df[['C', 'D']], train_labels)
    assert isinstance(fitted_estimator, KerasClassifier)
    assert isinstance(fit_node.make_classifier_func(), KerasClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_labels = OneHotEncoder(sparse=False).fit_transform(test_df[['target']])
    test_labels = test_label_node.processing_func(test_labels)
    test_predictions = predict_node.processing_func(fitted_estimator, test_data)
    test_score = score_node.processing_func(test_predictions, test_labels)
    assert test_score >= 0.5


def test_keras_wrapper_predict():
    """
    Tests whether the monkey patching of ('tensorflow.python.keras.wrappers.scikit_learn.KerasClassifier', 'score')
     works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import StandardScaler, OneHotEncoder
                from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
                from tensorflow.keras.layers import Dense
                from tensorflow.keras.models import Sequential
                from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
                import tensorflow as tf
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = OneHotEncoder(sparse=False).fit_transform(df[['target']])

                def create_model(input_dim):
                    clf = Sequential()
                    clf.add(Dense(2, activation='relu', input_dim=input_dim))
                    clf.add(Dense(2, activation='relu'))
                    clf.add(Dense(2, activation='softmax'))
                    clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
                    return clf

                np.random.seed(42)
                tf.random.set_seed(42)
                clf = KerasClassifier(build_fn=create_model, epochs=15, batch_size=1, verbose=0, input_dim=2)
                clf = clf.fit(train, target)

                test_df = pd.DataFrame({'A': [0., 0.8], 'B':  [0., 0.8], 'target': ['no', 'yes']})
                predictions = clf.predict(test_df[['A', 'B']])
                print(predictions.shape)
                assert predictions.shape == (2,)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    filter_dag_for_nodes_with_ids(inspector_result, {7, 9, 10, 11}, 12)

    expected_dag = networkx.DiGraph()
    expected_data_projection = DagNode(9,
                                       BasicCodeLocation("<string-source>", 29),
                                       OperatorContext(OperatorType.PROJECTION,
                                                       FunctionInfo('pandas.core.frame', '__getitem__')),
                                       DagNodeDetails("to ['A', 'B']", ['A', 'B'],
                                                      OptimizerInfo(RangeComparison(0, 200), (2, 2),
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(29, 26, 29, 45), "test_df[['A', 'B']]"),
                                       Comparison(FunctionType))
    expected_test_data = DagNode(10,
                                 BasicCodeLocation("<string-source>", 29),
                                 OperatorContext(OperatorType.TEST_DATA,
                                                 FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn.'
                                                              'KerasClassifier', 'predict')),
                                 DagNodeDetails(None, ['A', 'B'], OptimizerInfo(RangeComparison(0, 200), (2, 2),
                                                                                RangeComparison(0, 800))),
                                 OptionalCodeInfo(CodeReference(29, 14, 29, 46), "clf.predict(test_df[['A', 'B']])"),
                                 Comparison(FunctionType))
    expected_dag.add_edge(expected_data_projection, expected_test_data, arg_index=0)
    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 25),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn',
                                                               'KerasClassifier')),
                                  DagNodeDetails('Neural Network', [], OptimizerInfo(RangeComparison(0, 1000), None,
                                                                                     RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(25, 6, 25, 93),
                                                   'KerasClassifier(build_fn=create_model, epochs=15, batch_size=1, '
                                                   'verbose=0, input_dim=2)'),
                                  Comparison(FunctionType),
                                  Comparison(partial))
    expected_predict = DagNode(11,
                               BasicCodeLocation("<string-source>", 29),
                               OperatorContext(OperatorType.PREDICT,
                                               FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn.'
                                                            'KerasClassifier', 'predict')),
                               DagNodeDetails('Neural Network', [], OptimizerInfo(RangeComparison(0, 1000), (2, 1),
                                                                                  RangeComparison(0, 800))),
                               OptionalCodeInfo(CodeReference(29, 14, 29, 46), "clf.predict(test_df[['A', 'B']])"),
                               Comparison(FunctionType))
    expected_dag.add_edge(expected_classifier, expected_predict, arg_index=0)
    expected_dag.add_edge(expected_test_data, expected_predict, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.original_dag.nodes)[0]
    predict_node = list(inspector_result.original_dag.nodes)[3]
    test_data_node = list(inspector_result.original_dag.nodes)[2]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_labels = OneHotEncoder(sparse=False).fit_transform(train_df[['target']])
    fitted_estimator = fit_node.processing_func(train_df[['C', 'D']], train_labels)
    assert isinstance(fitted_estimator, KerasClassifier)
    assert isinstance(fit_node.make_classifier_func(), KerasClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_predictions = predict_node.processing_func(fitted_estimator, test_data)
    assert len(test_predictions) == 2


def test_accuracy_score():
    """
    Tests whether the monkey patching of ('pandas.core.series', 'test_series__logical_method') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.metrics import accuracy_score

                predictions = pd.Series([True, False, True, True], name='A')
                labels = pd.Series([True, False, False, True], name='B')
                metric = accuracy_score(y_pred=predictions, y_true=labels)
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
    expected_test_labels = DagNode(2,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.TEST_LABELS,
                                                   FunctionInfo('sklearn.metrics._classification',
                                                                'accuracy_score')),
                                   DagNodeDetails(None, ['B'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                             RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(6, 9, 6, 58),
                                                    'accuracy_score(y_pred=predictions, y_true=labels)'),
                                   Comparison(FunctionType))
    expected_score = DagNode(3,
                             BasicCodeLocation("<string-source>", 6),
                             OperatorContext(OperatorType.SCORE,
                                             FunctionInfo('sklearn.metrics._classification',
                                                          'accuracy_score')),
                             DagNodeDetails('accuracy_score', [],
                                            OptimizerInfo(RangeComparison(0, 200), (1, 1),
                                                          RangeComparison(0, 800))),
                             OptionalCodeInfo(CodeReference(6, 9, 6, 58),
                                              'accuracy_score(y_pred=predictions, y_true=labels)'),
                             Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source2, expected_test_labels, arg_index=0)
    expected_dag.add_edge(expected_data_source1, expected_score, arg_index=0)
    expected_dag.add_edge(expected_test_labels, expected_score, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    extracted_node = list(inspector_result.original_dag.nodes)[3]
    pd_series1 = pandas.Series([True, False, True, True], name='C')
    pd_series2 = pandas.Series([False, False, False, True], name='D')
    extracted_func_result = extracted_node.processing_func(pd_series1, pd_series2)
    assert 0.0 <= extracted_func_result <= 1.0


def test_grid_search_cv_sgd_classifier():
    """
    Tests whether the monkey patching of ('sklearn.linear_model._stochastic_gradient', 'SGDClassifier') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.linear_model import SGDClassifier
                from sklearn.model_selection import GridSearchCV
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3, 4, 5], 'B': [0, 1, 2, 3, 4, 5], 
                                   'target': ['no', 'no', 'no', 'yes', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                param_grid = {
                    'penalty': ['l2', 'l1'],
                }
                clf = GridSearchCV(SGDClassifier(loss='log', random_state=42), param_grid, cv=2)
                clf = clf.fit(train, target)

                test_predict = clf.predict(pd.DataFrame([[0., 0.], [0.6, 0.6]]))
                expected = np.array([0., 1.])
                assert np.allclose(test_predict, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    filter_dag_for_nodes_with_ids(inspector_result, {2, 5, 4, 6, 7}, 11)

    expected_dag = networkx.DiGraph()
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 10),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data',
                                                                    'StandardScaler')),
                                       DagNodeDetails('Standard Scaler: fit_transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (6, 2),
                                                                    RangeComparison(0, 3000))),
                                       OptionalCodeInfo(CodeReference(10, 8, 10, 24), 'StandardScaler()'),
                                       Comparison(FunctionType))
    expected_train_data = DagNode(5,
                                  BasicCodeLocation("<string-source>", 16),
                                  OperatorContext(OperatorType.TRAIN_DATA,
                                                  FunctionInfo('sklearn.linear_model._stochastic_gradient',
                                                               'SGDClassifier')),
                                  DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (6, 2),
                                                                                RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(16, 19, 16, 61),
                                                   "SGDClassifier(loss='log', random_state=42)"),
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_standard_scaler, expected_train_data, arg_index=0)
    expected_label_encode = DagNode(4,
                                    BasicCodeLocation("<string-source>", 11),
                                    OperatorContext(OperatorType.PROJECTION_MODIFY,
                                                    FunctionInfo('sklearn.preprocessing._label', 'label_binarize')),
                                    DagNodeDetails("label_binarize, classes: ['no', 'yes']", ['array'],
                                                   OptimizerInfo(RangeComparison(0, 200), (6, 1),
                                                                 RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(11, 9, 11, 60),
                                                     "label_binarize(df['target'], classes=['no', 'yes'])"),
                                    Comparison(FunctionType))
    expected_train_labels = DagNode(6,
                                    BasicCodeLocation("<string-source>", 16),
                                    OperatorContext(OperatorType.TRAIN_LABELS,
                                                    FunctionInfo('sklearn.linear_model._stochastic_gradient',
                                                                 'SGDClassifier')),
                                    DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (6, 1),
                                                                                  RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(16, 19, 16, 61),
                                                     "SGDClassifier(loss='log', random_state=42)"),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_train_labels, arg_index=0)
    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 16),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('sklearn.linear_model._stochastic_gradient',
                                                               'SGDClassifier')),
                                  DagNodeDetails('SGD Classifier', [],
                                                 OptimizerInfo(RangeComparison(0, 1000), None,
                                                               RangeComparison(0, 5000))),
                                  OptionalCodeInfo(CodeReference(16, 19, 16, 61),
                                                   "SGDClassifier(loss='log', random_state=42)"),
                                  Comparison(partial),
                                  Comparison(partial))
    expected_dag.add_edge(expected_train_data, expected_classifier, arg_index=0)
    expected_dag.add_edge(expected_train_labels, expected_classifier, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.original_dag.nodes)[4]
    train_data_node = list(inspector_result.original_dag.nodes)[2]
    train_label_node = list(inspector_result.original_dag.nodes)[3]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_data = train_data_node.processing_func(train_df[['C', 'D']])
    train_labels = label_binarize(train_df['target'], classes=['no', 'yes'])
    train_labels = train_label_node.processing_func(train_labels)
    fitted_estimator = fit_node.processing_func(train_data, train_labels)
    assert isinstance(fitted_estimator, GridSearchCV)
    assert isinstance(fitted_estimator.estimator, SGDClassifier)
    classifier = fit_node.make_classifier_func()
    assert isinstance(classifier, GridSearchCV)
    assert isinstance(classifier.estimator, SGDClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_score = fitted_estimator.score(test_df[['C', 'D']], test_labels)
    assert test_score == 0.5


def test_grid_search_cv_decision_tree():
    """
    Tests whether the monkey patching of DecisionTreeClassifier works with GridSearchCV
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.model_selection import GridSearchCV
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3, 4, 5], 'B': [0, 1, 2, 3, 4, 5], 
                                   'target': ['no', 'no', 'no', 'yes', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                param_grid = {
                    'criterion': ["gini", "entropy"],
                }
                clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=2)
                clf = clf.fit(train, target)

                test_predict = clf.predict(pd.DataFrame([[0., 0.], [0.6, 0.6]]))
                expected = np.array([0., 1.])
                assert np.allclose(test_predict, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    filter_dag_for_nodes_with_ids(inspector_result, {2, 5, 4, 6, 7}, 11)

    expected_dag = networkx.DiGraph()
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 10),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data',
                                                                    'StandardScaler')),
                                       DagNodeDetails('Standard Scaler: fit_transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (6, 2),
                                                                    RangeComparison(0, 3000))),
                                       OptionalCodeInfo(CodeReference(10, 8, 10, 24), 'StandardScaler()'),
                                       Comparison(FunctionType))
    expected_train_data = DagNode(5,
                                  BasicCodeLocation("<string-source>", 16),
                                  OperatorContext(OperatorType.TRAIN_DATA,
                                                  FunctionInfo('sklearn.tree._classes',
                                                               'DecisionTreeClassifier')),
                                  DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (6, 2),
                                                                                RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(16, 19, 16, 43),
                                                   "DecisionTreeClassifier()"),
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_standard_scaler, expected_train_data, arg_index=0)
    expected_label_encode = DagNode(4,
                                    BasicCodeLocation("<string-source>", 11),
                                    OperatorContext(OperatorType.PROJECTION_MODIFY,
                                                    FunctionInfo('sklearn.preprocessing._label', 'label_binarize')),
                                    DagNodeDetails("label_binarize, classes: ['no', 'yes']", ['array'],
                                                   OptimizerInfo(RangeComparison(0, 200), (6, 1),
                                                                 RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(11, 9, 11, 60),
                                                     "label_binarize(df['target'], classes=['no', 'yes'])"),
                                    Comparison(FunctionType))
    expected_train_labels = DagNode(6,
                                    BasicCodeLocation("<string-source>", 16),
                                    OperatorContext(OperatorType.TRAIN_LABELS,
                                                    FunctionInfo('sklearn.tree._classes',
                                                                 'DecisionTreeClassifier')),
                                    DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (6, 1),
                                                                                  RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(16, 19, 16, 43),
                                                     "DecisionTreeClassifier()"),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_train_labels, arg_index=0)
    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 16),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('sklearn.tree._classes',
                                                               'DecisionTreeClassifier')),
                                  DagNodeDetails('Decision Tree', [],
                                                 OptimizerInfo(RangeComparison(0, 1000), None,
                                                               RangeComparison(0, 5000))),
                                  OptionalCodeInfo(CodeReference(16, 19, 16, 43),
                                                   "DecisionTreeClassifier()"),
                                  Comparison(partial),
                                  Comparison(partial))
    expected_dag.add_edge(expected_train_data, expected_classifier, arg_index=0)
    expected_dag.add_edge(expected_train_labels, expected_classifier, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.original_dag.nodes)[4]
    train_data_node = list(inspector_result.original_dag.nodes)[2]
    train_label_node = list(inspector_result.original_dag.nodes)[3]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_data = train_data_node.processing_func(train_df[['C', 'D']])
    train_labels = label_binarize(train_df['target'], classes=['no', 'yes'])
    train_labels = train_label_node.processing_func(train_labels)
    fitted_estimator = fit_node.processing_func(train_data, train_labels)
    assert isinstance(fitted_estimator, GridSearchCV)
    assert isinstance(fitted_estimator.estimator, DecisionTreeClassifier)
    classifier = fit_node.make_classifier_func()
    assert isinstance(classifier, GridSearchCV)
    assert isinstance(classifier.estimator, DecisionTreeClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_score = fitted_estimator.score(test_df[['C', 'D']], test_labels)
    assert test_score == 0.5


def test_grid_search_cv_logistic_regression():
    """
    Tests whether the monkey patching of ('sklearn.linear_model._stochastic_gradient', 'SGDClassifier') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import GridSearchCV
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3, 4, 5], 'B': [0, 1, 2, 3, 4, 5], 
                                   'target': ['no', 'no', 'no', 'yes', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                param_grid = {
                    'penalty': ['l2', 'l1'],
                }
                clf = GridSearchCV(LogisticRegression(), param_grid, cv=2)
                clf = clf.fit(train, target)

                test_predict = clf.predict(pd.DataFrame([[0., 0.], [0.6, 0.6]]))
                expected = np.array([0., 1.])
                assert np.allclose(test_predict, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    filter_dag_for_nodes_with_ids(inspector_result, {2, 5, 4, 6, 7}, 11)

    expected_dag = networkx.DiGraph()
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 10),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data',
                                                                    'StandardScaler')),
                                       DagNodeDetails('Standard Scaler: fit_transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (6, 2),
                                                                    RangeComparison(0, 3000))),
                                       OptionalCodeInfo(CodeReference(10, 8, 10, 24), 'StandardScaler()'),
                                       Comparison(FunctionType))
    expected_train_data = DagNode(5,
                                  BasicCodeLocation("<string-source>", 16),
                                  OperatorContext(OperatorType.TRAIN_DATA,
                                                  FunctionInfo('sklearn.linear_model._logistic',
                                                               'LogisticRegression')),
                                  DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (6, 2),
                                                                                RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(16, 19, 16, 39),
                                                   "LogisticRegression()"),
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_standard_scaler, expected_train_data, arg_index=0)
    expected_label_encode = DagNode(4,
                                    BasicCodeLocation("<string-source>", 11),
                                    OperatorContext(OperatorType.PROJECTION_MODIFY,
                                                    FunctionInfo('sklearn.preprocessing._label', 'label_binarize')),
                                    DagNodeDetails("label_binarize, classes: ['no', 'yes']", ['array'],
                                                   OptimizerInfo(RangeComparison(0, 200), (6, 1),
                                                                 RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(11, 9, 11, 60),
                                                     "label_binarize(df['target'], classes=['no', 'yes'])"),
                                    Comparison(FunctionType))
    expected_train_labels = DagNode(6,
                                    BasicCodeLocation("<string-source>", 16),
                                    OperatorContext(OperatorType.TRAIN_LABELS,
                                                    FunctionInfo('sklearn.linear_model._logistic',
                                                                 'LogisticRegression')),
                                    DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (6, 1),
                                                                                  RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(16, 19, 16, 39),
                                                     "LogisticRegression()"),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_train_labels, arg_index=0)
    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 16),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('sklearn.linear_model._logistic',
                                                               'LogisticRegression')),
                                  DagNodeDetails('Logistic Regression', [],
                                                 OptimizerInfo(RangeComparison(0, 1000), None,
                                                               RangeComparison(0, 5000))),
                                  OptionalCodeInfo(CodeReference(16, 19, 16, 39),
                                                   "LogisticRegression()"),
                                  Comparison(partial),
                                  Comparison(partial))
    expected_dag.add_edge(expected_train_data, expected_classifier, arg_index=0)
    expected_dag.add_edge(expected_train_labels, expected_classifier, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.original_dag.nodes)[4]
    train_data_node = list(inspector_result.original_dag.nodes)[2]
    train_label_node = list(inspector_result.original_dag.nodes)[3]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_data = train_data_node.processing_func(train_df[['C', 'D']])
    train_labels = label_binarize(train_df['target'], classes=['no', 'yes'])
    train_labels = train_label_node.processing_func(train_labels)
    fitted_estimator = fit_node.processing_func(train_data, train_labels)
    assert isinstance(fitted_estimator, GridSearchCV)
    assert isinstance(fitted_estimator.estimator, LogisticRegression)
    classifier = fit_node.make_classifier_func()
    assert isinstance(classifier, GridSearchCV)
    assert isinstance(classifier.estimator, LogisticRegression)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_score = fitted_estimator.score(test_df[['C', 'D']], test_labels)
    assert test_score == 0.5


def test_grid_search_cv_keras_wrapper():
    """
    Tests whether the monkey patching of the KerasClassifier works with GridSearchCV
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import StandardScaler, OneHotEncoder
                from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
                from tensorflow.keras.layers import Dense
                from tensorflow.keras.models import Sequential
                from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
                from sklearn.linear_model import LogisticRegression
                from sklearn.model_selection import GridSearchCV
                import tensorflow as tf
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3, 4, 5], 'B': [0, 1, 2, 3, 4, 5], 
                                   'target': ['no', 'no', 'no', 'yes', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = OneHotEncoder(sparse=False).fit_transform(df[['target']])

                param_grid = {
                    'epochs': [10, 15],
                }
                
                def create_model(input_dim):
                    clf = Sequential()
                    clf.add(Dense(2, activation='relu', input_dim=input_dim))
                    clf.add(Dense(2, activation='relu'))
                    clf.add(Dense(2, activation='softmax'))
                    clf.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=["accuracy"])
                    return clf

                np.random.seed(42)
                tf.random.set_seed(42)
                
                clf = KerasClassifier(build_fn=create_model, batch_size=1, verbose=0, input_dim=2)
                clf = GridSearchCV(clf, param_grid, cv=2)
                clf = clf.fit(train, target)

                test_predict = clf.predict(pd.DataFrame([[0., 0.], [0.6, 0.6]]))
                assert test_predict.shape == (2,)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    filter_dag_for_nodes_with_ids(inspector_result, {2, 5, 4, 6, 7}, 11)

    expected_dag = networkx.DiGraph()
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 15),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data',
                                                                    'StandardScaler')),
                                       DagNodeDetails('Standard Scaler: fit_transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 1000), (6, 2),
                                                                    RangeComparison(0, 4000))),
                                       OptionalCodeInfo(CodeReference(15, 8, 15, 24), 'StandardScaler()'),
                                       Comparison(FunctionType))
    expected_train_data = DagNode(5,
                                  BasicCodeLocation("<string-source>", 27),
                                  OperatorContext(OperatorType.TRAIN_DATA,
                                                  FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn',
                                                               'KerasClassifier')),
                                  DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 1000), (6, 2),
                                                                                RangeComparison(0, 4000))),
                                  OptionalCodeInfo(CodeReference(27, 4, 27, 87),
                                                   'clf.compile(loss=\'categorical_crossentropy\', '
                                                   'optimizer=SGD(), metrics=["accuracy"])'),
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_standard_scaler, expected_train_data, arg_index=0)
    expected_label_encode = DagNode(4,
                                    BasicCodeLocation("<string-source>", 16),
                                    OperatorContext(OperatorType.TRANSFORMER,
                                                    FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')),
                                    DagNodeDetails('One-Hot Encoder: fit_transform', ['array'],
                                                   OptimizerInfo(RangeComparison(0, 800), (6, 2),
                                                                 RangeComparison(0, 5000))),
                                    OptionalCodeInfo(CodeReference(16, 9, 16, 36),
                                                     'OneHotEncoder(sparse=False)'),
                                    Comparison(FunctionType))
    expected_train_labels = DagNode(6,
                                    BasicCodeLocation("<string-source>", 27),
                                    OperatorContext(OperatorType.TRAIN_LABELS,
                                                    FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn',
                                                                 'KerasClassifier')),
                                    DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 800), (6, 2),
                                                                                  RangeComparison(0, 2000))),
                                    OptionalCodeInfo(CodeReference(27, 4, 27, 87),
                                                     'clf.compile(loss=\'categorical_crossentropy\', '
                                                     'optimizer=SGD(), metrics=["accuracy"])'),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_train_labels, arg_index=0)
    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 27),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn',
                                                               'KerasClassifier')),
                                  DagNodeDetails('Neural Network', [],
                                                 OptimizerInfo(RangeComparison(0, 10000), None,
                                                               RangeComparison(0, 8000))),
                                  OptionalCodeInfo(CodeReference(27, 4, 27, 87),
                                                   'clf.compile(loss=\'categorical_crossentropy\', '
                                                   'optimizer=SGD(), metrics=["accuracy"])'),
                                  Comparison(partial),
                                  Comparison(partial))
    expected_dag.add_edge(expected_train_data, expected_classifier, arg_index=0)
    expected_dag.add_edge(expected_train_labels, expected_classifier, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.original_dag.nodes)[4]
    train_data_node = list(inspector_result.original_dag.nodes)[2]
    train_label_node = list(inspector_result.original_dag.nodes)[3]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_data = train_data_node.processing_func(train_df[['C', 'D']])
    train_labels = OneHotEncoder(sparse=False).fit_transform(train_df[['target']])
    train_labels = train_label_node.processing_func(train_labels)
    fitted_estimator = fit_node.processing_func(train_data, train_labels)
    assert isinstance(fitted_estimator, GridSearchCV)
    assert isinstance(fitted_estimator.estimator, KerasClassifier)
    classifier = fit_node.make_classifier_func()
    assert isinstance(classifier, GridSearchCV)
    assert isinstance(classifier.estimator, KerasClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_labels = OneHotEncoder(sparse=False).fit_transform(test_df[['target']])
    test_score = fitted_estimator.score(test_df[['C', 'D']], test_labels)
    assert test_score >= 0.0


def test_dummy_classifier():
    """
    Tests whether the monkey patching of ('sklearn.tree._classes', 'DecisionTreeClassifier') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.dummy import DummyClassifier
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = DummyClassifier(strategy='constant', constant=0.)
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
                                                  FunctionInfo('sklearn.dummy', 'DummyClassifier')),
                                  DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                                RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 55),
                                                   "DummyClassifier(strategy='constant', constant=0.)"),
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_standard_scaler, expected_train_data, arg_index=0)
    expected_train_labels = DagNode(6,
                                    BasicCodeLocation("<string-source>", 11),
                                    OperatorContext(OperatorType.TRAIN_LABELS,
                                                    FunctionInfo('sklearn.dummy', 'DummyClassifier')),
                                    DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                  RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(11, 6, 11, 55),
                                                     "DummyClassifier(strategy='constant', constant=0.)"),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_train_labels, arg_index=0)
    expected_decision_tree = DagNode(7,
                                     BasicCodeLocation("<string-source>", 11),
                                     OperatorContext(OperatorType.ESTIMATOR,
                                                     FunctionInfo('sklearn.dummy', 'DummyClassifier')),
                                     DagNodeDetails('Dummy Classifier', [],
                                                    OptimizerInfo(RangeComparison(0, 1000), None,
                                                                  RangeComparison(0, 10000))),
                                     OptionalCodeInfo(CodeReference(11, 6, 11, 55),
                                                      "DummyClassifier(strategy='constant', constant=0.)"),
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
    assert isinstance(fitted_estimator, DummyClassifier)
    assert isinstance(fit_node.make_classifier_func(), DummyClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_score = fitted_estimator.score(test_df[['C', 'D']], test_labels)
    assert test_score == 0.5


def test_dummy_classifier_score():
    """
    Tests whether the monkey patching of ('sklearn.tree._classes.DecisionTreeClassifier', 'score') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.dummy import DummyClassifier
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = DummyClassifier(strategy='constant', constant=0.)
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
                                                 FunctionInfo('sklearn.dummy.DummyClassifier', 'score')),
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
                                                   FunctionInfo('sklearn.dummy.DummyClassifier', 'score')),
                                   DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (2, 1),
                                                                                 RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                                    "clf.score(test_df[['A', 'B']], test_labels)"),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_test_labels, arg_index=0)
    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 11),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('sklearn.dummy', 'DummyClassifier')),
                                  DagNodeDetails('Dummy Classifier', [], OptimizerInfo(RangeComparison(0, 1000), None,
                                                                                       RangeComparison(0, 10000))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 55),
                                                   "DummyClassifier(strategy='constant', constant=0.)"),
                                  Comparison(FunctionType),
                                  Comparison(partial))
    expected_predict = DagNode(14,
                               BasicCodeLocation("<string-source>", 16),
                               OperatorContext(OperatorType.PREDICT,
                                               FunctionInfo('sklearn.dummy.DummyClassifier', 'score')),
                               DagNodeDetails('Dummy Classifier', [], OptimizerInfo(RangeComparison(0, 1000), (2, 1),
                                                                                    RangeComparison(0, 800))),
                               OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                                "clf.score(test_df[['A', 'B']], test_labels)"),
                               Comparison(FunctionType))
    expected_dag.add_edge(expected_classifier, expected_predict, arg_index=0)
    expected_dag.add_edge(expected_test_data, expected_predict, arg_index=1)

    expected_score = DagNode(15,
                             BasicCodeLocation("<string-source>", 16),
                             OperatorContext(OperatorType.SCORE,
                                             FunctionInfo('sklearn.dummy.DummyClassifier', 'score')),
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
    assert isinstance(fitted_estimator, DummyClassifier)
    assert isinstance(fit_node.make_classifier_func(), DummyClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_labels = test_label_node.processing_func(test_labels)
    test_predictions = predict_node.processing_func(fitted_estimator, test_data)
    test_score = score_node.processing_func(test_predictions, test_labels)
    assert test_score == 0.5


def test_dummy_classifier_predict():
    """
    Tests whether the monkey patching of ('sklearn.tree._classes.DecisionTreeClassifier', 'predict') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.dummy import DummyClassifier
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = DummyClassifier(strategy='constant', constant=0.)
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
                                                 FunctionInfo('sklearn.dummy.DummyClassifier',
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
                                                  FunctionInfo('sklearn.dummy', 'DummyClassifier')),
                                  DagNodeDetails('Dummy Classifier', [], OptimizerInfo(RangeComparison(0, 1000), None,
                                                                                       RangeComparison(0, 10000))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 55),
                                                   "DummyClassifier(strategy='constant', constant=0.)"),
                                  Comparison(FunctionType),
                                  Comparison(partial))
    expected_predict = DagNode(11,
                               BasicCodeLocation("<string-source>", 15),
                               OperatorContext(OperatorType.PREDICT,
                                               FunctionInfo('sklearn.dummy.DummyClassifier', 'predict')),
                               DagNodeDetails('Dummy Classifier', [], OptimizerInfo(RangeComparison(0, 1000), (2, 1),
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
    assert isinstance(fitted_estimator, DummyClassifier)
    assert isinstance(fit_node.make_classifier_func(), DummyClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_predict = predict_node.processing_func(fitted_estimator, test_data)
    assert len(test_predict) == 2


def test_svc():
    """
    Tests whether the monkey patching of ('sklearn.tree._classes', 'DecisionTreeClassifier') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.svm import SVC
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = SVC(random_state=42)
                clf = clf.fit(train, target)

                test_predict = clf.predict(pd.DataFrame([[0., 0.], [0.6, 0.6]], columns=['A', 'B']))
                print(test_predict)
                expected = np.array([1., 1.])
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
                                                  FunctionInfo('sklearn.svm._classes', 'SVC')),
                                  DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                                RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 26), 'SVC(random_state=42)'),
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_standard_scaler, expected_train_data, arg_index=0)
    expected_train_labels = DagNode(6,
                                    BasicCodeLocation("<string-source>", 11),
                                    OperatorContext(OperatorType.TRAIN_LABELS,
                                                    FunctionInfo('sklearn.svm._classes', 'SVC')),
                                    DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                                  RangeComparison(0, 800))),
                                    OptionalCodeInfo(CodeReference(11, 6, 11, 26), 'SVC(random_state=42)'),
                                    Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_train_labels, arg_index=0)
    expected_decision_tree = DagNode(7,
                                     BasicCodeLocation("<string-source>", 11),
                                     OperatorContext(OperatorType.ESTIMATOR,
                                                     FunctionInfo('sklearn.svm._classes', 'SVC')),
                                     DagNodeDetails('SVC', [],
                                                    OptimizerInfo(RangeComparison(0, 1000), None,
                                                                  RangeComparison(0, 10000))),
                                     OptionalCodeInfo(CodeReference(11, 6, 11, 26),
                                                      'SVC(random_state=42)'),
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
    assert isinstance(fitted_estimator, SVC)
    assert isinstance(fit_node.make_classifier_func(), SVC)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_score = fitted_estimator.score(test_df[['C', 'D']], test_labels)
    assert test_score == 0.5


def test_svc_score():
    """
    Tests whether the monkey patching of ('sklearn.tree._classes.DecisionTreeClassifier', 'score') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.svm import SVC
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = SVC(random_state=42)
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
                                                 FunctionInfo('sklearn.svm._classes.SVC', 'score')),
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
                                                   FunctionInfo('sklearn.svm._classes.SVC', 'score')),
                                   DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 200), (2, 1),
                                                                                 RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                                    "clf.score(test_df[['A', 'B']], test_labels)"),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_label_encode, expected_test_labels, arg_index=0)
    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 11),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('sklearn.svm._classes', 'SVC')),
                                  DagNodeDetails('SVC', [], OptimizerInfo(RangeComparison(0, 1000), None,
                                                                          RangeComparison(0, 10000))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 26), 'SVC(random_state=42)'),
                                  Comparison(FunctionType),
                                  Comparison(partial))
    expected_predict = DagNode(14,
                               BasicCodeLocation("<string-source>", 16),
                               OperatorContext(OperatorType.PREDICT,
                                               FunctionInfo('sklearn.svm._classes.SVC', 'score')),
                               DagNodeDetails('SVC', [], OptimizerInfo(RangeComparison(0, 1000), (2, 1),
                                                                       RangeComparison(0, 800))),
                               OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                                "clf.score(test_df[['A', 'B']], test_labels)"),
                               Comparison(FunctionType))
    expected_dag.add_edge(expected_classifier, expected_predict, arg_index=0)
    expected_dag.add_edge(expected_test_data, expected_predict, arg_index=1)

    expected_score = DagNode(15,
                             BasicCodeLocation("<string-source>", 16),
                             OperatorContext(OperatorType.SCORE,
                                             FunctionInfo('sklearn.svm._classes.SVC', 'score')),
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
    assert isinstance(fitted_estimator, SVC)
    assert isinstance(fit_node.make_classifier_func(), SVC)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_labels = test_label_node.processing_func(test_labels)
    test_predictions = predict_node.processing_func(fitted_estimator, test_data)
    test_score = score_node.processing_func(test_predictions, test_labels)
    assert test_score == 0.5


def test_svc_predict():
    """
    Tests whether the monkey patching of ('sklearn.tree._classes.DecisionTreeClassifier', 'predict') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler
                from sklearn.svm import SVC
                import numpy as np

                df = pd.DataFrame({'A': [0, 1, 2, 3], 'B': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})

                train = StandardScaler().fit_transform(df[['A', 'B']])
                target = label_binarize(df['target'], classes=['no', 'yes'])

                clf = SVC(random_state=42)
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
                                                 FunctionInfo('sklearn.svm._classes.SVC', 'predict')),
                                 DagNodeDetails(None, ['A', 'B'], OptimizerInfo(RangeComparison(0, 200), (2, 2),
                                                                                RangeComparison(0, 800))),
                                 OptionalCodeInfo(CodeReference(15, 14, 15, 46),
                                                  "clf.predict(test_df[['A', 'B']])"),
                                 Comparison(FunctionType))
    expected_dag.add_edge(expected_data_projection, expected_test_data, arg_index=0)
    expected_classifier = DagNode(7,
                                  BasicCodeLocation("<string-source>", 11),
                                  OperatorContext(OperatorType.ESTIMATOR,
                                                  FunctionInfo('sklearn.svm._classes', 'SVC')),
                                  DagNodeDetails('SVC', [], OptimizerInfo(RangeComparison(0, 1000), None,
                                                                          RangeComparison(0, 10000))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 26), 'SVC(random_state=42)'),
                                  Comparison(FunctionType),
                                  Comparison(partial))
    expected_predict = DagNode(11,
                               BasicCodeLocation("<string-source>", 15),
                               OperatorContext(OperatorType.PREDICT,
                                               FunctionInfo('sklearn.svm._classes.SVC', 'predict')),
                               DagNodeDetails('SVC', [], OptimizerInfo(RangeComparison(0, 1000), (2, 1),
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
    assert isinstance(fitted_estimator, SVC)
    assert isinstance(fit_node.make_classifier_func(), SVC)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_predict = predict_node.processing_func(fitted_estimator, test_data)
    assert len(test_predict) == 2


def test_count_vectorizer():
    """
    Tests whether the monkey patching of ('sklearn.compose._column_transformer', 'ColumnTransformer') works with
    multiple transformers with dense output
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.feature_extraction.text import CountVectorizer
                import numpy
                from scipy.sparse import csr_matrix
                df = pd.Series(['cat_a', 'cat_b', 'cat_a', 'cat_c'], name='A').to_numpy()
                transformer = CountVectorizer()
                transformed_data = transformer.fit_transform(df)
                test_transform_function = transformer.transform(df)
                print(transformed_data)
                expected = numpy.array([[1., 0., 0.],
                                        [0., 1., 0.],
                                        [1., 0., 0.],
                                        [0., 0., 1.]])
                expected = csr_matrix([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
                assert numpy.allclose(transformed_data.A, expected.A) and isinstance(transformed_data, csr_matrix)
                assert numpy.allclose(test_transform_function.A, expected.A) and isinstance(test_transform_function, 
                    csr_matrix)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 5),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.series', 'Series')),
                                   DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                             RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(5, 5, 5, 62),
                                                    "pd.Series(['cat_a', 'cat_b', 'cat_a', 'cat_c'], name='A')"),
                                   Comparison(partial))
    expected_project = DagNode(1,
                               BasicCodeLocation('<string-source>', 5),
                               OperatorContext(OperatorType.PROJECTION,
                                               FunctionInfo('pandas.core.series.Series', 'to_numpy')),
                               DagNodeDetails('numpy conversion', ['array'], OptimizerInfo(RangeComparison(0, 200),
                                                                                           (4, 1),
                                                                                           RangeComparison(0, 800))),
                               OptionalCodeInfo(CodeReference(5, 5, 5, 73),
                                                "pd.Series(['cat_a', 'cat_b', 'cat_a', 'cat_c'], name='A').to_numpy()"),
                               Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_project, arg_index=0)
    expected_transformer = DagNode(2,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('sklearn.feature_extraction.text', 'CountVectorizer')),
                                   DagNodeDetails('Count Vectorizer: fit_transform', ['array'],
                                                  OptimizerInfo(RangeComparison(0, 200), (4, 3),
                                                                RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(6, 14, 6, 31),
                                                    'CountVectorizer()'),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_project, expected_transformer, arg_index=0)
    expected_transform_test = DagNode(3,
                                      BasicCodeLocation("<string-source>", 6),
                                      OperatorContext(OperatorType.TRANSFORMER,
                                                      FunctionInfo('sklearn.feature_extraction.text',
                                                                   'CountVectorizer')),
                                      DagNodeDetails('Count Vectorizer: transform', ['array']),
                                      OptionalCodeInfo(CodeReference(6, 14, 6, 31),
                                                       'CountVectorizer()'),
                                      Comparison(FunctionType))
    expected_dag.add_edge(expected_transformer, expected_transform_test, arg_index=0)
    expected_dag.add_edge(expected_project, expected_transform_test, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.original_dag.nodes)[2]
    transform_node = list(inspector_result.original_dag.nodes)[3]
    pandas_df = pandas.Series(['cat_a', 'cat_b', 'cat_c', 'cat_c'], name='A').to_numpy()
    fit_transformed_result = fit_transform_node.processing_func(pandas_df)
    expected_fit_transform_data = csr_matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 1.]])
    assert numpy.allclose(fit_transformed_result.A, expected_fit_transform_data.A) and \
           isinstance(fit_transformed_result, csr_matrix)

    test_df = pandas.Series(['cat_c', 'cat_c', 'cat_b', 'cat_c'], name='A').to_numpy()
    encoded_data = transform_node.processing_func(fit_transformed_result, test_df)
    expected = csr_matrix([[0., 0., 1.], [0., 0., 1.], [0., 1., 0.], [0., 0., 1.]])
    assert numpy.allclose(encoded_data.A, expected.A) and \
           isinstance(encoded_data, csr_matrix)


def test_tfidf_vectorizer():
    """
    Tests whether the monkey patching of ('sklearn.compose._column_transformer', 'ColumnTransformer') works with
    multiple transformers with dense output
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.feature_extraction.text import  CountVectorizer, TfidfTransformer
                import numpy
                from scipy.sparse import csr_matrix
                df = CountVectorizer().fit_transform(
                    pd.Series(['cat_a', 'cat_b', 'cat_a', 'cat_c'], name='A').to_numpy())
                transformer = TfidfTransformer()
                transformed_data = transformer.fit_transform(df)
                test_transform_function = transformer.transform(df)
                print(transformed_data)
                expected = numpy.array([[1., 0., 0.],
                                        [0., 1., 0.],
                                        [1., 0., 0.],
                                        [0., 0., 1.]])
                expected = csr_matrix([[1., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.]])
                assert numpy.allclose(transformed_data.A, expected.A) and isinstance(transformed_data, csr_matrix)
                assert numpy.allclose(test_transform_function.A, expected.A) and isinstance(test_transform_function, 
                    csr_matrix)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.series', 'Series')),
                                   DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 200), (4, 1),
                                                                             RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(6, 4, 6, 61),
                                                    "pd.Series(['cat_a', 'cat_b', 'cat_a', 'cat_c'], name='A')"),
                                   Comparison(partial))
    expected_project = DagNode(1,
                               BasicCodeLocation('<string-source>', 6),
                               OperatorContext(OperatorType.PROJECTION,
                                               FunctionInfo('pandas.core.series.Series', 'to_numpy')),
                               DagNodeDetails('numpy conversion', ['array'],
                                              OptimizerInfo(RangeComparison(0, 200), (4, 1), RangeComparison(0, 800))),
                               OptionalCodeInfo(CodeReference(6, 4, 6, 72),
                                                "pd.Series(['cat_a', 'cat_b', 'cat_a', 'cat_c'], name='A').to_numpy()"),
                               Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_project, arg_index=0)
    expected_count_vectorizer = DagNode(2,
                                        BasicCodeLocation("<string-source>", 5),
                                        OperatorContext(OperatorType.TRANSFORMER,
                                                        FunctionInfo('sklearn.feature_extraction.text',
                                                                     'CountVectorizer')),
                                        DagNodeDetails('Count Vectorizer: fit_transform', ['array'],
                                                       OptimizerInfo(RangeComparison(0, 2000), (4, 3),
                                                                     RangeComparison(0, 2000))),
                                        OptionalCodeInfo(CodeReference(5, 5, 5, 22),
                                                         'CountVectorizer()'),
                                        Comparison(FunctionType))
    expected_dag.add_edge(expected_project, expected_count_vectorizer, arg_index=0)
    expected_transformer = DagNode(3,
                                   BasicCodeLocation("<string-source>", 7),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('sklearn.feature_extraction.text', 'TfidfTransformer')),
                                   DagNodeDetails('Tfidf Transformer: fit_transform', ['array'],
                                                  OptimizerInfo(RangeComparison(0, 2000), (4, 3),
                                                                RangeComparison(0, 2000))),
                                   OptionalCodeInfo(CodeReference(7, 14, 7, 32),
                                                    'TfidfTransformer()'),
                                   Comparison(FunctionType)
                                   )
    expected_dag.add_edge(expected_count_vectorizer, expected_transformer, arg_index=0)
    expected_transform_test = DagNode(4,
                                      BasicCodeLocation("<string-source>", 7),
                                      OperatorContext(OperatorType.TRANSFORMER,
                                                      FunctionInfo('sklearn.feature_extraction.text',
                                                                   'TfidfTransformer')),
                                      DagNodeDetails('Tfidf Transformer: transform', ['array'],
                                                     OptimizerInfo(RangeComparison(0, 2000), (4, 3),
                                                                   RangeComparison(0, 2000))),
                                      OptionalCodeInfo(CodeReference(7, 14, 7, 32), 'TfidfTransformer()'),
                                      Comparison(FunctionType)
                                      )
    expected_dag.add_edge(expected_transformer, expected_transform_test, arg_index=0)
    expected_dag.add_edge(expected_count_vectorizer, expected_transform_test, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    count_vectorizer = list(inspector_result.original_dag.nodes)[2]
    fit_transform_node = list(inspector_result.original_dag.nodes)[3]
    transform_node = list(inspector_result.original_dag.nodes)[4]
    pandas_df = pandas.Series(['cat_a', 'cat_b', 'cat_c', 'cat_c'], name='A').to_numpy()
    count_vectorizer_result = count_vectorizer.processing_func(pandas_df)
    fit_transformed_result = fit_transform_node.processing_func(count_vectorizer_result)
    expected_fit_transform_data = csr_matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 1.]])
    assert numpy.allclose(fit_transformed_result.A, expected_fit_transform_data.A) and \
           isinstance(fit_transformed_result, csr_matrix)

    test_df = pandas.Series(['cat_c', 'cat_c', 'cat_b', 'cat_a'], name='A').to_numpy()
    count_vectorizer_result = count_vectorizer.processing_func(test_df)
    encoded_data = transform_node.processing_func(fit_transformed_result, count_vectorizer_result)
    expected = csr_matrix([[0., 0., 1.], [0., 0., 1.], [0., 1., 0.], [1., 0., 0.]])
    assert numpy.allclose(encoded_data.A, expected.A) and \
           isinstance(encoded_data, csr_matrix)


def test_truncated_svd():
    """
    Tests whether the monkey patching of ('sklearn.compose._column_transformer', 'ColumnTransformer') works with
    multiple transformers with dense output
    """
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder
                from sklearn.compose import ColumnTransformer
                from sklearn.decomposition import TruncatedSVD
                from scipy.sparse import csr_matrix
                import numpy

                df = pd.DataFrame({'A': [1, 2, 10, 5], 'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                column_transformer = ColumnTransformer(transformers=[
                    ('numeric', StandardScaler(), ['A']),
                    ('categorical', OneHotEncoder(sparse=False), ['B'])
                ])
                encoded_data = column_transformer.fit_transform(df)
                transformer = TruncatedSVD(n_iter=1, random_state=42)
                reduced_data = transformer.fit_transform(encoded_data)
                test_transform_function = transformer.transform(encoded_data)
                expected = numpy.array([[-0.7135186,   1.16732311],
                                        [-0.88316363,  0.30897343],
                                        [ 1.72690506,  0.64664575],
                                        [ 0.17663273, -0.06179469]])
                assert numpy.allclose(reduced_data, expected)
                assert numpy.allclose(test_transform_function, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    filter_dag_for_nodes_with_ids(inspector_result, [5, 6, 7], 8)

    expected_dag = networkx.DiGraph()
    expected_concat = DagNode(5,
                              BasicCodeLocation("<string-source>", 9),
                              OperatorContext(OperatorType.CONCATENATION,
                                              FunctionInfo('sklearn.compose._column_transformer', 'ColumnTransformer')),
                              DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 2000), (4, 4),
                                                                            RangeComparison(0, 2000))),
                              OptionalCodeInfo(CodeReference(9, 21, 12, 2),
                                               "ColumnTransformer(transformers=[\n"
                                               "    ('numeric', StandardScaler(), ['A']),\n"
                                               "    ('categorical', OneHotEncoder(sparse=False), ['B'])\n])"),
                              Comparison(FunctionType))
    expected_transformer = DagNode(6,
                                   BasicCodeLocation("<string-source>", 14),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('sklearn.decomposition._truncated_svd',
                                                                'TruncatedSVD')),
                                   DagNodeDetails('Truncated SVD: fit_transform', ['array'],
                                                  OptimizerInfo(RangeComparison(0, 2000), (4, 2),
                                                                RangeComparison(0, 2000))),
                                   OptionalCodeInfo(CodeReference(14, 14, 14, 53),
                                                    'TruncatedSVD(n_iter=1, random_state=42)'),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_concat, expected_transformer, arg_index=0)
    expected_transform_test = DagNode(7,
                                      BasicCodeLocation("<string-source>", 14),
                                      OperatorContext(OperatorType.TRANSFORMER,
                                                      FunctionInfo('sklearn.decomposition._truncated_svd',
                                                                   'TruncatedSVD')),
                                      DagNodeDetails('Truncated SVD: transform', ['array'],
                                                     OptimizerInfo(RangeComparison(0, 2000), (4, 2),
                                                                   RangeComparison(0, 2000))),
                                      OptionalCodeInfo(CodeReference(14, 14, 14, 53),
                                                       'TruncatedSVD(n_iter=1, random_state=42)'),
                                      Comparison(FunctionType))
    expected_dag.add_edge(expected_transformer, expected_transform_test, arg_index=0)
    expected_dag.add_edge(expected_concat, expected_transform_test, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.original_dag.nodes)[1]
    transform_node = list(inspector_result.original_dag.nodes)[2]
    pandas_df = numpy.array([[2., 4., 3.], [0., 2., 2.], [0., 0., 1.], [4., 4., 3.]])
    fit_transformed_result = fit_transform_node.processing_func(pandas_df)
    expected_fit_transform_data = numpy.array([[5.34468848, 0.60472355],
                                               [2.46206848, 1.3921952],
                                               [0.54114431, 0.47790364],
                                               [6.30638433, -1.09703966]])
    assert numpy.allclose(fit_transformed_result, expected_fit_transform_data)

    test_df = numpy.array([[0., 2., 2.], [0., 0., 1.], [4., 4., 3.], [2., 4., 3.]])
    encoded_data = transform_node.processing_func(fit_transformed_result, test_df)
    expected = numpy.array([[2.46206848, 1.3921952],
                            [0.54114431, 0.47790364],
                            [6.30638433, -1.09703966],
                            [5.34468848, 0.60472355]])
    assert numpy.allclose(encoded_data, expected)


def test_feature_union():
    """
    Tests whether the monkey patching of ('sklearn.preprocessing._data', 'StandardScaler') works
    """
    # pylint: disable=too-many-locals
    test_code = cleandoc("""
                import pandas as pd
                from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder
                from sklearn.compose import ColumnTransformer
                from sklearn.pipeline import FeatureUnion
                from sklearn.decomposition import PCA
                from sklearn.decomposition import TruncatedSVD
                from scipy.sparse import csr_matrix
                import numpy
                
                df = pd.DataFrame({'A': [1, 2, 10, 5], 'B': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                column_transformer = ColumnTransformer(transformers=[
                    ('numeric', StandardScaler(), ['A']),
                    ('categorical', OneHotEncoder(sparse=False), ['B'])
                ])
                encoded_data = column_transformer.fit_transform(df)
                transformer = FeatureUnion([('pca', PCA(n_components=2, random_state=42)),
                                            ('svd', TruncatedSVD(n_iter=1, random_state=42))])
                reduced_data = transformer.fit_transform(encoded_data)
                test_transform_function = transformer.transform(encoded_data)
                print(reduced_data)
                expected = numpy.array([[-0.80795996, -0.75406407, -0.7135186,   1.16732311],
                                        [-0.953227,    0.23384447, -0.88316363,  0.30897343],
                                        [ 1.64711991, -0.29071836,  1.72690506,  0.64664575],
                                        [ 0.11406705,  0.81093796,  0.17663273, -0.06179469]])
                assert numpy.allclose(reduced_data, expected)
                assert numpy.allclose(test_transform_function, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    filter_dag_for_nodes_with_ids(inspector_result, [5, 6, 7, 8, 9, 10, 11], 12)

    expected_dag = networkx.DiGraph()
    expected_concat = DagNode(5,
                              BasicCodeLocation("<string-source>", 11),
                              OperatorContext(OperatorType.CONCATENATION,
                                              FunctionInfo('sklearn.compose._column_transformer', 'ColumnTransformer')),
                              DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 2000), (4, 4),
                                                                            RangeComparison(0, 2000))),
                              OptionalCodeInfo(CodeReference(11, 21, 14, 2),
                                               "ColumnTransformer(transformers=[\n"
                                               "    ('numeric', StandardScaler(), ['A']),\n"
                                               "    ('categorical', OneHotEncoder(sparse=False), ['B'])\n])"),
                              Comparison(FunctionType))
    expected_transformer_pca = DagNode(6,
                                       BasicCodeLocation("<string-source>", 16),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.decomposition._pca',
                                                                    'PCA')),
                                       DagNodeDetails('PCA: fit_transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(16, 36, 16, 72),
                                                        'PCA(n_components=2, random_state=42)'),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_concat, expected_transformer_pca, arg_index=0)
    expected_transformer_svd = DagNode(7,
                                       BasicCodeLocation("<string-source>", 17),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.decomposition._truncated_svd',
                                                                    'TruncatedSVD')),
                                       DagNodeDetails('Truncated SVD: fit_transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(17, 36, 17, 75),
                                                        'TruncatedSVD(n_iter=1, random_state=42)'),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_concat, expected_transformer_svd, arg_index=0)
    expected_transformer_concat = DagNode(8,
                                          BasicCodeLocation("<string-source>", 16),
                                          OperatorContext(OperatorType.CONCATENATION,
                                                          FunctionInfo('sklearn.pipeline', 'FeatureUnion')),
                                          DagNodeDetails('Feature Union', ['array'],
                                                         OptimizerInfo(RangeComparison(0, 200), (4, 4),
                                                                       RangeComparison(0, 800))),
                                          OptionalCodeInfo(CodeReference(16, 14, 17, 78),
                                                           "FeatureUnion([('pca', PCA(n_components=2, "
                                                           "random_state=42)),\n                            "
                                                           "('svd', TruncatedSVD(n_iter=1, random_state=42))])"),
                                          Comparison(FunctionType))
    expected_dag.add_edge(expected_transformer_pca, expected_transformer_concat, arg_index=0)
    expected_dag.add_edge(expected_transformer_svd, expected_transformer_concat, arg_index=1)
    expected_transform_test_pca = DagNode(9,
                                          BasicCodeLocation("<string-source>", 16),
                                          OperatorContext(OperatorType.TRANSFORMER,
                                                          FunctionInfo('sklearn.decomposition._pca',
                                                                       'PCA')),
                                          DagNodeDetails('PCA: transform', ['array'],
                                                         OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                       RangeComparison(0, 800))),
                                          OptionalCodeInfo(CodeReference(16, 36, 16, 72),
                                                           'PCA(n_components=2, random_state=42)'),
                                          Comparison(FunctionType))
    expected_dag.add_edge(expected_transformer_pca, expected_transform_test_pca, arg_index=0)
    expected_dag.add_edge(expected_concat, expected_transform_test_pca, arg_index=1)
    expected_transform_test_svd = DagNode(10,
                                          BasicCodeLocation("<string-source>", 17),
                                          OperatorContext(OperatorType.TRANSFORMER,
                                                          FunctionInfo('sklearn.decomposition._truncated_svd',
                                                                       'TruncatedSVD')),
                                          DagNodeDetails('Truncated SVD: transform', ['array'],
                                                         OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                       RangeComparison(0, 800))),
                                          OptionalCodeInfo(CodeReference(17, 36, 17, 75),
                                                           'TruncatedSVD(n_iter=1, random_state=42)'),
                                          Comparison(FunctionType))
    expected_dag.add_edge(expected_transformer_svd, expected_transform_test_svd, arg_index=0)
    expected_dag.add_edge(expected_concat, expected_transform_test_svd, arg_index=1)
    expected_transform_test_concat = DagNode(11,
                                             BasicCodeLocation("<string-source>", 16),
                                             OperatorContext(OperatorType.CONCATENATION,
                                                             FunctionInfo('sklearn.pipeline', 'FeatureUnion')),
                                             DagNodeDetails('Feature Union', ['array'],
                                                            OptimizerInfo(RangeComparison(0, 200), (4, 4),
                                                                          RangeComparison(0, 800))),
                                             OptionalCodeInfo(CodeReference(16, 14, 17, 78),
                                                              "FeatureUnion([('pca', PCA(n_components=2, "
                                                              "random_state=42)),\n                            "
                                                              "('svd', TruncatedSVD(n_iter=1, random_state=42))])"),
                                             Comparison(FunctionType))
    expected_dag.add_edge(expected_transform_test_pca, expected_transform_test_concat, arg_index=0)
    expected_dag.add_edge(expected_transform_test_svd, expected_transform_test_concat, arg_index=1)
    compare(list(inspector_result.original_dag)[6], expected_transform_test_concat)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    transformer1_node_fit = list(inspector_result.original_dag.nodes)[1]
    transformer2_node_fit = list(inspector_result.original_dag.nodes)[2]
    concat_node_fit = list(inspector_result.original_dag.nodes)[3]
    num_array = numpy.array([[0., 2., 2., 2.], [0., 0., 1., 2.], [4., 4., 3., 2.], [2., 4., 3., 200.]])
    transformed_data1 = transformer1_node_fit.processing_func(num_array)
    transformed_data2 = transformer2_node_fit.processing_func(num_array)
    concatenated_data = concat_node_fit.processing_func(transformed_data1, transformed_data2)
    expected = numpy.array([[-4.95079323e+01, -9.59880042e-01,  2.07047289e+00,  2.28904433e+00],
                            [-4.95331939e+01, -2.51111219e+00,  2.01455480e+00,  4.68660159e-01],
                            [-4.94691849e+01,  3.47199196e+00,  2.16722945e+00,  6.31931174e+00],
                            [ 1.48510311e+02, -9.99733102e-04,  2.00072463e+02, -9.68596279e-02]])
    assert numpy.allclose(concatenated_data, expected)

    transformer1_node_transform = list(inspector_result.original_dag.nodes)[4]
    transformer2_node_transform = list(inspector_result.original_dag.nodes)[5]
    concat_node_transform = list(inspector_result.original_dag.nodes)[6]
    num_array_transform = numpy.array([[5., 2., 2., 2.], [0., 0., 0., 0.], [4., 4., 3., 5.], [2., 4., 3., 2.]])
    transformed_data1_transform = transformer1_node_transform.processing_func(transformed_data1, num_array_transform)
    transformed_data2_transform = transformer2_node_transform.processing_func(transformed_data2, num_array_transform)
    concatenated_data_transform = concat_node_transform.processing_func(transformed_data1_transform,
                                                                        transformed_data2_transform)
    expected = numpy.array([[-49.49107512,   2.64091978,   2.12152098,   5.05139838],
                            [-51.53810725,  -2.80082658,   0.,           0.        ],
                            [-46.4693934,    3.44119391,   5.16610297,   6.23883856],
                            [-49.47592776,   2.03167203,   2.14681021,   5.21437012]])
    assert numpy.allclose(concatenated_data_transform, expected)
