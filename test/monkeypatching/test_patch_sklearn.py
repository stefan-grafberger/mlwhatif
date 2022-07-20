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
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import label_binarize, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier  # pylint: disable=no-name-in-module
from testfixtures import compare, Comparison, RangeComparison

from mlwhatif import OperatorType, OperatorContext, FunctionInfo
from mlwhatif.instrumentation import _pipeline_executor
from mlwhatif.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo, OptimizerInfo
from mlwhatif.monkeypatching._patch_sklearn import TrainTestSplitResult


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
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    label_binarize_node = list(inspector_result.dag.nodes)[1]
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
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[5])
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[4])

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

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    split_node = list(inspector_result.dag.nodes)[1]
    train_node = list(inspector_result.dag.nodes)[2]
    test_node = list(inspector_result.dag.nodes)[3]
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
                                                                RangeComparison(0, 800))),
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
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.dag.nodes)[1]
    transform_node = list(inspector_result.dag.nodes)[3]
    pandas_df = pandas.DataFrame({'A': [5, 1, 100, 2]})
    fit_transformed_result = fit_transform_node.processing_func(pandas_df)
    expected_fit_transform_data = numpy.array([[-0.52166986], [-0.61651893], [1.73099545], [-0.59280666]])
    assert numpy.allclose(fit_transformed_result, expected_fit_transform_data)

    test_df = pandas.DataFrame({'A': [50, 2, 10, 1]})
    encoded_data = transform_node.processing_func(fit_transformed_result, test_df)
    expected = numpy.array([[0.54538213], [-0.59280666], [-0.40310853], [-0.61651893]])
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
                                                                RangeComparison(0, 800))),
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
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(9, 23, 9, 65),
                                                        'FunctionTransformer(lambda x: safe_log(x))'),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_transformer, expected_transformer_two, arg_index=0)
    expected_dag.add_edge(expected_data_source_two, expected_transformer_two, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.dag.nodes)[1]
    transform_node = list(inspector_result.dag.nodes)[3]
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
                                                                RangeComparison(0, 800))),
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
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.dag.nodes)[1]
    transform_node = list(inspector_result.dag.nodes)[3]
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
                                                                RangeComparison(0, 800))),
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
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.dag.nodes)[1]
    transform_node = list(inspector_result.dag.nodes)[3]
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
                                                                RangeComparison(0, 800))),
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
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.dag.nodes)[1]
    transform_node = list(inspector_result.dag.nodes)[3]
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
                                                                RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(7, 18, 7, 33), 'OneHotEncoder()'),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_transformer, arg_index=0)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.dag.nodes)[1]
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
    inspector_result.dag.remove_node(list(inspector_result.dag)[0])
    inspector_result.dag.remove_node(list(inspector_result.dag)[2])

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
                                                                RangeComparison(0, 800))),
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
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.dag.nodes)[1]
    transform_node = list(inspector_result.dag.nodes)[3]
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
                                                                    RangeComparison(0, 800))),
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
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    project_node = list(inspector_result.dag.nodes)[1]
    transformer_node = list(inspector_result.dag.nodes)[2]
    concat_node = list(inspector_result.dag.nodes)[3]
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
                                                               RangeComparison(0, 800))),
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

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    project_node = list(inspector_result.dag.nodes)[0]
    transformer_node = list(inspector_result.dag.nodes)[1]
    concat_node = list(inspector_result.dag.nodes)[2]
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
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(9, 16, 9, 32), 'StandardScaler()'),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_projection_1, expected_standard_scaler, arg_index=0)
    expected_one_hot = DagNode(4,
                               BasicCodeLocation("<string-source>", 10),
                               OperatorContext(OperatorType.TRANSFORMER,
                                               FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')),
                               DagNodeDetails('One-Hot Encoder: fit_transform', ['array'],
                                              OptimizerInfo(RangeComparison(0, 200), (4, 3),
                                                            RangeComparison(0, 800))),
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
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    project1_node = list(inspector_result.dag.nodes)[1]
    project2_node = list(inspector_result.dag.nodes)[3]
    transformer1_node = list(inspector_result.dag.nodes)[2]
    transformer2_node = list(inspector_result.dag.nodes)[4]
    concat_node = list(inspector_result.dag.nodes)[5]
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
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(9, 16, 9, 32), 'StandardScaler()'),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_projection_1, expected_standard_scaler, arg_index=0)
    expected_one_hot = DagNode(4,
                               BasicCodeLocation("<string-source>", 10),
                               OperatorContext(OperatorType.TRANSFORMER,
                                               FunctionInfo('sklearn.preprocessing._encoders', 'OneHotEncoder')),
                               DagNodeDetails('One-Hot Encoder: fit_transform', ['array'],
                                              OptimizerInfo(RangeComparison(0, 200), (4, 3),
                                                            RangeComparison(0, 800))),
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
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    project1_node = list(inspector_result.dag.nodes)[1]
    project2_node = list(inspector_result.dag.nodes)[3]
    transformer1_node = list(inspector_result.dag.nodes)[2]
    transformer2_node = list(inspector_result.dag.nodes)[4]
    concat_node = list(inspector_result.dag.nodes)[5]
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
    fit_func_transformer1 = list(inspector_result.dag.nodes)[2].processing_func
    fit_func_transformer2 = list(inspector_result.dag.nodes)[4].processing_func
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
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    project1_node = list(inspector_result.dag.nodes)[1]
    project2_node = list(inspector_result.dag.nodes)[3]
    transformer1_node = list(inspector_result.dag.nodes)[2]
    transformer2_node = list(inspector_result.dag.nodes)[4]
    concat_node = list(inspector_result.dag.nodes)[5]
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
                
                test_predict = clf.predict([[0., 0.], [0.6, 0.6]])
                expected = np.array([0., 1.])
                assert np.allclose(test_predict, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

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
                                                                    RangeComparison(0, 800))),
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
                                                                                       RangeComparison(0, 800))),
                                     OptionalCodeInfo(CodeReference(11, 6, 11, 30), 'DecisionTreeClassifier()'),
                                     Comparison(FunctionType))
    expected_dag.add_edge(expected_train_data, expected_decision_tree, arg_index=0)
    expected_dag.add_edge(expected_train_labels, expected_decision_tree, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.dag.nodes)[7]
    train_data_node = list(inspector_result.dag.nodes)[5]
    train_label_node = list(inspector_result.dag.nodes)[6]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_data = train_data_node.processing_func(train_df[['C', 'D']])
    train_labels = label_binarize(train_df['target'], classes=['no', 'yes'])
    train_labels = train_label_node.processing_func(train_labels)
    fitted_estimator = fit_node.processing_func(train_data, train_labels)
    assert isinstance(fitted_estimator, DecisionTreeClassifier)

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
    filter_dag_for_nodes_with_ids(inspector_result, {7, 10, 11, 12, 13, 14}, 15)

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
                                                                                    RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 30),
                                                   'DecisionTreeClassifier()'),
                                  Comparison(FunctionType))
    expected_score = DagNode(14,
                             BasicCodeLocation("<string-source>", 16),
                             OperatorContext(OperatorType.SCORE,
                                             FunctionInfo('sklearn.tree._classes.DecisionTreeClassifier', 'score')),
                             DagNodeDetails('Decision Tree', [], OptimizerInfo(RangeComparison(0, 1000), (1, 1),
                                                                               RangeComparison(0, 800))),
                             OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                              "clf.score(test_df[['A', 'B']], test_labels)"),
                             Comparison(FunctionType))
    expected_dag.add_edge(expected_classifier, expected_score, arg_index=0)
    expected_dag.add_edge(expected_test_data, expected_score, arg_index=1)
    expected_dag.add_edge(expected_test_labels, expected_score, arg_index=2)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.dag.nodes)[0]
    score_node = list(inspector_result.dag.nodes)[5]
    test_data_node = list(inspector_result.dag.nodes)[3]
    test_label_node = list(inspector_result.dag.nodes)[4]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_labels = label_binarize(train_df['target'], classes=['no', 'yes'])
    fitted_estimator = fit_node.processing_func(train_df[['C', 'D']], train_labels)
    assert isinstance(fitted_estimator, DecisionTreeClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_labels = test_label_node.processing_func(test_labels)
    test_score = score_node.processing_func(fitted_estimator, test_data, test_labels)
    assert test_score == 0.5


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

                test_predict = clf.predict([[0., 0.], [0.6, 0.6]])
                expected = np.array([0., 1.])
                assert np.allclose(test_predict, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    filter_dag_for_nodes_with_ids(inspector_result, {2, 5, 4, 6, 7}, 8)

    expected_dag = networkx.DiGraph()
    expected_standard_scaler = DagNode(2,
                                       BasicCodeLocation("<string-source>", 8),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('sklearn.preprocessing._data', 'StandardScaler')),
                                       DagNodeDetails('Standard Scaler: fit_transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 200), (4, 2),
                                                                    RangeComparison(0, 800))),
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
                                                                                     RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 48),
                                                   "SGDClassifier(loss='log', random_state=42)"),
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_train_data, expected_classifier, arg_index=0)
    expected_dag.add_edge(expected_train_labels, expected_classifier, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.dag.nodes)[4]
    train_data_node = list(inspector_result.dag.nodes)[2]
    train_label_node = list(inspector_result.dag.nodes)[3]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_data = train_data_node.processing_func(train_df[['C', 'D']])
    train_labels = label_binarize(train_df['target'], classes=['no', 'yes'])
    train_labels = train_label_node.processing_func(train_labels)
    fitted_estimator = fit_node.processing_func(train_data, train_labels)
    assert isinstance(fitted_estimator, SGDClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_score = fitted_estimator.score(test_df[['C', 'D']], test_labels)
    assert test_score == 0.5


def filter_dag_for_nodes_with_ids(inspector_result, node_ids, total_expected_node_num):
    """
    Filter for DAG Nodes relevant for this test
    """
    assert len(inspector_result.dag.nodes) == total_expected_node_num
    dag_nodes_irrelevant__for_test = [dag_node for dag_node in list(inspector_result.dag.nodes)
                                      if dag_node.node_id not in node_ids]
    inspector_result.dag.remove_nodes_from(dag_nodes_irrelevant__for_test)
    assert len(inspector_result.dag.nodes) == len(node_ids)


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
    filter_dag_for_nodes_with_ids(inspector_result, {7, 10, 11, 12, 13, 14}, 15)

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
                                                                                     RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 48),
                                                   "SGDClassifier(loss='log', random_state=42)"),
                                  Comparison(FunctionType))
    expected_score = DagNode(14,
                             BasicCodeLocation("<string-source>", 16),
                             OperatorContext(OperatorType.SCORE,
                                             FunctionInfo('sklearn.linear_model._stochastic_gradient.SGDClassifier',
                                                          'score')),
                             DagNodeDetails('SGD Classifier', [], OptimizerInfo(RangeComparison(0, 1000), (1, 1),
                                                                                RangeComparison(0, 800))),
                             OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                              "clf.score(test_df[['A', 'B']], test_labels)"),
                             Comparison(FunctionType))
    expected_dag.add_edge(expected_classifier, expected_score, arg_index=0)
    expected_dag.add_edge(expected_test_data, expected_score, arg_index=1)
    expected_dag.add_edge(expected_test_labels, expected_score, arg_index=2)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.dag.nodes)[0]
    score_node = list(inspector_result.dag.nodes)[5]
    test_data_node = list(inspector_result.dag.nodes)[3]
    test_label_node = list(inspector_result.dag.nodes)[4]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_labels = label_binarize(train_df['target'], classes=['no', 'yes'])
    fitted_estimator = fit_node.processing_func(train_df[['C', 'D']], train_labels)
    assert isinstance(fitted_estimator, SGDClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_labels = test_label_node.processing_func(test_labels)
    test_score = score_node.processing_func(fitted_estimator, test_data, test_labels)
    assert test_score == 0.5


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

                test_predict = clf.predict([[0., 0.], [0.6, 0.6]])
                expected = np.array([0., 1.])
                assert np.allclose(test_predict, expected)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

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
                                                                    RangeComparison(0, 800))),
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
                                                              RangeComparison(0, 800))),
                                 OptionalCodeInfo(CodeReference(11, 6, 11, 26), 'LogisticRegression()'),
                                 Comparison(FunctionType))
    expected_dag.add_edge(expected_train_data, expected_estimator, arg_index=0)
    expected_dag.add_edge(expected_train_labels, expected_estimator, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.dag.nodes)[7]
    train_data_node = list(inspector_result.dag.nodes)[5]
    train_label_node = list(inspector_result.dag.nodes)[6]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_data = train_data_node.processing_func(train_df[['C', 'D']])
    train_labels = label_binarize(train_df['target'], classes=['no', 'yes'])
    train_labels = train_label_node.processing_func(train_labels)
    fitted_estimator = fit_node.processing_func(train_data, train_labels)
    assert isinstance(fitted_estimator, LogisticRegression)

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
    filter_dag_for_nodes_with_ids(inspector_result, {7, 10, 11, 12, 13, 14}, 15)

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
                                                               RangeComparison(0, 800))),
                                  OptionalCodeInfo(CodeReference(11, 6, 11, 26),
                                                   'LogisticRegression()'),
                                  Comparison(FunctionType))
    expected_score = DagNode(14,
                             BasicCodeLocation("<string-source>", 16),
                             OperatorContext(OperatorType.SCORE,
                                             FunctionInfo('sklearn.linear_model._logistic.LogisticRegression',
                                                          'score')),
                             DagNodeDetails('Logistic Regression', [], OptimizerInfo(RangeComparison(0, 1000), (1, 1),
                                                                                     RangeComparison(0, 800))),
                             OptionalCodeInfo(CodeReference(16, 13, 16, 56),
                                              "clf.score(test_df[['A', 'B']], test_labels)"),
                             Comparison(FunctionType))
    expected_dag.add_edge(expected_classifier, expected_score, arg_index=0)
    expected_dag.add_edge(expected_test_data, expected_score, arg_index=1)
    expected_dag.add_edge(expected_test_labels, expected_score, arg_index=2)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.dag.nodes)[0]
    score_node = list(inspector_result.dag.nodes)[5]
    test_data_node = list(inspector_result.dag.nodes)[3]
    test_label_node = list(inspector_result.dag.nodes)[4]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_labels = label_binarize(train_df['target'], classes=['no', 'yes'])
    fitted_estimator = fit_node.processing_func(train_df[['C', 'D']], train_labels)
    assert isinstance(fitted_estimator, LogisticRegression)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
    test_labels = test_label_node.processing_func(test_labels)
    test_score = score_node.processing_func(fitted_estimator, test_data, test_labels)
    assert test_score == 0.5


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

                test_predict = clf.predict([[0., 0.], [0.6, 0.6]])
                assert test_predict.shape == (2,)
                """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

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
                                                                    RangeComparison(0, 800))),
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
                                                                 RangeComparison(0, 800))),
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
                                  Comparison(FunctionType))
    expected_dag.add_edge(expected_train_data, expected_classifier, arg_index=0)
    expected_dag.add_edge(expected_train_labels, expected_classifier, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.dag.nodes)[7]
    train_data_node = list(inspector_result.dag.nodes)[5]
    train_label_node = list(inspector_result.dag.nodes)[6]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_data = train_data_node.processing_func(train_df[['C', 'D']])
    train_labels = OneHotEncoder(sparse=False).fit_transform(train_df[['target']])
    train_labels = train_label_node.processing_func(train_labels)
    fitted_estimator = fit_node.processing_func(train_data, train_labels)
    assert isinstance(fitted_estimator, KerasClassifier)

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
    filter_dag_for_nodes_with_ids(inspector_result, {7, 10, 11, 12, 13, 14}, 15)

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
                                                                 RangeComparison(0, 800))),
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
                                  Comparison(FunctionType))
    expected_score = DagNode(14,
                             BasicCodeLocation("<string-source>", 30),
                             OperatorContext(OperatorType.SCORE,
                                             FunctionInfo('tensorflow.python.keras.wrappers.scikit_learn.'
                                                          'KerasClassifier', 'score')),
                             DagNodeDetails('Neural Network', [], OptimizerInfo(RangeComparison(0, 1000), (1, 1),
                                                                                RangeComparison(0, 800))),
                             OptionalCodeInfo(CodeReference(30, 13, 30, 56),
                                              "clf.score(test_df[['A', 'B']], test_labels)"),
                             Comparison(FunctionType))
    expected_dag.add_edge(expected_classifier, expected_score, arg_index=0)
    expected_dag.add_edge(expected_test_data, expected_score, arg_index=1)
    expected_dag.add_edge(expected_test_labels, expected_score, arg_index=2)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))

    fit_node = list(inspector_result.dag.nodes)[0]
    score_node = list(inspector_result.dag.nodes)[5]
    test_data_node = list(inspector_result.dag.nodes)[3]
    test_label_node = list(inspector_result.dag.nodes)[4]
    train_df = pandas.DataFrame({'C': [0, 1, 2, 3], 'D': [0, 1, 2, 3], 'target': ['no', 'no', 'yes', 'yes']})
    train_labels = OneHotEncoder(sparse=False).fit_transform(train_df[['target']])
    fitted_estimator = fit_node.processing_func(train_df[['C', 'D']], train_labels)
    assert isinstance(fitted_estimator, KerasClassifier)

    test_df = pandas.DataFrame({'C': [0., 0.6], 'D': [0., 0.6], 'target': ['no', 'yes']})
    test_data = test_data_node.processing_func(test_df[['C', 'D']])
    test_labels = OneHotEncoder(sparse=False).fit_transform(test_df[['target']])
    test_labels = test_label_node.processing_func(test_labels)
    test_score = score_node.processing_func(fitted_estimator, test_data, test_labels)
    assert test_score >= 0.5
