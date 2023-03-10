"""
Tests whether the Model Variants analysis works
"""
# pylint: disable=unused-argument
from functools import partial
from inspect import cleandoc

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from example_pipelines import HEALTHCARE_PY, COMPAS_PY, ADULT_COMPLEX_PY
from example_pipelines.healthcare import custom_monkeypatching
from mlwhatif import PipelineAnalyzer
from mlwhatif.analysis._model_variants import ModelVariants


def test_model_variants_mini_example_with_transformer_processing_multiple_columns(tmpdir):
    """
    Tests whether the Model Variants analysis works for a very simple pipeline with a DecisionTree score
    """
    test_code = cleandoc("""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler
        from sklearn.tree import DecisionTreeClassifier
        import numpy as np

        df = pd.DataFrame({'A': [0, 0, 0, 0], 'B': [0, 1, 3, 4], 'target': ['no', 'no', 'yes', 'yes']})

        standard_scaler = StandardScaler()
        train = standard_scaler.fit_transform(df[['A', 'B']])
        target = label_binarize(df['target'], classes=['no', 'yes'])

        clf = DecisionTreeClassifier()
        clf = clf.fit(train, target)

        test_df = pd.DataFrame({'A': [0, 0, 0, 0], 'B':  [4, 3, 4, 3], 'target': ['yes', 'yes', 'yes', 'yes']})
        test_data = standard_scaler.transform(test_df[['A', 'B']])
        test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
        test_score = clf.score(test_data, test_labels)
        assert test_score == 1.0
        """)

    model_variants = ModelVariants([('logistic_regression', partial(LogisticRegression)),
                                    ('dummy_classifier', partial(DummyClassifier, strategy='most_frequent',
                                                                 constant=0.))])

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_what_if_analysis(model_variants) \
        .execute()

    report = analysis_result.analysis_to_result_reports[model_variants]
    assert report.shape == (3, 2)

    # visualize_dags(analysis_result, tmpdir)


def test_model_variants_mini_example_with_projection_modify(tmpdir):
    """
    Tests whether the Model Variants analysis works for a very simple pipeline with a DecisionTree score
    """
    test_code = cleandoc("""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler
        from sklearn.tree import DecisionTreeClassifier
        import numpy as np

        df = pd.DataFrame({'A': [0, 0, 0, 0], 'B': [0, 1, 3, 4], 'target': ['no', 'no', 'yes', 'yes']})

        df['A'] = df['A'] / 4
        df['B'] = df['B'] / 4
        train = df[['A', 'B']]
        target = label_binarize(df['target'], classes=['no', 'yes'])

        clf = DecisionTreeClassifier()
        clf = clf.fit(train, target)

        test_df = pd.DataFrame({'A': [0, 0, 0, 0], 'B':  [4, 3, 4, 3], 'target': ['yes', 'yes', 'yes', 'yes']})
        test_df['A'] = test_df['A'] / 4
        test_df['B'] = test_df['B'] / 4
        test_data = test_df[['A', 'B']]
        test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
        test_score = clf.score(test_data, test_labels)
        assert test_score == 1.0
        """)

    model_variants = ModelVariants([('logistic_regression', partial(LogisticRegression)),
                                    ('dummy_classifier', partial(DummyClassifier, strategy='most_frequent',
                                                                 constant=0.))])

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_what_if_analysis(model_variants) \
        .execute()

    report = analysis_result.analysis_to_result_reports[model_variants]
    assert report.shape == (3, 2)

    # visualize_dags(analysis_result, tmpdir)


def test_model_variants_mini_example_only_train_test_split(tmpdir):
    """
    Tests whether the Model Variants analysis works for a very simple pipeline with a DecisionTree score
    """
    test_code = cleandoc("""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler
        from sklearn.tree import DecisionTreeClassifier
        import numpy as np
        from sklearn.model_selection import train_test_split

        pandas_df = pd.DataFrame({'A': [0, 0, 0, 0, 0, 0, 0, 0], 'B': [0, 1, 3, 4, 4, 3, 4, 3], 
            'target': ['no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes']})
        train_data, test_data = train_test_split(pandas_df, random_state=0)

        target = label_binarize(train_data['target'], classes=['no', 'yes'])
        train_data = train_data[['A', 'B']]

        clf = DecisionTreeClassifier()
        clf = clf.fit(train_data, target)

        test_labels = label_binarize(test_data['target'], classes=['no', 'yes'])
        test_data = test_data[['A', 'B']]
        test_score = clf.score(test_data, test_labels)
        assert test_score == 1.0
        """)

    model_variants = ModelVariants([('logistic_regression', partial(LogisticRegression)),
                                    ('dummy_classifier', partial(DummyClassifier, strategy='most_frequent',
                                                                 constant=0.))])

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_what_if_analysis(model_variants) \
        .execute()

    report = analysis_result.analysis_to_result_reports[model_variants]
    assert report.shape == (3, 2)

    # visualize_dags(analysis_result, tmpdir)


def test_model_variants_mini_example_only_train_test_split_without_optimizer(tmpdir):
    """
    Tests whether the Model Variants analysis works for a very simple pipeline with a DecisionTree score
    """
    test_code = cleandoc("""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler
        from sklearn.tree import DecisionTreeClassifier
        import numpy as np
        from sklearn.model_selection import train_test_split

        pandas_df = pd.DataFrame({'A': [0, 0, 0, 0, 0, 0, 0, 0], 'B': [0, 1, 3, 4, 4, 3, 4, 3], 
            'target': ['no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes']})
        train_data, test_data = train_test_split(pandas_df, random_state=0)

        target = label_binarize(train_data['target'], classes=['no', 'yes'])
        train_data = train_data[['A', 'B']]

        clf = DecisionTreeClassifier()
        clf = clf.fit(train_data, target)

        test_labels = label_binarize(test_data['target'], classes=['no', 'yes'])
        test_data = test_data[['A', 'B']]
        test_score = clf.score(test_data, test_labels)
        assert test_score == 1.0
        """)

    model_variants = ModelVariants([('logistic_regression', partial(LogisticRegression)),
                                    ('dummy_classifier', partial(DummyClassifier, strategy='most_frequent',
                                                                 constant=0.))])

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_what_if_analysis(model_variants) \
        .skip_multi_query_optimization(True) \
        .execute()

    report = analysis_result.analysis_to_result_reports[model_variants]
    assert report.shape == (3, 2)

    # visualize_dags(analysis_result, tmpdir, skip_combined_dag=True)


def test_model_variants_mini_example_manual_split(tmpdir):
    """
    Tests whether the Model Variants analysis works for a very simple pipeline with a DecisionTree score
    """
    test_code = cleandoc("""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler
        from sklearn.tree import DecisionTreeClassifier
        import numpy as np
        from sklearn.model_selection import train_test_split

        pandas_df = pd.DataFrame({'A': [0, 0, 0, 0, 0, 0, 0, 0], 'B': [0, 1, 3, 4, 4, 3, 4, 3], 
            'target': ['no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes']})
        train_data = pandas_df[pandas_df['B'] < 4]
        test_data = pandas_df[pandas_df['B'] >= 4]

        target = label_binarize(train_data['target'], classes=['no', 'yes'])
        train_data = train_data[['A', 'B']]

        clf = DecisionTreeClassifier()
        clf = clf.fit(train_data, target)

        test_labels = label_binarize(test_data['target'], classes=['no', 'yes'])
        test_data = test_data[['A', 'B']]
        test_score = clf.score(test_data, test_labels)
        assert test_score == 1.0
        """)

    model_variants = ModelVariants([('logistic_regression', partial(LogisticRegression)),
                                    ('dummy_classifier', partial(DummyClassifier, strategy='most_frequent',
                                                                 constant=0.))])

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_what_if_analysis(model_variants) \
        .execute()

    report = analysis_result.analysis_to_result_reports[model_variants]
    assert report.shape == (3, 2)

    # visualize_dags(analysis_result, tmpdir)


def test_model_variants_healthcare(tmpdir):
    """
    Tests whether the Model Variants analysis works for a very simple pipeline with a DecisionTree score
    """

    model_variants = ModelVariants([('logistic_regression', partial(LogisticRegression)),
                                    ('dummy_classifier', partial(DummyClassifier, strategy='most_frequent',
                                                                 constant=0.))])

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(HEALTHCARE_PY) \
        .skip_multi_query_optimization(False) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .add_what_if_analysis(model_variants) \
        .execute()

    report = analysis_result.analysis_to_result_reports[model_variants]
    assert report.shape == (3, 4)

    # visualize_dags(analysis_result, tmpdir)


def test_model_variants_compas(tmpdir):
    """
    Tests whether the Model Variants analysis works for a very simple pipeline with a DecisionTree score
    """

    model_variants = ModelVariants([('logistic_regression', partial(LogisticRegression)),
                                    ('dummy_classifier', partial(DummyClassifier, strategy='most_frequent',
                                                                 constant=0.))])

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(COMPAS_PY) \
        .add_what_if_analysis(model_variants) \
        .execute()

    report = analysis_result.analysis_to_result_reports[model_variants]
    assert report.shape == (3, 2)

    # visualize_dags(analysis_result, tmpdir)


def test_model_variants_adult_complex():
    """
    Tests whether the Model Variants analysis works for a very simple pipeline with a DecisionTree score
    """

    model_variants = ModelVariants([('logistic_regression', partial(LogisticRegression, solver='saga')),
                                    ('dummy_classifier', partial(DummyClassifier, strategy='most_frequent',
                                                                 constant=0.))])

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(ADULT_COMPLEX_PY) \
        .add_what_if_analysis(model_variants) \
        .execute()

    report = analysis_result.analysis_to_result_reports[model_variants]
    assert report.shape == (3, 2)
