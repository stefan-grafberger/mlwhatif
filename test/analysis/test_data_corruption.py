"""
Tests whether the Data Corruption analysis works
"""
from inspect import cleandoc

from example_pipelines import HEALTHCARE_PY, COMPAS_PY, ADULT_COMPLEX_PY
from example_pipelines.healthcare import custom_monkeypatching
from mlwhatif import PipelineAnalyzer
from mlwhatif.analysis._data_corruption import DataCorruption, CorruptionType


def test_data_corruption_mini_example_with_transformer_processing_multiple_columns(tmpdir):
    """
    Tests whether the Data Corruption analysis works for a very simple pipeline with a DecisionTree score
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

    def corruption(pandas_df):
        pandas_df['B'] = 0
        return pandas_df

    data_corruption = DataCorruption([('A', CorruptionType.SCALING), ('B', corruption)], also_corrupt_train=True)

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_what_if_analysis(data_corruption) \
        .execute()

    report = analysis_result.analysis_to_result_reports[data_corruption]
    assert report.shape == (7, 4)

    # visualize_dags(analysis_result, tmpdir)


def test_data_corruption_mini_example_with_projection_modify(tmpdir):
    """
    Tests whether the Data Corruption analysis works for a very simple pipeline with a DecisionTree score
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

    def corruption(pandas_df):
        pandas_df['B'] = 0
        return pandas_df

    def custom_index_selection_func(pandas_df):
        return pandas_df['B'] >= 3

    data_corruption = DataCorruption([('A', CorruptionType.SCALING), ('B', corruption)], also_corrupt_train=True,
                                     corruption_percentages=[0.2, custom_index_selection_func])

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_what_if_analysis(data_corruption) \
        .execute()

    report = analysis_result.analysis_to_result_reports[data_corruption]
    assert report.shape == (5, 4)

    # visualize_dags(analysis_result, tmpdir)


def test_data_corruption_mini_example_only_train_test_split(tmpdir):
    """
    Tests whether the Data Corruption analysis works for a very simple pipeline with a DecisionTree score
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

    def corruption(pandas_df):
        pandas_df['B'] = 0
        return pandas_df

    data_corruption = DataCorruption([('A', CorruptionType.SCALING), ('B', corruption)], also_corrupt_train=True)

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_what_if_analysis(data_corruption) \
        .execute()

    report = analysis_result.analysis_to_result_reports[data_corruption]
    assert report.shape == (7, 4)

    # visualize_dags(analysis_result, tmpdir)


def test_data_corruption_mini_example_only_train_test_split_without_optimizer(tmpdir):
    """
    Tests whether the Data Corruption analysis works for a very simple pipeline with a DecisionTree score
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

    def corruption(pandas_df):
        pandas_df['B'] = 0
        return pandas_df

    data_corruption = DataCorruption([('A', CorruptionType.SCALING), ('B', corruption)], also_corrupt_train=True)

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_what_if_analysis(data_corruption) \
        .skip_multi_query_optimization(True) \
        .execute()

    report = analysis_result.analysis_to_result_reports[data_corruption]
    assert report.shape == (7, 4)

    # visualize_dags(analysis_result, tmpdir, skip_combined_dag=True)


def test_data_corruption_mini_example_manual_split(tmpdir):
    """
    Tests whether the Data Corruption analysis works for a very simple pipeline with a DecisionTree score
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

    def corruption(pandas_df):
        pandas_df['B'] = 0
        return pandas_df

    data_corruption = DataCorruption([('A', CorruptionType.SCALING), ('B', corruption)], also_corrupt_train=True)

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_what_if_analysis(data_corruption) \
        .execute()

    report = analysis_result.analysis_to_result_reports[data_corruption]
    assert report.shape == (7, 4)

    # visualize_dags(analysis_result, tmpdir)


def test_data_corruption_healthcare(tmpdir):
    """
    Tests whether the Data Corruption analysis works for a very simple pipeline with a DecisionTree score
    """
    def corruption(pandas_df):
        pandas_df['num_children'] = 0
        return pandas_df

    data_corruption = DataCorruption([('income', CorruptionType.SCALING), ('num_children', corruption)],
                                     corruption_percentages=[0.3, 0.6],
                                     also_corrupt_train=True)

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(HEALTHCARE_PY) \
        .skip_multi_query_optimization(False) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .add_what_if_analysis(data_corruption) \
        .execute()

    report = analysis_result.analysis_to_result_reports[data_corruption]
    assert report.shape == (5, 8)

    # visualize_dags(analysis_result, tmpdir)


def test_data_corruption_compas(tmpdir):
    """
    Tests whether the Data Corruption analysis works for a very simple pipeline with a DecisionTree score
    """

    def corruption(pandas_df):
        pandas_df['is_recid'] = 1
        return pandas_df

    data_corruption = DataCorruption([('age', CorruptionType.SCALING), ('is_recid', corruption)],
                                     also_corrupt_train=False)

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(COMPAS_PY) \
        .add_what_if_analysis(data_corruption) \
        .execute()

    report = analysis_result.analysis_to_result_reports[data_corruption]
    assert report.shape == (7, 3)

    # visualize_dags(analysis_result, tmpdir)


def test_data_corruption_adult_complex():
    """
    Tests whether the Data Corruption analysis works for a very simple pipeline with a DecisionTree score
    """

    def corruption(pandas_df):
        pandas_df['hours-per-week'] = 400
        return pandas_df

    data_corruption = DataCorruption([('education', CorruptionType.CATEGORICAL_SHIFT),
                                      ('workclass', CorruptionType.CATEGORICAL_SHIFT),
                                      ('hours-per-week', corruption)],
                                     corruption_percentages=[0.25, 0.5, 0.75, 1.0],
                                     also_corrupt_train=True)

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(ADULT_COMPLEX_PY) \
        .add_what_if_analysis(data_corruption) \
        .execute()

    report = analysis_result.analysis_to_result_reports[data_corruption]
    assert report.shape == (13, 4)
