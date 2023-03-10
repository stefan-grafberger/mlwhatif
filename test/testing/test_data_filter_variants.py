"""
Tests whether the Data Filter Variants analysis works
"""
# pylint: disable=unused-argument
from inspect import cleandoc

from example_pipelines import HEALTHCARE_PY, COMPAS_PY, ADULT_COMPLEX_PY
from example_pipelines.healthcare import custom_monkeypatching
from mlwhatif import PipelineAnalyzer
from mlwhatif.testing._data_filter_variants import DataFilterVariants


def test_data_filter_variants_mini_example_with_transformer_processing_multiple_columns(tmpdir):
    """
    Tests whether the Data Filter Variants analysis works for a very simple pipeline with a DecisionTree score
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

    filter_variants = DataFilterVariants({'filter_0': ('B', lambda df: df[df['B'] >= 3]),
                                          'filter_1': ('B', lambda df: df[df['B'] >= 4])})

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_what_if_analysis(filter_variants) \
        .execute()

    report = analysis_result.analysis_to_result_reports[filter_variants]
    assert report.shape == (3, 3)

    # visualize_dags(analysis_result, tmpdir)


def test_data_filter_variants_healthcare(tmpdir):
    """
    Tests whether the Data Filter Variants analysis works for a very simple pipeline with a DecisionTree score
    """

    filter_variants = DataFilterVariants({'filter_0': ('num_children', lambda df: df[df['num_children'] <= 2]),
                                          'filter_1': ('num_children', lambda df: df[df['num_children'] <= 3])},
                                         True)

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(HEALTHCARE_PY) \
        .skip_multi_query_optimization(False) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .add_what_if_analysis(filter_variants) \
        .execute()

    report = analysis_result.analysis_to_result_reports[filter_variants]
    assert report.shape == (4, 5)

    # visualize_dags(analysis_result, tmpdir)


def test_data_filter_variants_compas(tmpdir):
    """
    Tests whether the Data Filter Variants analysis works for a very simple pipeline with a DecisionTree score
    """

    filter_variants = DataFilterVariants({'filter_0': ('age', lambda df: df[df['age'] >= 30]),
                                          'filter_1': ('age', lambda df: df[df['age'] >= 40])})

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(COMPAS_PY) \
        .add_what_if_analysis(filter_variants) \
        .execute()

    report = analysis_result.analysis_to_result_reports[filter_variants]
    assert report.shape == (3, 3)

    # visualize_dags(analysis_result, tmpdir)


def test_data_filter_variants_adult_complex():
    """
    Tests whether the Data Filter Variants analysis works for a very simple pipeline with a DecisionTree score
    """

    filter_variants = DataFilterVariants({'filter_0': ('hours-per-week', lambda df: df[df['hours-per-week'] >= 30]),
                                          'filter_1': ('hours-per-week', lambda df: df[df['hours-per-week'] >= 40])})

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(ADULT_COMPLEX_PY) \
        .add_what_if_analysis(filter_variants) \
        .execute()

    report = analysis_result.analysis_to_result_reports[filter_variants]
    assert report.shape == (3, 3)
