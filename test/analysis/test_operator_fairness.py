"""
Tests whether the Data Corruption analysis works
"""
from inspect import cleandoc

from example_pipelines import HEALTHCARE_PY, ADULT_COMPLEX_PY, COMPAS_PY
from example_pipelines.healthcare import custom_monkeypatching
from mlwhatif import PipelineAnalyzer
from mlwhatif.analysis._operator_fairness import OperatorFairness
from mlwhatif.testing._testing_helper_utils import visualize_dags


def test_operator_fairness_mini_example_with_transformer_processing_multiple_columns(tmpdir):
    """
    Tests whether the Operator Fairness analysis works for a very simple pipeline with a DecisionTree score
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

        test_df = pd.DataFrame({'A': [0, 0, 0, 0], 'B':  [4, 3, 4, 3], 
            'sensitive': ["cat_a", "cat_b", "cat_a", "cat_b"], 'target': ['yes', 'yes', 'yes', 'yes']})
        test_data = standard_scaler.transform(test_df[['A', 'B']])
        test_labels = label_binarize(test_df['target'], classes=['no', 'yes'])
        test_score = clf.score(test_data, test_labels)
        assert test_score == 1.0
        """)

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_what_if_analysis(OperatorFairness(True, True)) \
        .execute()

    report = analysis_result.analysis_to_result_reports[OperatorFairness(True, True)]
    assert report.shape == (2, 5)

    visualize_dags(analysis_result, tmpdir)


def test_operator_fairness_healthcare(tmpdir):
    """
    Tests whether the Operator Fairness analysis works for a very simple pipeline with a DecisionTree score
    """
    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(HEALTHCARE_PY) \
        .skip_multi_query_optimization(False) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .add_what_if_analysis(OperatorFairness(True, True)) \
        .execute()

    report = analysis_result.analysis_to_result_reports[OperatorFairness(True, True)]
    assert report.shape == (5, 7)

    visualize_dags(analysis_result, tmpdir)


def test_operator_fairness_compas(tmpdir):
    """
    Tests whether the Operator Fairness analysis works for a very simple pipeline with a DecisionTree score
    """

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(COMPAS_PY) \
        .add_what_if_analysis(OperatorFairness(True, True)) \
        .execute()

    report = analysis_result.analysis_to_result_reports[OperatorFairness(True, True)]
    assert report.shape == (8, 5)

    visualize_dags(analysis_result, tmpdir)


def test_operator_fairness_restrict_to_linenos(tmpdir):
    """
    Tests whether the Operator Fairness analysis works for a very simple pipeline with a DecisionTree score
    """

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(COMPAS_PY) \
        .add_what_if_analysis(OperatorFairness(True, True, [41, 29])) \
        .execute()

    report = analysis_result.analysis_to_result_reports[OperatorFairness(True, True)]
    assert report.shape == (3, 5)

    visualize_dags(analysis_result, tmpdir)


def test_operator_fairness_test_transformers(tmpdir):
    """
    Tests whether the Operator Fairness analysis works for a very simple pipeline with a DecisionTree score
    """

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(COMPAS_PY) \
        .add_what_if_analysis(OperatorFairness(True, False)) \
        .execute()

    report = analysis_result.analysis_to_result_reports[OperatorFairness(True, False)]
    assert report.shape == (4, 5)

    visualize_dags(analysis_result, tmpdir)


def test_operator_fairness_test_selections(tmpdir):
    """
    Tests whether the Operator Fairness analysis works for a very simple pipeline with a DecisionTree score
    """

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(COMPAS_PY) \
        .add_what_if_analysis(OperatorFairness(False, True)) \
        .execute()

    report = analysis_result.analysis_to_result_reports[OperatorFairness(False, True)]
    assert report.shape == (5, 5)

    visualize_dags(analysis_result, tmpdir)


def test_operator_fairness_adult_complex(tmpdir):
    """
    Tests whether the Operator Fairness analysis works for a very simple pipeline with a DecisionTree score
    """
    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(ADULT_COMPLEX_PY) \
        .add_what_if_analysis(OperatorFairness(True, True)) \
        .execute()

    report = analysis_result.analysis_to_result_reports[OperatorFairness(True, True)]
    assert report.shape == (3, 5)

    visualize_dags(analysis_result, tmpdir)
