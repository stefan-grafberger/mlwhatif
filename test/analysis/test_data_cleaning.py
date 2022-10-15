"""
Tests whether the Data Cleaning analysis works
"""
from inspect import cleandoc

from example_pipelines import HEALTHCARE_PY, ADULT_COMPLEX_PY, COMPAS_PY
from example_pipelines.healthcare import custom_monkeypatching
from mlmq import PipelineAnalyzer
from mlmq.analysis._data_cleaning import DataCleaning, ErrorType
from mlmq.testing._testing_helper_utils import visualize_dags


def test_data_cleaning_mini_example_with_transformer_processing_multiple_columns(tmpdir):
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

    data_cleaning = DataCleaning({'B': ErrorType.DUPLICATES})
    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_what_if_analysis(data_cleaning) \
        .execute()

    report = analysis_result.analysis_to_result_reports[data_cleaning]
    assert report.shape == (2, 4)

    visualize_dags(analysis_result, tmpdir)


def test_data_cleaning_healthcare(tmpdir):
    """
    Tests whether the Data Corruption analysis works for a very simple pipeline with a DecisionTree score
    """
    data_cleaning = DataCleaning({'smoker': ErrorType.CAT_MISSING_VALUES,
                                  'num_children': ErrorType.NUM_MISSING_VALUES,
                                  'income': ErrorType.OUTLIERS,
                                  'ssn': ErrorType.DUPLICATES,
                                  None: ErrorType.MISLABEL})

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(HEALTHCARE_PY) \
        .skip_multi_query_optimization(False) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .add_what_if_analysis(data_cleaning) \
        .execute()

    report = analysis_result.analysis_to_result_reports[data_cleaning]
    assert report.shape == (20, 6)

    visualize_dags(analysis_result, tmpdir)


def test_data_cleaning_compas(tmpdir):
    """
    Tests whether the Data Corruption analysis works for a very simple pipeline with a DecisionTree score
    """

    data_cleaning = DataCleaning({'is_recid': ErrorType.CAT_MISSING_VALUES,
                                  'age': ErrorType.OUTLIERS,
                                  None: ErrorType.MISLABEL})

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(COMPAS_PY) \
        .add_what_if_analysis(data_cleaning) \
        .execute()

    report = analysis_result.analysis_to_result_reports[data_cleaning]
    assert report.shape == (15, 4)

    visualize_dags(analysis_result, tmpdir)


def test_data_cleaning_adult_complex(tmpdir):
    """
    Tests whether the Data Cleaning analysis works for a very simple pipeline with a DecisionTree score
    """

    data_cleaning = DataCleaning({'education': ErrorType.CAT_MISSING_VALUES,
                                  'age': ErrorType.NUM_MISSING_VALUES,
                                  'hours-per-week': ErrorType.OUTLIERS,
                                  None: ErrorType.MISLABEL})

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(ADULT_COMPLEX_PY) \
        .add_what_if_analysis(data_cleaning) \
        .execute()

    report = analysis_result.analysis_to_result_reports[data_cleaning]
    assert report.shape == (19, 4)

    visualize_dags(analysis_result, tmpdir)
