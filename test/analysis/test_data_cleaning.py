"""
Tests whether the Data Cleaning analysis works
"""
import os
from inspect import cleandoc

from mlwhatif import PipelineAnalyzer
from mlwhatif.analysis._data_cleaning import DataCleaning, ErrorType
from mlwhatif.utils import get_project_root

INTERMEDIATE_EXTRACTION_ORIG_PATH = os.path.join(str(get_project_root()), "test", "analysis", "debug-dags",
                                                 "data-cleaning-orig")
INTERMEDIATE_EXTRACTION_GENERATED_PATH = os.path.join(str(get_project_root()), "test", "analysis", "debug-dags",
                                                      "data-cleaning-what-if")
INTERMEDIATE_EXTRACTION_OPTIMISED_PATH = os.path.join(str(get_project_root()), "test", "analysis", "debug-dags",
                                                      "data-cleaning-what-if-optimised")


def test_data_cleaning_mini_example_with_transformer_processing_multiple_columns():
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

    data_cleaning = DataCleaning({'A': ErrorType.OUTLIERS})
    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_what_if_analysis(data_cleaning) \
        .save_original_dag_to_path(INTERMEDIATE_EXTRACTION_ORIG_PATH) \
        .save_what_if_dags_to_path(INTERMEDIATE_EXTRACTION_GENERATED_PATH) \
        .save_optimised_what_if_dags_to_path(INTERMEDIATE_EXTRACTION_OPTIMISED_PATH) \
        .execute()

    report = analysis_result.analysis_to_result_reports[data_cleaning]
    assert report.shape == (1, 4)
