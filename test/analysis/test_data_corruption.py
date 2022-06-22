"""
Tests whether the Data Corruption analysis works
"""
from inspect import cleandoc

from mlwhatif import PipelineAnalyzer
from mlwhatif.analysis._data_corruption import DataCorruption


def test_data_corruption_score():
    """
    Tests whether the Data Corruption analysis works for a very simple pipeline with a DecisionTree score
    """
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

    data_corruption = DataCorruption({'test': lambda pandas_df: pandas_df.dropna()})
    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_what_if_analysis(data_corruption) \
        .execute()
    report = analysis_result.analysis_to_result_reports[data_corruption]
    assert report == "TODO"
