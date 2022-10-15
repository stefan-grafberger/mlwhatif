"""
Tests whether the patch creation utils work
"""
import os
from inspect import cleandoc

import pandas

from mlmq import PipelineAnalyzer, OperatorType
from mlmq.analysis._analysis_utils import find_nodes_by_type
from mlmq.analysis._patch_creation import get_intermediate_extraction_patch_after_node
from mlmq.execution._dag_executor import DagExecutor

from mlmq.execution._pipeline_executor import singleton
from mlmq.visualisation import save_fig_to_path


def test_add_intermediate_extraction_after_node_intermediate_df(tmpdir):
    """
    Tests whether the Data Corruption analysis works for a very simple pipeline with a DecisionTree score
    """
    test_code = cleandoc("""
        import pandas as pd

        df = pd.DataFrame([0, 2, 4, 5, None], columns=['A'])
        assert len(df) == 5
        df = df.dropna()
        assert len(df) == 4
        df_a = df['A']
        """)

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute()
    dag = analysis_result.original_dag
    intermediate_pdf_result_value = find_nodes_by_type(dag, OperatorType.SELECTION)[0]
    label = "util-test"

    save_fig_to_path(dag, os.path.join(str(tmpdir), "test_add_intermediate_extraction_before.png"))
    patch = get_intermediate_extraction_patch_after_node(singleton, None, intermediate_pdf_result_value, label)
    patch.apply(dag, singleton)
    save_fig_to_path(dag, os.path.join(str(tmpdir), "test_add_intermediate_extraction_after.png"))
    DagExecutor(singleton).execute(dag)
    extracted_value = singleton.labels_to_extracted_plan_results[label]

    df_expected = pandas.DataFrame([0, 2, 4, 5.], columns=['A'])
    pandas.testing.assert_frame_equal(extracted_value, df_expected)


def test_add_intermediate_extraction_after_node_final_score(tmpdir):
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

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute()
    dag = analysis_result.original_dag
    final_result_value = find_nodes_by_type(dag, OperatorType.SCORE)[0]
    label = "util-test"
    save_fig_to_path(dag, os.path.join(str(tmpdir), "test_add_intermediate_extraction_before.png"))
    patch = get_intermediate_extraction_patch_after_node(singleton, None, final_result_value, label)
    patch.apply(dag, singleton)
    save_fig_to_path(dag, os.path.join(str(tmpdir), "test_add_intermediate_extraction_after.png"))
    DagExecutor(singleton).execute(dag)
    extracted_value = singleton.labels_to_extracted_plan_results[label]
    assert isinstance(extracted_value, float) and 0. <= extracted_value <= 1.
