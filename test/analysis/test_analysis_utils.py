"""
Tests whether the analysis utils work
"""
from inspect import cleandoc

from mlmq import PipelineAnalyzer, OperatorType
from mlmq.analysis._analysis_utils import find_nodes_by_type


def test_find_nodes_by_type():
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
    selections = find_nodes_by_type(dag, OperatorType.SELECTION)
    assert len(selections) == 1
