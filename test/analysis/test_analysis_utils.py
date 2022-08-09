"""
Tests whether the analysis utils work
"""
import os
from inspect import cleandoc

from mlwhatif import PipelineAnalyzer, OperatorType
from mlwhatif.analysis._analysis_utils import find_nodes_by_type
from mlwhatif.utils import get_project_root

INTERMEDIATE_EXTRACTION_ADD_BEFORE_PATH = os.path.join(str(get_project_root()), "test", "analysis", "debug-dags",
                                                       "test_add_intermediate_extraction_before.png")
INTERMEDIATE_EXTRACTION_ADD_AFTER_PATH = os.path.join(str(get_project_root()), "test", "analysis", "debug-dags",
                                                      "test_add_intermediate_extraction_after.png")


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
    dag = analysis_result.dag
    selections = find_nodes_by_type(dag, OperatorType.SELECTION)
    assert len(selections) == 1
