"""
Tests whether the DAG execution works for extracted DAGs
"""
from inspect import cleandoc

import pandas as pd

from example_pipelines import HEALTHCARE_PY
from example_pipelines.healthcare import custom_monkeypatching as healthcare_patching
from mlwhatif.execution._dag_executor import DagExecutor
from mlwhatif.instrumentation import _pipeline_executor


def test_simple_dag_execution():
    """
    Tests whether the execution works for a very simple example
    """
    test_code = cleandoc("""
        import pandas as pd

        df = pd.DataFrame([0, 2, 4, 5, None], columns=['A'])
        assert len(df) == 5
        df = df.dropna()
        assert len(df) == 4
        """)
    extracted_dag = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True).dag
    final_pipeline_result = DagExecutor().execute(extracted_dag)

    df_expected = pd.DataFrame([0, 2, 4, 5.], columns=['A'])
    pd.testing.assert_frame_equal(final_pipeline_result, df_expected)


def test_healthcare_dag_execution():
    """
    Tests whether the execution works for a very simple example
    """
    extracted_dag = _pipeline_executor.singleton.run(python_path=HEALTHCARE_PY, track_code_references=True,
                                                     custom_monkey_patching=[healthcare_patching]).dag
    final_pipeline_result = DagExecutor().execute(extracted_dag)

    df_expected = pd.DataFrame([0, 2, 4, 5.], columns=['A'])
    pd.testing.assert_frame_equal(final_pipeline_result, df_expected)
