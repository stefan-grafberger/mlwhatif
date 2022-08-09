"""
Tests whether the DAG execution works for extracted DAGs
"""
from inspect import cleandoc

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from example_pipelines import HEALTHCARE_PY, COMPAS_PY, ADULT_SIMPLE_PY, ADULT_COMPLEX_PY
from example_pipelines.healthcare import custom_monkeypatching as healthcare_patching
from mlwhatif.analysis._patch_creation import get_intermediate_extraction_patch_after_node
from mlwhatif.execution._dag_executor import DagExecutor
from mlwhatif.instrumentation import _pipeline_executor
from mlwhatif.instrumentation._pipeline_executor import singleton


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
    extracted_final_pipeline_result = execute_dag_and_get_final_pipeline_result(extracted_dag)
    df_expected = pd.DataFrame([0, 2, 4, 5.], columns=['A'])
    pd.testing.assert_frame_equal(extracted_final_pipeline_result, df_expected)


def test_healthcare_dag_execution():
    """
    Tests whether the execution works for the healthcare pipeline
    """
    extracted_dag = _pipeline_executor.singleton.run(python_path=HEALTHCARE_PY, track_code_references=True,
                                                     custom_monkey_patching=[healthcare_patching]).dag
    extracted_final_pipeline_result = execute_dag_and_get_final_pipeline_result(extracted_dag)
    assert isinstance(extracted_final_pipeline_result, float) and 0. <= extracted_final_pipeline_result <= 1.


def test_compas_dag_execution():
    """
    Tests whether the execution works for the compas pipeline
    """
    extracted_dag = _pipeline_executor.singleton.run(python_path=COMPAS_PY, track_code_references=True).dag
    extracted_final_pipeline_result = execute_dag_and_get_final_pipeline_result(extracted_dag)
    assert isinstance(extracted_final_pipeline_result, float) and 0. <= extracted_final_pipeline_result <= 1.


def test_adult_simple_dag_execution():
    """
    Tests whether the execution works for the adult_simple pipeline
    """
    extracted_dag = _pipeline_executor.singleton.run(python_path=ADULT_SIMPLE_PY, track_code_references=True).dag
    extracted_final_pipeline_result = execute_dag_and_get_final_pipeline_result(extracted_dag)
    assert isinstance(extracted_final_pipeline_result, DecisionTreeClassifier)


def test_adult_complex_dag_execution():
    """
    Tests whether the execution works for the adult_complex pipeline
    """
    extracted_dag = _pipeline_executor.singleton.run(python_path=ADULT_COMPLEX_PY, track_code_references=True).dag
    extracted_final_pipeline_result = execute_dag_and_get_final_pipeline_result(extracted_dag)
    assert isinstance(extracted_final_pipeline_result, float) and 0. <= extracted_final_pipeline_result <= 1.


def execute_dag_and_get_final_pipeline_result(extracted_dag):
    """Utility function to extract the final result of a pipeline"""
    final_result_op = [node for node, out_degree in extracted_dag.out_degree() if out_degree == 0][0]
    label = "dag-exec-test"
    patch = get_intermediate_extraction_patch_after_node(singleton, None, final_result_op, label)
    patch.apply(extracted_dag)
    DagExecutor().execute(extracted_dag)
    extracted_final_pipeline_result = singleton.labels_to_extracted_plan_results[label]
    return extracted_final_pipeline_result
