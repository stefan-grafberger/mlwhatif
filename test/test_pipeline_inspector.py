"""
Tests whether the fluent API works
"""

import networkx
from testfixtures import compare

from example_pipelines.healthcare import custom_monkeypatching
from example_pipelines import ADULT_SIMPLE_PY, ADULT_SIMPLE_IPYNB, HEALTHCARE_PY
from mlwhatif import PipelineInspector, OperatorType
from mlwhatif.testing._testing_helper_utils import get_expected_dag_adult_easy


def test_inspector_adult_easy_py_pipeline():
    """
    Tests whether the .py version of the inspector works
    """
    inspector_result = PipelineInspector\
        .on_pipeline_from_py_file(ADULT_SIMPLE_PY)\
        .execute()
    extracted_dag = inspector_result.dag
    expected_dag = get_expected_dag_adult_easy(ADULT_SIMPLE_PY)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))


def test_inspector_adult_easy_py_pipeline_without_inspections():
    """
    Tests whether the .py version of the inspector works
    """
    inspector_result = PipelineInspector\
        .on_pipeline_from_py_file(ADULT_SIMPLE_PY)\
        .execute()
    extracted_dag = inspector_result.dag
    expected_dag = get_expected_dag_adult_easy(ADULT_SIMPLE_PY)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))


def test_inspector_adult_easy_ipynb_pipeline():
    """
    Tests whether the .ipynb version of the inspector works
    """
    inspector_result = PipelineInspector\
        .on_pipeline_from_ipynb_file(ADULT_SIMPLE_IPYNB)\
        .execute()
    extracted_dag = inspector_result.dag
    expected_dag = get_expected_dag_adult_easy(ADULT_SIMPLE_IPYNB, 6)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))


def test_inspector_adult_easy_str_pipeline():
    """
    Tests whether the str version of the inspector works
    """
    with open(ADULT_SIMPLE_PY) as file:
        code = file.read()

        inspector_result = PipelineInspector\
            .on_pipeline_from_string(code)\
            .execute()
        extracted_dag = inspector_result.dag
        expected_dag = get_expected_dag_adult_easy("<string-source>")
        compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))


def test_inspector_additional_module():
    """
    Tests whether the str version of the inspector works
    """
    inspector_result = PipelineInspector \
        .on_pipeline_from_py_file(HEALTHCARE_PY) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .execute()

    assert_healthcare_pipeline_output_complete(inspector_result)


def test_inspector_additional_modules():
    """
    Tests whether the str version of the inspector works
    """
    inspector_result = PipelineInspector \
        .on_pipeline_from_py_file(HEALTHCARE_PY) \
        .add_custom_monkey_patching_modules([custom_monkeypatching]) \
        .execute()

    assert_healthcare_pipeline_output_complete(inspector_result)


def assert_healthcare_pipeline_output_complete(inspector_result):
    """ Assert that the healthcare DAG was extracted completely """
    for dag_node, _ in inspector_result.dag_node_to_inspection_results.items():
        assert dag_node.operator_info.operator != OperatorType.MISSING_OP
    assert len(inspector_result.dag) == 37
