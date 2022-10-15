"""
Tests whether the fluent API works
"""

import networkx
from testfixtures import compare

from example_pipelines.healthcare import custom_monkeypatching
from example_pipelines import ADULT_SIMPLE_PY, ADULT_SIMPLE_IPYNB, HEALTHCARE_PY, ADULT_COMPLEX_PY
from mlwhatif import PipelineAnalyzer, OperatorType
from mlwhatif.analysis._data_cleaning import DataCleaning, ErrorType
from mlwhatif.testing._testing_helper_utils import get_expected_dag_adult_easy


def test_inspector_adult_easy_py_pipeline():
    """
    Tests whether the .py version of the inspector works
    """
    inspector_result = PipelineAnalyzer\
        .on_pipeline_from_py_file(ADULT_SIMPLE_PY)\
        .execute()
    extracted_dag = inspector_result.original_dag
    expected_dag = get_expected_dag_adult_easy(ADULT_SIMPLE_PY)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))


def test_inspector_adult_easy_py_pipeline_without_inspections():
    """
    Tests whether the .py version of the inspector works
    """
    inspector_result = PipelineAnalyzer\
        .on_pipeline_from_py_file(ADULT_SIMPLE_PY)\
        .execute()
    extracted_dag = inspector_result.original_dag
    expected_dag = get_expected_dag_adult_easy(ADULT_SIMPLE_PY)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))


def test_inspector_adult_easy_ipynb_pipeline():
    """
    Tests whether the .ipynb version of the inspector works
    """
    inspector_result = PipelineAnalyzer\
        .on_pipeline_from_ipynb_file(ADULT_SIMPLE_IPYNB)\
        .execute()
    extracted_dag = inspector_result.original_dag
    expected_dag = get_expected_dag_adult_easy(ADULT_SIMPLE_IPYNB, 6)
    compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))


def test_inspector_adult_easy_str_pipeline():
    """
    Tests whether the str version of the inspector works
    """
    with open(ADULT_SIMPLE_PY) as file:
        code = file.read()

        inspector_result = PipelineAnalyzer\
            .on_pipeline_from_string(code)\
            .execute()
        extracted_dag = inspector_result.original_dag
        expected_dag = get_expected_dag_adult_easy("<string-source>")
        compare(networkx.to_dict_of_dicts(extracted_dag), networkx.to_dict_of_dicts(expected_dag))


def test_inspector_additional_module():
    """
    Tests whether the str version of the inspector works
    """
    inspector_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(HEALTHCARE_PY) \
        .add_custom_monkey_patching_module(custom_monkeypatching) \
        .execute()

    assert_healthcare_pipeline_output_complete(inspector_result)


def test_inspector_additional_modules():
    """
    Tests whether the str version of the inspector works
    """
    inspector_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(HEALTHCARE_PY) \
        .add_custom_monkey_patching_modules([custom_monkeypatching]) \
        .execute()

    assert_healthcare_pipeline_output_complete(inspector_result)


def test_dag_extraction_reuse():
    """
    Tests whether the Data Cleaning analysis works for a very simple pipeline with a DecisionTree score
    """

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(ADULT_COMPLEX_PY) \
        .execute()

    data_cleaning = DataCleaning({'education': ErrorType.CAT_MISSING_VALUES,
                                  'age': ErrorType.NUM_MISSING_VALUES,
                                  'hours-per-week': ErrorType.OUTLIERS,
                                  None: ErrorType.MISLABEL})

    analysis_result = PipelineAnalyzer \
        .on_previously_extracted_pipeline(analysis_result.dag_extraction_info) \
        .add_what_if_analysis(data_cleaning) \
        .execute()

    report = analysis_result.analysis_to_result_reports[data_cleaning]
    assert report.shape == (19, 4)


def test_estimation():
    """
    Tests whether the Data Cleaning analysis works for a very simple pipeline with a DecisionTree score
    """
    data_cleaning = DataCleaning({'education': ErrorType.CAT_MISSING_VALUES,
                                  'age': ErrorType.NUM_MISSING_VALUES,
                                  'hours-per-week': ErrorType.OUTLIERS,
                                  None: ErrorType.MISLABEL})

    estimation_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(ADULT_COMPLEX_PY) \
        .add_what_if_analysis(data_cleaning) \
        .estimate()

    estimation_result.print_estimate()

    analysis_result = PipelineAnalyzer \
        .on_previously_extracted_pipeline(estimation_result.dag_extraction_info) \
        .add_what_if_analysis(data_cleaning) \
        .execute()

    report = analysis_result.analysis_to_result_reports[data_cleaning]
    assert report.shape == (19, 4)


def assert_healthcare_pipeline_output_complete(inspector_result):
    """ Assert that the healthcare DAG was extracted completely """
    for dag_node, _ in inspector_result.analysis_to_result_reports.items():
        assert dag_node.operator_info.operator != OperatorType.MISSING_OP
    assert len(inspector_result.original_dag) == 52
