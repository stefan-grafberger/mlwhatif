"""
Tests whether the optimization works
"""
# pylint: disable=too-many-locals,invalid-name
import os
import sys
from unittest.mock import patch

from example_pipelines.healthcare import custom_monkeypatching
from experiments.end_to_end.benchmarks_end_to_end import get_analysis_for_scenario_and_dataset
from mlwhatif import PipelineAnalyzer
from mlwhatif.testing._testing_helper_utils import run_scenario_and_visualize_dags
from mlwhatif.utils import get_project_root


def test_data_corruption_reviews(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_corruption"
    dataset = "reviews"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape == (26, 3)


def test_permutation_feature_importance_reviews(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "feature_importance"
    dataset = "reviews"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape == (6, 2)


def test_data_cleaning_reviews(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_cleaning"
    dataset = "reviews"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape == (23, 4)


def test_operator_impact_reviews(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "operator_impact"
    dataset = "reviews"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape == (4, 5)


def test_data_corruption_healthcare(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_corruption"
    dataset = "healthcare"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape == (31, 3)


def test_feature_importance_healthcare(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "feature_importance"
    dataset = "healthcare"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape == (7, 2)


def test_data_cleaning_healthcare(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_cleaning"
    dataset = "healthcare"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape == (26, 4)


def test_operator_impact_healthcare(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "operator_impact"
    dataset = "healthcare"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape == (3, 5)


def test_data_corruption_folktables(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_corruption"
    dataset = "folktables"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape == (33, 3)


def test_feature_importance_folktables(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "feature_importance"
    dataset = "folktables"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape == (9, 2)


def test_data_cleaning_folktables(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_cleaning"
    dataset = "folktables"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape == (34, 4)


def test_operator_impact_folktables(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "operator_impact"
    dataset = "folktables"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape == (2, 5)


def test_data_corruption_cardio(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_corruption"
    dataset = "cardio"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape == (34, 3)


def test_feature_importance_cardio(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "feature_importance"
    dataset = "cardio"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape == (12, 2)


def test_data_cleaning_cardio(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_cleaning"
    dataset = "cardio"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape == (43, 4)


def test_operator_impact_cardio(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "operator_impact"
    dataset = "cardio"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape == (2, 5)


def test_data_corruption_sneakers(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_corruption"
    dataset = "sneakers"

    pipeline_run_file = os.path.join(str(get_project_root()), "experiments", "end_to_end", "run_pipeline.py")
    analysis = get_analysis_for_scenario_and_dataset(scenario, dataset)
    with patch.object(sys, 'argv', ["mlwhatif", dataset, "fast_loading", "image", "image"]):
        analysis_result_no_opt = PipelineAnalyzer \
            .on_pipeline_from_py_file(pipeline_run_file) \
            .add_custom_monkey_patching_modules([custom_monkeypatching])\
            .add_what_if_analysis(analysis) \
            .execute()
    analysis_result_no_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "with-opt-orig"))
    analysis_result_no_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if"))
    analysis_result_no_opt.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if-optimised"))
    analysis_output = analysis_result_no_opt.analysis_to_result_reports[analysis]
    assert analysis_output.shape == (4, 3)


def test_data_cleaning_sneakers(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_cleaning"
    dataset = "sneakers"

    pipeline_run_file = os.path.join(str(get_project_root()), "experiments", "end_to_end", "run_pipeline.py")
    analysis = get_analysis_for_scenario_and_dataset(scenario, dataset)
    with patch.object(sys, 'argv', ["mlwhatif", dataset, "fast_loading", "image", "image"]):
        analysis_result_no_opt = PipelineAnalyzer \
            .on_pipeline_from_py_file(pipeline_run_file) \
            .add_custom_monkey_patching_modules([custom_monkeypatching])\
            .add_what_if_analysis(analysis) \
            .execute()
    analysis_result_no_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "with-opt-orig"))
    analysis_result_no_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if"))
    analysis_result_no_opt.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if-optimised"))
    analysis_output = analysis_result_no_opt.analysis_to_result_reports[analysis]
    assert analysis_output.shape == (3, 4)
