"""
Tests whether the optimization works
"""
# pylint: disable=too-many-locals,invalid-name
import os
import sys
from unittest.mock import patch

from example_pipelines.healthcare import custom_monkeypatching
from mlwhatif import PipelineAnalyzer, OperatorType
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
    assert analysis_output.shape == (21, 4)


def test_operator_impact_reviews(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "operator_impact"
    dataset = "reviews"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir, all_stages=True)
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
    assert analysis_output.shape == (28, 4)


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
    assert analysis_output.shape == (36, 4)


def test_operator_impact_folktables(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "operator_impact"
    dataset = "folktables"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape == (2, 5)


# Only works if the git lfs file is loaded, which is not the case during the GitHub CI
# def test_data_corruption_folktables_large(tmpdir):
#     """
#     Tests whether the .py version of the inspector works
#     """
#     scenario = "data_corruption"
#     dataset = "folktables_5x"
#     analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
#     assert analysis_output.shape == (33, 3)
#
#
# def test_feature_importance_folktables_large(tmpdir):
#     """
#     Tests whether the .py version of the inspector works
#     """
#     scenario = "feature_importance"
#     dataset = "folktables_5x"
#     analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
#     assert analysis_output.shape == (9, 2)
#
#
# def test_data_cleaning_folktables_large(tmpdir):
#     """
#     Tests whether the .py version of the inspector works
#     """
#     scenario = "data_cleaning"
#     dataset = "folktables_5x"
#     analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
#     assert analysis_output.shape == (36, 4)
#
#
# def test_operator_impact_folktables_large(tmpdir):
#     """
#     Tests whether the .py version of the inspector works
#     """
#     scenario = "operator_impact"
#     dataset = "folktables_5x"
#     analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
#     assert analysis_output.shape == (2, 5)
#
#
# def test_data_corruption_reviews_large(tmpdir):
#     """
#     Tests whether the .py version of the inspector works
#     """
#     scenario = "data_corruption"
#     dataset = "reviews_5x"
#     analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
#     assert analysis_output.shape == (26, 3)
#
#
# def test_permutation_feature_importance_reviews_large(tmpdir):
#     """
#     Tests whether the .py version of the inspector works
#     """
#     scenario = "feature_importance"
#     dataset = "reviews_5x"
#     analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
#     assert analysis_output.shape == (6, 2)
#
#
# def test_data_cleaning_reviews_large(tmpdir):
#     """
#     Tests whether the .py version of the inspector works
#     """
#     scenario = "data_cleaning"
#     dataset = "reviews_5x"
#     analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
#     assert analysis_output.shape == (21, 4)
#
#
# def test_operator_impact_reviews_large(tmpdir):
#     """
#     Tests whether the .py version of the inspector works
#     """
#     scenario = "operator_impact"
#     dataset = "reviews_5x"
#     analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
#     assert analysis_output.shape == (4, 5)


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
    assert analysis_output.shape == (45, 4)


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
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir, "image", "image")
    assert analysis_output.shape == (21, 3)


def test_data_cleaning_sneakers(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_cleaning"
    dataset = "sneakers"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir, "image", "image")
    assert analysis_output.shape == (5, 4)


def test_data_corruption_reddit(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_corruption"
    dataset = "reddit"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir, "reddit", "reddit")
    assert analysis_output.shape == (21, 3)


def test_feature_importance_reddit(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "feature_importance"
    dataset = "reddit"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir, "reddit", "reddit")
    assert analysis_output.shape == (2, 2)


def test_data_cleaning_reddit(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_cleaning"
    dataset = "reddit"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir, "reddit", "reddit")
    assert analysis_output.shape == (5, 4)


def test_operator_impact_reddit(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "operator_impact"
    dataset = "reddit"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir, "reddit", "reddit")
    assert analysis_output.shape == (3, 5)


def test_data_corruption_walmart_amazon(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_corruption"
    dataset = "walmart_amazon"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir, "walmart_amazon", "walmart_amazon")
    assert analysis_output.shape == (25, 3)


def test_feature_importance_walmart_amazon(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "feature_importance"
    dataset = "walmart_amazon"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir, "walmart_amazon", "walmart_amazon")
    assert analysis_output.shape == (5, 2)


def test_data_cleaning_walmart_amazon(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_cleaning"
    dataset = "walmart_amazon"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir, "walmart_amazon", "walmart_amazon")
    assert analysis_output.shape == (5, 4)


def test_operator_impact_walmart_amazon(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "operator_impact"
    dataset = "walmart_amazon"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir, "walmart_amazon", "walmart_amazon")
    assert analysis_output.shape == (2, 5)


def test_reddit(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    dataset = "reddit"
    featurization = "reddit"
    model = "reddit"
    run_base_pipeline_and_visualize_dag(dataset, featurization, model, tmpdir)


def test_walmart_amazon(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    dataset = "walmart_amazon"
    featurization = "walmart_amazon"
    model = "walmart_amazon"
    run_base_pipeline_and_visualize_dag(dataset, featurization, model, tmpdir)


def run_base_pipeline_and_visualize_dag(dataset, featurization, model, tmpdir):
    """Run a base pipeline and visualize the DAGs for debugging and save them to same temporary directory"""
    pipeline_run_file = os.path.join(str(get_project_root()), "experiments", "end_to_end", "run_pipeline.py")
    with patch.object(sys, 'argv', ["mlwhatif", dataset, "fast_loading", featurization, model]):
        analysis_result = PipelineAnalyzer \
            .on_pipeline_from_py_file(pipeline_run_file) \
            .add_custom_monkey_patching_modules([custom_monkeypatching]) \
            .execute()
    analysis_result.save_original_dag_to_path(os.path.join(str(tmpdir), "orig"))
    nodes = list(analysis_result.original_dag.nodes)
    for node in nodes:
        if node.operator_info.operator == OperatorType.MISSING_OP:
            raise Exception(f"Missing Ops not supported currently! The operator: {node}")
