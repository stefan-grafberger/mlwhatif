"""
Tests whether the optimization works
"""
# pylint: disable=too-many-locals,invalid-name
# TODO: Clean up these tests

from mlwhatif.testing._testing_helper_utils import run_scenario_and_visualize_dags


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
    assert analysis_output.shape.shape == (23, 4)


def test_operator_impact_reviews(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "operator_impact"
    dataset = "reviews"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape.shape == (4, 5)


def test_data_corruption_healthcare(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_corruption"
    dataset = "healthcare"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape.shape == (25, 3)


def test_feature_importance_healthcare(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "feature_importance"
    dataset = "healthcare"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape.shape == (7, 2)


def test_data_cleaning_healthcare(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_cleaning"
    dataset = "healthcare"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape.shape == (26, 4)


def test_operator_impact_healthcare(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "operator_impact"
    dataset = "healthcare"
    analysis_output = run_scenario_and_visualize_dags(dataset, scenario, tmpdir)
    assert analysis_output.shape.shape == (3, 5)


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
