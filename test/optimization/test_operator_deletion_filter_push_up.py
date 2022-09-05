"""
Tests whether the optimization works
"""
# pylint: disable=too-many-locals,invalid-name
# TODO: Clean up these tests
import os

from experiments.operator_deletion_filter_push_up import execute_operator_deletion_filter_push_up_ideal_case, \
    execute_operator_deletion_filter_push_up_worst_case_safety_inactive, \
    execute_operator_deletion_filter_push_up_worst_case_safety_active, \
    execute_operator_deletion_filter_push_up_worst_case_safety_too_defensive_for_scenario, \
    execute_operator_deletion_filter_push_up_average_case, \
    execute_operator_deletion_filter_push_up_worst_case_only_some_filters_worth_pushing_up


def test_operator_deletion_filter_push_up_ideal_case(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_operator_deletion_filter_push_up_ideal_case(0.5, tmpdir, variant_count)

    data_corruption = scenario_result_dict['analysis']
    analysis_result_with_opt_rule = scenario_result_dict['analysis_result_with_opt_rule']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']
    analysis_result_without_opt_rule = scenario_result_dict['analysis_result_without_opt_rule']

    assert analysis_result_with_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)

    assert analysis_result_with_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    analysis_result_with_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-opt-orig"))
    analysis_result_with_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if"))
    analysis_result_with_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "with-opt-what-if-optimised"))

    analysis_result_without_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "without-opt-orig"))
    analysis_result_without_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-opt-what-if"))
    analysis_result_without_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-opt-what-if-optimised"))

    analysis_result_without_any_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "without-any-opt-orig"))
    analysis_result_without_any_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-any-opt-what-if"))
    analysis_result_without_any_opt.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-any-opt-what-if-optimised"))


def test_operator_deletion_filter_push_up_worst_case_safety_inactive(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_operator_deletion_filter_push_up_worst_case_safety_inactive(0.5, tmpdir,
                                                                                               variant_count)

    data_corruption = scenario_result_dict['analysis']
    analysis_result_with_opt_rule = scenario_result_dict['analysis_result_with_opt_rule']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']
    analysis_result_without_opt_rule = scenario_result_dict['analysis_result_without_opt_rule']

    assert analysis_result_with_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)

    assert analysis_result_with_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    analysis_result_with_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-opt-orig"))
    analysis_result_with_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if"))
    analysis_result_with_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "with-opt-what-if-optimised"))

    analysis_result_without_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "without-opt-orig"))
    analysis_result_without_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-opt-what-if"))
    analysis_result_without_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-opt-what-if-optimised"))

    analysis_result_without_any_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "without-any-opt-orig"))
    analysis_result_without_any_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-any-opt-what-if"))
    analysis_result_without_any_opt.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-any-opt-what-if-optimised"))


def test_operator_deletion_filter_push_up_worst_case_safety_active(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_operator_deletion_filter_push_up_worst_case_safety_active(0.5, tmpdir,
                                                                                             variant_count)

    data_corruption = scenario_result_dict['analysis']
    analysis_result_with_opt_rule = scenario_result_dict['analysis_result_with_opt_rule']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']
    analysis_result_without_opt_rule = scenario_result_dict['analysis_result_without_opt_rule']

    assert analysis_result_with_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)

    assert analysis_result_with_opt_rule.runtime_info.what_if_optimized_estimated <= \
           analysis_result_without_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    analysis_result_with_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-opt-orig"))
    analysis_result_with_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if"))
    analysis_result_with_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "with-opt-what-if-optimised"))

    analysis_result_without_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "without-opt-orig"))
    analysis_result_without_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-opt-what-if"))
    analysis_result_without_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-opt-what-if-optimised"))

    analysis_result_without_any_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "without-any-opt-orig"))
    analysis_result_without_any_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-any-opt-what-if"))
    analysis_result_without_any_opt.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-any-opt-what-if-optimised"))


def test_operator_deletion_filter_push_up_average_case(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_operator_deletion_filter_push_up_average_case(0.5, tmpdir, variant_count)

    data_corruption = scenario_result_dict['analysis']
    analysis_result_with_opt_rule = scenario_result_dict['analysis_result_with_opt_rule']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']
    analysis_result_without_opt_rule = scenario_result_dict['analysis_result_without_opt_rule']

    assert analysis_result_with_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)

    assert analysis_result_with_opt_rule.runtime_info.what_if_optimized_estimated <= \
           analysis_result_without_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    analysis_result_with_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-opt-orig"))
    analysis_result_with_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if"))
    analysis_result_with_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "with-opt-what-if-optimised"))

    analysis_result_without_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "without-opt-orig"))
    analysis_result_without_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-opt-what-if"))
    analysis_result_without_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-opt-what-if-optimised"))

    analysis_result_without_any_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "without-any-opt-orig"))
    analysis_result_without_any_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-any-opt-what-if"))
    analysis_result_without_any_opt.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-any-opt-what-if-optimised"))


def test_operator_deletion_filter_push_up_worst_case_safety_too_defensive_for_scenario(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_operator_deletion_filter_push_up_worst_case_safety_too_defensive_for_scenario(
        0.5, tmpdir, variant_count)

    data_corruption = scenario_result_dict['analysis']
    analysis_result_with_opt_rule = scenario_result_dict['analysis_result_with_opt_rule']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']
    analysis_result_without_opt_rule = scenario_result_dict['analysis_result_without_opt_rule']

    assert analysis_result_with_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)

    assert analysis_result_with_opt_rule.runtime_info.what_if_optimized_estimated <= \
           analysis_result_without_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    analysis_result_with_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-opt-orig"))
    analysis_result_with_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if"))
    analysis_result_with_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "with-opt-what-if-optimised"))

    analysis_result_without_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "without-opt-orig"))
    analysis_result_without_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-opt-what-if"))
    analysis_result_without_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-opt-what-if-optimised"))

    analysis_result_without_any_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "without-any-opt-orig"))
    analysis_result_without_any_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-any-opt-what-if"))
    analysis_result_without_any_opt.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-any-opt-what-if-optimised"))


def test_operator_deletion_filter_push_up_worst_case_only_some_filters_worth_pushing_up(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_operator_deletion_filter_push_up_worst_case_only_some_filters_worth_pushing_up(
        0.5, tmpdir, variant_count)

    data_corruption = scenario_result_dict['analysis']
    analysis_result_with_opt_rule = scenario_result_dict['analysis_result_with_opt_rule']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']
    analysis_result_without_opt_rule = scenario_result_dict['analysis_result_without_opt_rule']

    assert analysis_result_with_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 5)

    assert analysis_result_with_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    analysis_result_with_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-opt-orig"))
    analysis_result_with_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if"))
    analysis_result_with_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "with-opt-what-if-optimised"))

    analysis_result_without_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "without-opt-orig"))
    analysis_result_without_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-opt-what-if"))
    analysis_result_without_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-opt-what-if-optimised"))

    analysis_result_without_any_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "without-any-opt-orig"))
    analysis_result_without_any_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-any-opt-what-if"))
    analysis_result_without_any_opt.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-any-opt-what-if-optimised"))
