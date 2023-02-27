"""
Tests whether the optimization works
"""
# pylint: disable=too-many-locals,invalid-name
# TODO: Clean up these tests
import os

from experiments.optimizations.simple_filter_addition_push_up import execute_filter_addition_push_up_ideal_case, \
    execute_filter_addition_push_up_average_case, execute_filter_addition_push_up_worst_case_no_original_pipeline, \
    execute_filter_addition_push_up_worst_case_original_pipeline, \
    execute_filter_addition_push_up_worst_case_no_original_pipeline_heuristic


def test_filter_push_up_ideal_case(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_filter_addition_push_up_ideal_case(1.0, tmpdir, variant_count)

    data_corruption = scenario_result_dict['analysis']
    analysis_result_with_opt_rule = scenario_result_dict['analysis_result_with_opt_rule']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']
    analysis_result_without_opt_rule = scenario_result_dict['analysis_result_without_opt_rule']

    assert analysis_result_with_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 3)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 3)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 3)

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


def test_filter_push_up_average_case(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_filter_addition_push_up_average_case(0.5, tmpdir, variant_count)

    data_corruption = scenario_result_dict['analysis']
    analysis_result_with_opt_rule = scenario_result_dict['analysis_result_with_opt_rule']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']
    analysis_result_without_opt_rule = scenario_result_dict['analysis_result_without_opt_rule']

    assert analysis_result_with_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 3)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 3)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 3)

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


def test_filter_push_up_worst_case_no_original_pipeline(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_filter_addition_push_up_worst_case_no_original_pipeline(0.5, tmpdir,
                                                                                           variant_count)

    data_corruption = scenario_result_dict['analysis']
    analysis_result_with_opt_rule = scenario_result_dict['analysis_result_with_opt_rule']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']
    analysis_result_without_opt_rule = scenario_result_dict['analysis_result_without_opt_rule']

    assert analysis_result_with_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 3)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 3)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 3)

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


def test_filter_push_up_worst_case_original_pipeline(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_filter_addition_push_up_worst_case_original_pipeline(0.5, tmpdir,
                                                                                        variant_count)

    data_corruption = scenario_result_dict['analysis']
    analysis_result_with_opt_rule = scenario_result_dict['analysis_result_with_opt_rule']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']
    analysis_result_without_opt_rule = scenario_result_dict['analysis_result_without_opt_rule']

    assert analysis_result_with_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 3)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 3)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 3)

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


def test_filter_push_up_worst_case_no_original_pipeline_heuristic(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_filter_addition_push_up_worst_case_no_original_pipeline_heuristic(1.0, tmpdir,
                                                                                                     variant_count)

    data_corruption = scenario_result_dict['analysis']
    analysis_result_with_opt_rule = scenario_result_dict['analysis_result_with_opt_rule']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']
    analysis_result_without_opt_rule = scenario_result_dict['analysis_result_without_opt_rule']

    assert analysis_result_with_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 3)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 3)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 3)

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
