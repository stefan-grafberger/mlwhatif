"""
Tests whether the optimization works
"""
# pylint: disable=too-many-locals,invalid-name
# TODO: Clean up these tests
import os

from experiments.optimizations.common_subexpression_elimination import execute_common_subexpression_elimination_worst_case, \
    execute_common_subexpression_elimunation_average_case, execute_common_subexpression_elimination_ideal_case


def test_common_subexpression_elimination_ideal_case(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_common_subexpression_elimination_ideal_case(0.5, tmpdir, variant_count)

    analysis = scenario_result_dict['analysis']
    analysis_result_with_opt = scenario_result_dict['analysis_result_with_opt']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']

    assert analysis_result_with_opt.analysis_to_result_reports[analysis].shape == (variant_count + 1, 2)
    assert analysis_result_without_any_opt.analysis_to_result_reports[analysis].shape == (variant_count + 1, 2)

    assert analysis_result_with_opt.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    analysis_result_with_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "with-opt-orig"))
    analysis_result_with_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if"))
    analysis_result_with_opt.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir),
                                                                              "with-opt-what-if-optimised"))

    analysis_result_without_any_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "without-any-opt-orig"))
    analysis_result_without_any_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-any-opt-what-if"))


def test_common_subexpression_elimination_average_case(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_common_subexpression_elimunation_average_case(0.5, tmpdir, variant_count)

    analysis = scenario_result_dict['analysis']
    analysis_result_with_opt = scenario_result_dict['analysis_result_with_opt']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']

    assert analysis_result_with_opt.analysis_to_result_reports[analysis].shape == (variant_count + 1, 2)
    assert analysis_result_without_any_opt.analysis_to_result_reports[analysis].shape == (variant_count + 1, 2)

    assert analysis_result_with_opt.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    analysis_result_with_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "with-opt-orig"))
    analysis_result_with_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if"))
    analysis_result_with_opt.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir),
                                                                              "with-opt-what-if-optimised"))

    analysis_result_without_any_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "without-any-opt-orig"))
    analysis_result_without_any_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-any-opt-what-if"))
    analysis_result_without_any_opt.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-any-opt-what-if-optimised"))


def test_common_subexpression_elimination_worst_case(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_common_subexpression_elimination_worst_case(0.5, tmpdir, variant_count)

    analysis = scenario_result_dict['analysis']
    analysis_result_with_opt = scenario_result_dict['analysis_result_with_opt']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']

    assert analysis_result_with_opt.analysis_to_result_reports[analysis].shape == (variant_count + 1, 2)
    assert analysis_result_without_any_opt.analysis_to_result_reports[analysis].shape == (variant_count + 1, 2)

    assert analysis_result_with_opt.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    analysis_result_with_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "with-opt-orig"))
    analysis_result_with_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if"))
    analysis_result_with_opt.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir),
                                                                              "with-opt-what-if-optimised"))

    analysis_result_without_any_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "without-any-opt-orig"))
    analysis_result_without_any_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-any-opt-what-if"))
    analysis_result_without_any_opt.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-any-opt-what-if-optimised"))
