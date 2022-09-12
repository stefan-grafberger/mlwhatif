"""
Tests whether the optimization works
"""
# pylint: disable=too-many-locals,invalid-name
# TODO: Clean up these tests
import os

from experiments.optimizations.udf_split_and_reuse import execute_udf_split_and_reuse_ideal_case, \
    execute_udf_split_and_reuse_average_case, execute_udf_split_and_reuse_worst_case_with_selectivity_safety_active, \
    execute_udf_split_and_reuse_worst_case_with_selectivity_inactive, \
    execute_udf_split_and_reuse_worst_case_with_constant


def test_udf_split_and_reuse_ideal_case(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_udf_split_and_reuse_ideal_case(0.5, tmpdir, variant_count)

    data_corruption = scenario_result_dict['analysis']
    analysis_result_with_push_up_opt_rule = scenario_result_dict['analysis_result_with_push_up_opt_rule']
    analysis_result_with_reuse_opt_rule = scenario_result_dict['analysis_result_with_reuse_opt_rule']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']
    analysis_result_without_opt_rule = scenario_result_dict['analysis_result_without_opt_rule']

    assert analysis_result_with_reuse_opt_rule.analysis_to_result_reports[data_corruption].shape == (
        variant_count + 1, 2)
    assert analysis_result_with_push_up_opt_rule.analysis_to_result_reports[data_corruption].shape == \
           (variant_count + 1, 2)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)

    assert analysis_result_with_reuse_opt_rule.runtime_info.what_if_optimized_estimated <= \
           analysis_result_with_push_up_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    analysis_result_with_reuse_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-reuse-opt-orig"))
    analysis_result_with_reuse_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-reuse-opt-what-if"))
    analysis_result_with_reuse_opt_rule.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir),
                                                                                         "with-reuse-opt-what-if-"
                                                                                         "optimised"))

    analysis_result_with_push_up_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-pushup-opt-orig"))
    analysis_result_with_push_up_opt_rule.save_what_if_dags_to_path(
        os.path.join(str(tmpdir), "with-pushup-opt-what-if"))
    analysis_result_with_push_up_opt_rule.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir),
                                                                                           "with-pushup-opt-what-if"
                                                                                           "-optimised"))

    analysis_result_without_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "without-opt-orig"))
    analysis_result_without_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-opt-what-if"))
    analysis_result_without_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-opt-what-if-optimised"))

    analysis_result_without_any_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "without-any-opt-orig"))
    analysis_result_without_any_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-any-opt-what-if"))
    analysis_result_without_any_opt.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-any-opt-what-if-optimised"))


def test_udf_split_and_reuse_average_case(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    # Difference is pretty small if its a corruption that has some backing C implementation, here, the ordering of
    #  the variant with udf split and reuse is important enough to be more significant than the optimization, when
    #  executing this test just once, the option listed first will always be faster between push_up with and without
    #  udf split reuse.
    variant_count = 4

    scenario_result_dict = execute_udf_split_and_reuse_average_case(0.5, tmpdir, variant_count)

    data_corruption = scenario_result_dict['analysis']
    analysis_result_with_push_up_opt_rule = scenario_result_dict['analysis_result_with_push_up_opt_rule']
    analysis_result_with_reuse_opt_rule = scenario_result_dict['analysis_result_with_reuse_opt_rule']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']
    analysis_result_without_opt_rule = scenario_result_dict['analysis_result_without_opt_rule']

    assert analysis_result_with_reuse_opt_rule.analysis_to_result_reports[data_corruption].shape == (
        variant_count + 1, 2)
    assert analysis_result_with_push_up_opt_rule.analysis_to_result_reports[data_corruption].shape == \
           (variant_count + 1, 2)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)

    assert analysis_result_with_reuse_opt_rule.runtime_info.what_if_optimized_estimated <= \
           analysis_result_with_push_up_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    analysis_result_with_reuse_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-reuse-opt-orig"))
    analysis_result_with_reuse_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-reuse-opt-what-if"))
    analysis_result_with_reuse_opt_rule.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir),
                                                                                         "with-reuse-opt-what-if-"
                                                                                         "optimised"))

    analysis_result_with_push_up_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-pushup-opt-orig"))
    analysis_result_with_push_up_opt_rule.save_what_if_dags_to_path(
        os.path.join(str(tmpdir), "with-pushup-opt-what-if"))
    analysis_result_with_push_up_opt_rule.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir),
                                                                                           "with-pushup-opt-what-if"
                                                                                           "-optimised"))

    analysis_result_without_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "without-opt-orig"))
    analysis_result_without_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-opt-what-if"))
    analysis_result_without_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-opt-what-if-optimised"))

    analysis_result_without_any_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "without-any-opt-orig"))
    analysis_result_without_any_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-any-opt-what-if"))
    analysis_result_without_any_opt.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-any-opt-what-if-optimised"))


def test_udf_split_and_reuse_worst_case_with_selectivity_safety_active(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_udf_split_and_reuse_worst_case_with_selectivity_safety_active(0.5, tmpdir,
                                                                                                 variant_count)

    data_corruption = scenario_result_dict['analysis']
    analysis_result_with_push_up_opt_rule = scenario_result_dict['analysis_result_with_push_up_opt_rule']
    analysis_result_with_reuse_opt_rule = scenario_result_dict['analysis_result_with_reuse_opt_rule']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']
    analysis_result_without_opt_rule = scenario_result_dict['analysis_result_without_opt_rule']

    assert analysis_result_with_reuse_opt_rule.analysis_to_result_reports[data_corruption].shape == (
        variant_count + 1, 2)
    assert analysis_result_with_push_up_opt_rule.analysis_to_result_reports[data_corruption].shape == \
           (variant_count + 1, 2)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)

    assert analysis_result_with_reuse_opt_rule.runtime_info.what_if_optimized_estimated <= \
           analysis_result_with_push_up_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    analysis_result_with_reuse_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-reuse-opt-orig"))
    analysis_result_with_reuse_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-reuse-opt-what-if"))
    analysis_result_with_reuse_opt_rule.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir),
                                                                                         "with-reuse-opt-what-if-"
                                                                                         "optimised"))

    analysis_result_with_push_up_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-pushup-opt-orig"))
    analysis_result_with_push_up_opt_rule.save_what_if_dags_to_path(
        os.path.join(str(tmpdir), "with-pushup-opt-what-if"))
    analysis_result_with_push_up_opt_rule.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir),
                                                                                          "with-pushup-opt-what-if"
                                                                                          "-optimised"))

    analysis_result_without_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "without-opt-orig"))
    analysis_result_without_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-opt-what-if"))
    analysis_result_without_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-opt-what-if-optimised"))

    analysis_result_without_any_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "without-any-opt-orig"))
    analysis_result_without_any_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-any-opt-what-if"))
    analysis_result_without_any_opt.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-any-opt-what-if-optimised"))


def test_udf_split_and_reuse_worst_case_with_selectivity_safety_inactive(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_udf_split_and_reuse_worst_case_with_selectivity_inactive(0.5, tmpdir,
                                                                                            variant_count)

    data_corruption = scenario_result_dict['analysis']
    analysis_result_with_push_up_opt_rule = scenario_result_dict['analysis_result_with_push_up_opt_rule']
    analysis_result_with_reuse_opt_rule = scenario_result_dict['analysis_result_with_reuse_opt_rule']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']
    analysis_result_without_opt_rule = scenario_result_dict['analysis_result_without_opt_rule']

    assert analysis_result_with_reuse_opt_rule.analysis_to_result_reports[data_corruption].shape == (
        variant_count + 1, 2)
    assert analysis_result_with_push_up_opt_rule.analysis_to_result_reports[data_corruption].shape == \
           (variant_count + 1, 2)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)

    assert analysis_result_with_reuse_opt_rule.runtime_info.what_if_optimized_estimated <= \
           analysis_result_with_push_up_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    analysis_result_with_reuse_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-reuse-opt-orig"))
    analysis_result_with_reuse_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-reuse-opt-what-if"))
    analysis_result_with_reuse_opt_rule.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir),
                                                                                         "with-reuse-opt-what-if-"
                                                                                         "optimised"))

    analysis_result_with_push_up_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-pushup-opt-orig"))
    analysis_result_with_push_up_opt_rule.save_what_if_dags_to_path(
        os.path.join(str(tmpdir), "with-pushup-opt-what-if"))
    analysis_result_with_push_up_opt_rule.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir),
                                                                                          "with-pushup-opt-what-if"
                                                                                          "-optimised"))

    analysis_result_without_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "without-opt-orig"))
    analysis_result_without_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-opt-what-if"))
    analysis_result_without_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-opt-what-if-optimised"))

    analysis_result_without_any_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "without-any-opt-orig"))
    analysis_result_without_any_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-any-opt-what-if"))
    analysis_result_without_any_opt.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-any-opt-what-if-optimised"))


def test_udf_split_and_reuse_worst_case_with_constant(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    variant_count = 4

    scenario_result_dict = execute_udf_split_and_reuse_worst_case_with_constant(0.5, tmpdir, variant_count)

    data_corruption = scenario_result_dict['analysis']
    analysis_result_with_push_up_opt_rule = scenario_result_dict['analysis_result_with_push_up_opt_rule']
    analysis_result_with_reuse_opt_rule = scenario_result_dict['analysis_result_with_reuse_opt_rule']
    analysis_result_without_any_opt = scenario_result_dict['analysis_result_without_any_opt']
    analysis_result_without_opt_rule = scenario_result_dict['analysis_result_without_opt_rule']

    assert analysis_result_with_reuse_opt_rule.analysis_to_result_reports[data_corruption].shape == (
        variant_count + 1, 2)
    assert analysis_result_with_push_up_opt_rule.analysis_to_result_reports[data_corruption].shape == \
           (variant_count + 1, 2)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)

    assert analysis_result_with_reuse_opt_rule.runtime_info.what_if_optimized_estimated <= \
           analysis_result_with_push_up_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    analysis_result_with_reuse_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-reuse-opt-orig"))
    analysis_result_with_reuse_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-reuse-opt-what-if"))
    analysis_result_with_reuse_opt_rule.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir),
                                                                                         "with-reuse-opt-what-if-"
                                                                                         "optimised"))

    analysis_result_with_push_up_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-pushup-opt-orig"))
    analysis_result_with_push_up_opt_rule.save_what_if_dags_to_path(
        os.path.join(str(tmpdir), "with-pushup-opt-what-if"))
    analysis_result_with_push_up_opt_rule.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir),
                                                                                          "with-pushup-opt-what-if"
                                                                                          "-optimised"))

    analysis_result_without_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "without-opt-orig"))
    analysis_result_without_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-opt-what-if"))
    analysis_result_without_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-opt-what-if-optimised"))

    analysis_result_without_any_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "without-any-opt-orig"))
    analysis_result_without_any_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-any-opt-what-if"))
    analysis_result_without_any_opt.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-any-opt-what-if-optimised"))
