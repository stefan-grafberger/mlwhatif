"""
Tests whether the optimization works
"""
# pylint: disable=too-many-locals,invalid-name
# TODO: Clean up these tests
import os
import random
import sys
from unittest.mock import patch

import numpy

from example_pipelines.healthcare import custom_monkeypatching
from experiments.end_to_end.benchmarks_end_to_end import get_analysis_for_scenario_and_dataset
from mlwhatif import PipelineAnalyzer
from mlwhatif.utils import get_project_root


def test_data_corruption_reviews(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    scenario = "data_corruption"
    dataset = "reviews"

    pipeline_run_file = os.path.join(str(get_project_root()), "experiments", "end_to_end", "run_pipeline.py")
    analysis = get_analysis_for_scenario_and_dataset(scenario, dataset)

    with patch.object(sys, 'argv', ["mlwhatif", "reviews", "fast", "fast", "logistic_regression"]):
        analysis_result_no_opt = PipelineAnalyzer \
            .on_pipeline_from_py_file(pipeline_run_file) \
            .add_custom_monkey_patching_modules([custom_monkeypatching]) \
            .add_what_if_analysis(analysis) \
            .execute()

    analysis_result_no_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "with-opt-orig"))
    analysis_result_no_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if"))
    analysis_result_no_opt.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if-optimised"))

    assert analysis_result_no_opt.analysis_to_result_reports[analysis].shape == (13, 3)
