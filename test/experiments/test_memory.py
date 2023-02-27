"""A filez to test  some stuff for the memory benchmark"""
import os
import sys
from unittest.mock import patch

from example_pipelines.healthcare import custom_monkeypatching
from mlmq._pipeline_analyzer import PipelineAnalyzer
from mlmq.utils import get_project_root


def run_base_pipeline_and_visualize_dag(dataset, tmpdir):
    """Run a base pipeline and visualize the DAGs for debugging and save them to same temporary directory"""
    pipeline_run_file = os.path.join(str(get_project_root()), "experiments", "end_to_end", "run_pipeline.py")
    with patch.object(sys, 'argv', ["mlmq", dataset, "fast_loading", "featurization_0", "xgboost"]):
        analysis_result_no_opt = PipelineAnalyzer \
            .on_pipeline_from_py_file(pipeline_run_file) \
            .add_custom_monkey_patching_modules([custom_monkeypatching]) \
            .execute()
    analysis_result_no_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "orig"))

# def test_folktables_large(tmpdir):
#     """
#     Tests whether the .py version of the inspector works
#     """
#     dataset = "folktables_5x"
#     run_base_pipeline_and_visualize_dag(dataset, tmpdir)
