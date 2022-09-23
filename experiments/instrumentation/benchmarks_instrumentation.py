# pylint: disable=invalid-name,missing-module-docstring
import ast
import logging
import os
import random
# import subprocess
import sys
import time
import warnings
from unittest.mock import patch

import numpy
import pandas

from example_pipelines.healthcare import custom_monkeypatching
from experiments.end_to_end import run_pipeline
from mlwhatif import PipelineAnalyzer
# Make sure this code is not executed during imports
from mlwhatif.utils import get_project_root

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    if len(sys.argv) < 5:
        logger.info("dataset_name data_loading_name featurization_name model_name")

    dataset_name = sys.argv[1]
    data_loading_name = sys.argv[2]
    featurization_name = sys.argv[3]
    model_name = sys.argv[4]

    num_repetitions = 30
    seeds = list(range(42, 42 + num_repetitions))
    assert len(seeds) == num_repetitions

    pipeline_run_file = os.path.join(str(get_project_root()), "experiments", "end_to_end",
                                     "../end_to_end/run_pipeline.py")
    # Warm-up run to ignore effect of imports
    synthetic_cmd_args = ['mlwhatif']
    cmd_args = sys.argv[1:].copy()
    synthetic_cmd_args.extend(cmd_args)

    logging.info(f'Patching sys.argv with {synthetic_cmd_args}')
    current_directory = os.getcwd()
    output_directory = os.path.join(current_directory, r'../instrumentation/instrumentation-benchmark-results')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with patch.object(sys, 'argv', synthetic_cmd_args):
        _ = PipelineAnalyzer \
            .on_pipeline_from_py_file(pipeline_run_file) \
            .add_custom_monkey_patching_modules([custom_monkeypatching]) \
            .execute()
        # analysis_result.save_original_dag_to_path(os.path.join(output_directory, "test"))
        # FIXME: Remove this later

    result_df_repetitions = []
    result_df_dataset = []
    result_df_data_loading = []
    result_df_featurization = []
    result_df_model = []
    result_df_total_exec_opt = []
    result_df_total_exec_no_instrum_main_func = []
    result_df_total_exec_no_instrum_load_ast_compile = []

    result_df_instrum_original_pipeline_estimated = []
    result_df_instrum_original_pipeline_importing_and_monkeypatching = []
    result_df_instrum_original_pipeline_without_importing_and_monkeypatching = []
    result_df_instrum_original_pipeline_model_training = []
    result_df_instrum_original_pipeline_train_shape = []
    result_df_instrum_original_pipeline_test_shape = []

    for repetition, seed in enumerate(seeds):
        numpy.random.seed(seed)
        random.seed(seed)
        logger.info(f'# Optimization benchmark for instrumentation with args [{sys.argv[2:]}] '
                    f'-- repetition {repetition + 1} of {num_repetitions}')

        print("main_func baseline")
        with patch.object(sys, 'argv', synthetic_cmd_args):
            total_no_instrum_main_func_exec_start = time.time()
            run_pipeline.main_function()
        total_no_instrum_main_func_exec_duration = (time.time() - total_no_instrum_main_func_exec_start) * 1000

        print("load_ast_compile baseline")
        # subprocess_cmd = f'python {pipeline_run_file} {" ".join(synthetic_cmd_args[1:])}'
        with patch.object(sys, 'argv', synthetic_cmd_args):
            total_no_instrum_load_ast_compile_exec_start = time.time()
            with open(pipeline_run_file) as file:
                source_code = file.read()
            parsed_ast = ast.parse(source_code)
            exec(compile(parsed_ast, filename="does-not-matter", mode="exec"))
        # subprocess.Popen(subprocess_cmd, shell=True).wait()
        total_no_instrum_load_ast_compile_exec_duration = (time.time() - total_no_instrum_load_ast_compile_exec_start) * 1000

        print("with instrumentation")
        with patch.object(sys, 'argv', synthetic_cmd_args):
            total_instrum_exec_start = time.time()
            analysis_result_instrum = PipelineAnalyzer \
                .on_pipeline_from_py_file(pipeline_run_file) \
                .add_custom_monkey_patching_modules([custom_monkeypatching]) \
                .execute()
        total_instrum_exec_duration = (time.time() - total_instrum_exec_start) * 1000


        result_df_repetitions.append(repetition)
        result_df_dataset.append(dataset_name)
        result_df_data_loading.append(data_loading_name)
        result_df_featurization.append(featurization_name)
        result_df_model.append(model_name)
        result_df_total_exec_opt.append(total_instrum_exec_duration)
        result_df_total_exec_no_instrum_main_func.append(total_no_instrum_main_func_exec_duration)
        result_df_total_exec_no_instrum_load_ast_compile.append(total_no_instrum_load_ast_compile_exec_duration)

        result_df_instrum_original_pipeline_estimated.append(
            analysis_result_instrum.runtime_info.original_pipeline_estimated)
        result_df_instrum_original_pipeline_importing_and_monkeypatching.append(
            analysis_result_instrum.runtime_info.original_pipeline_importing_and_monkeypatching)
        result_df_instrum_original_pipeline_without_importing_and_monkeypatching.append(
            analysis_result_instrum.runtime_info.original_pipeline_without_importing_and_monkeypatching)
        result_df_instrum_original_pipeline_train_shape.append(
            analysis_result_instrum.runtime_info.original_pipeline_train_data_shape)
        result_df_instrum_original_pipeline_test_shape.append(
            analysis_result_instrum.runtime_info.original_pipeline_test_data_shape)
        result_df_instrum_original_pipeline_model_training.append(
            analysis_result_instrum.runtime_info.original_model_training)

        logger.info(f'# Finished -- repetition {repetition + 1} of {num_repetitions} ')
        # print(results_dict_items)

    result_df = pandas.DataFrame({'repetition': result_df_repetitions,
                                  'dataset': result_df_dataset,
                                  'data_loading': result_df_data_loading,
                                  'featurization': result_df_featurization,
                                  'model': result_df_model,
                                  'total_exec_duration_with_instrum': result_df_total_exec_opt,
                                  'total_exec_duration_without_instrum_main_func':
                                      result_df_total_exec_no_instrum_main_func,
                                  'total_exec_duration_without_instrum_load_ast_compile':
                                      result_df_total_exec_no_instrum_load_ast_compile,
                                  'instrum_original_pipeline_estimated': result_df_instrum_original_pipeline_estimated,
                                  'instrum_original_pipeline_importing_and_monkeypatching':
                                      result_df_instrum_original_pipeline_importing_and_monkeypatching,
                                  'instrum_original_pipeline_without_importing_and_monkeypatching':
                                      result_df_instrum_original_pipeline_without_importing_and_monkeypatching,
                                  'instrum_original_pipeline_model_training':
                                      result_df_instrum_original_pipeline_model_training,
                                  'instrum_original_pipeline_train_data_shape':
                                      result_df_instrum_original_pipeline_train_shape,
                                  'instrum_original_pipeline_test_data_shape':
                                      result_df_instrum_original_pipeline_test_shape,
                                  })
    result_df_path = os.path.join(output_directory, f"results-instrumentation-"
                                                    f"{dataset_name}-{data_loading_name}-"
                                                    f"{featurization_name}-{model_name}.csv")
    result_df.to_csv(result_df_path, index=False)
    logger.info(f'# Final results after all repetitions')
    logger.info(result_df)
