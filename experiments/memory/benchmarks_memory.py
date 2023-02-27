# pylint: disable-all
import concurrent
import gc
import logging
import os
import random
import sys
import time
import warnings
from concurrent.futures.thread import ThreadPoolExecutor
from unittest.mock import patch

import numpy
import pandas
import psutil

from example_pipelines.healthcare import custom_monkeypatching
from experiments.memory.run_pipeline import main_function
from experiments.memory.memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)


from mlwhatif.utils import get_project_root
pipeline_run_file = os.path.join(str(get_project_root()), "experiments", "memory", "run_pipeline.py")


def get_analysis_for_scenario_and_dataset(scenario_name, dataset_name, variant_count):
    from mlwhatif.analysis._data_cleaning import DataCleaning, ErrorType
    from mlwhatif.analysis._data_corruption import DataCorruption, CorruptionType

    if scenario_name == 'data_corruption' and dataset_name in {'reviews', 'reviews_5x', 'reviews_10x'}:
        corruption_percentages = [(1. / variant_count) * variant_number for variant_number
                                  in range(1, variant_count + 1)]
        analysis = DataCorruption([('vine', CorruptionType.CATEGORICAL_SHIFT),
                                   ('review_body', CorruptionType.BROKEN_CHARACTERS),
                                   # ('category', CorruptionType.CATEGORICAL_SHIFT),
                                   ('total_votes', CorruptionType.SCALING),
                                   ('star_rating', CorruptionType.GAUSSIAN_NOISE)],
                                  corruption_percentages=corruption_percentages)
    elif scenario_name == 'data_cleaning' and dataset_name in {'reviews', 'reviews_5x', 'reviews_10x'}:
        columns_with_error = [('star_rating', ErrorType.NUM_MISSING_VALUES) for _ in range(variant_count)]
        analysis = DataCleaning(columns_with_error)
    else:
        raise ValueError(f"Invalid scenario or dataset: {scenario_name} {dataset_name}!")
    return analysis


def exec_with_memory_tracking(exec_func, exec_strategy, analysis):
    with ThreadPoolExecutor() as thread_executor:
        monitor = MemoryMonitor()
        mem_thread = thread_executor.submit(monitor.measure_usage)
        try:
            fn_thread = thread_executor.submit(exec_func, exec_strategy=exec_strategy, analysis=analysis)
            fn_result = fn_thread.result()
        finally:
            monitor.keep_measuring = False
            max_usage_per_sec = mem_thread.result()
        return fn_result, max_usage_per_sec


def exec_and_get_memory_info(exec_func, exec_strategy, analysis):
    with concurrent.futures.ProcessPoolExecutor() as process_executor:
        result = process_executor.submit(exec_with_memory_tracking, exec_func,
                                         exec_strategy=exec_strategy, analysis=analysis).result()
    return result


def exec_warmup(exec_strategy, analysis):
    print(f"Warmup!")
    print(f"Patched command line arguments: {sys.argv}")
    from mlwhatif import PipelineAnalyzer
    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(pipeline_run_file) \
        .add_custom_monkey_patching_modules([custom_monkeypatching]) \
        .overwrite_exec_strategy(exec_strategy) \
        .execute().runtime_info
    return analysis_result


def exec_opt(exec_strategy, analysis):
    print(f"Opt!")
    print(f"Patched command line arguments: {sys.argv}")
    from mlwhatif import PipelineAnalyzer
    analysis_result_opt = PipelineAnalyzer \
        .on_pipeline_from_py_file(pipeline_run_file) \
        .add_custom_monkey_patching_modules([custom_monkeypatching]) \
        .skip_multi_query_optimization(False) \
        .add_what_if_analysis(analysis) \
        .overwrite_exec_strategy(exec_strategy) \
        .execute().runtime_info
    return analysis_result_opt


def exec_no_opt(exec_strategy, analysis):
    print(f"No Opt!")
    print(f"Patched command line arguments: {sys.argv}")
    from mlwhatif import PipelineAnalyzer
    analysis_result_no_opt = PipelineAnalyzer \
        .on_pipeline_from_py_file(pipeline_run_file) \
        .add_custom_monkey_patching_modules([custom_monkeypatching]) \
        .skip_multi_query_optimization(True) \
        .add_what_if_analysis(analysis) \
        .overwrite_exec_strategy(exec_strategy) \
        .execute().runtime_info
    return analysis_result_no_opt


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    if len(sys.argv) < 6:
        logger.info("scenario_name dataset_name data_loading_name featurization_name model_name exec_stragegy "
                    "variant_count")

    scenario_name = sys.argv[1]
    dataset_name = sys.argv[2]
    data_loading_name = sys.argv[3]
    featurization_name = sys.argv[4]
    model_name = sys.argv[5]
    exec_strategy = sys.argv[6] == "dfs"
    variant_count = int(sys.argv[7])

    num_repetitions = 7
    seeds = [42, 43, 44, 45, 46, 47, 48][:num_repetitions]
    assert len(seeds) == num_repetitions

    # Warm-up run to ignore effect of imports
    synthetic_cmd_args = ['mlwhatif']
    cmd_args = sys.argv[2:6].copy()
    synthetic_cmd_args.extend(cmd_args)

    analysis = get_analysis_for_scenario_and_dataset(scenario_name, dataset_name, variant_count)

    logging.info(f'Patching sys.argv with {synthetic_cmd_args}')
    current_directory = os.getcwd()
    output_directory = os.path.join(current_directory, r'memory-benchmark-results')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with patch.object(sys, 'argv', synthetic_cmd_args):
        _, _ = exec_and_get_memory_info(exec_warmup, exec_strategy, analysis)
        # analysis_result.save_original_dag_to_path(os.path.join(output_directory, "test"))

    gc.collect()
    memory_inbetween_0 = psutil.Process(os.getpid()).memory_info().rss

    with patch.object(sys, 'argv', synthetic_cmd_args):
        print(f"Original!")
        _, memory_info_base = exec_and_get_memory_info(main_function, exec_strategy, analysis)

    gc.collect()
    memory_inbetween_1 = psutil.Process(os.getpid()).memory_info().rss

    result_df_repetitions = []
    result_df_scenarios = []
    result_df_dataset = []
    result_df_data_loading = []
    result_df_featurization = []
    result_df_model = []
    result_df_total_exec_opt = []
    result_df_total_exec_no_opt = []
    result_df_variant_factor = []
    result_df_variant_actual_count = []

    result_df_opt_what_if_optimized_estimated = []
    result_df_opt_what_if_unoptimized_estimated = []
    result_df_opt_what_if_optimization_saving_estimated = []
    result_df_opt_original_pipeline_estimated = []
    result_df_opt_original_pipeline_importing_and_monkeypatching = []
    result_df_opt_original_pipeline_without_importing_and_monkeypatching = []
    result_df_opt_original_pipeline_model_training = []
    result_df_opt_original_pipeline_train_shape = []
    result_df_opt_original_pipeline_test_shape = []
    result_df_opt_what_if_plan_generation = []
    result_df_opt_what_if_query_optimization_duration = []
    result_df_opt_what_if_execution = []
    result_df_opt_what_if_execution_combined_model_training = []
    result_df_no_opt_what_if_plan_generation = []
    result_df_no_opt_what_if_query_optimization_duration = []
    result_df_no_opt_what_if_execution = []
    result_df_no_opt_what_if_execution_combined_model_training = []

    result_df_base_memory_per_sec_psutil_rss = []
    result_df_base_memory_per_sec_psutil_vms = []
    result_df_base_memory_max_psutil_rss = []
    result_df_no_opt_memory_per_sec_psutil_rss = []
    result_df_no_opt_memory_per_sec_psutil_vms = []
    result_df_no_opt_memory_max_psutil_rss = []
    result_df_opt_memory_per_sec_psutil_rss = []
    result_df_opt_memory_per_sec_psutil_vms = []
    result_df_opt_memory_max_psutil_rss = []

    result_df_memory_inbetween_0 = []
    result_df_memory_inbetween_1 = []
    result_df_memory_inbetween_2 = []

    for repetition, seed in enumerate(seeds):
        numpy.random.seed(seed)
        random.seed(seed)
        logger.info(f'# Optimization benchmark for {scenario_name} with args [{sys.argv[2:]}] '
                    f'-- repetition {repetition + 1} of {num_repetitions}')

        total_opt_exec_start = time.time()
        with patch.object(sys, 'argv', synthetic_cmd_args):
            analysis_result_opt, memory_info_opt = exec_and_get_memory_info(exec_opt, exec_strategy, analysis)
            # print(f"Opt peak memory usage per sec: {memory_info_opt}")
        total_opt_exec_duration = (time.time() - total_opt_exec_start) * 1000

        gc.collect()
        memory_inbetween_2 = psutil.Process(os.getpid()).memory_info().rss

        total_no_opt_exec_start = time.time()
        with patch.object(sys, 'argv', synthetic_cmd_args):
            analysis_result_no_opt, memory_info_no_opt = exec_and_get_memory_info(exec_no_opt, exec_strategy, analysis)
            # print(f"No opt peak memory usage per sec: {memory_info_no_opt}")

        total_no_opt_exec_duration = (time.time() - total_no_opt_exec_start) * 1000

        result_df_repetitions.append(repetition)
        result_df_scenarios.append(scenario_name)
        result_df_dataset.append(dataset_name)
        result_df_data_loading.append(data_loading_name)
        result_df_featurization.append(featurization_name)
        result_df_model.append(model_name)
        result_df_total_exec_opt.append(total_opt_exec_duration)
        result_df_total_exec_no_opt.append(total_no_opt_exec_duration)
        result_df_variant_factor.append(variant_count)
        result_df_variant_actual_count.append(1 + variant_count * 4)

        result_df_opt_what_if_optimized_estimated.append(analysis_result_opt.what_if_optimized_estimated)
        result_df_opt_what_if_unoptimized_estimated.append(
            analysis_result_opt.what_if_unoptimized_estimated)
        result_df_opt_what_if_optimization_saving_estimated.append(
            analysis_result_opt.what_if_optimization_saving_estimated)
        result_df_opt_original_pipeline_estimated.append(analysis_result_opt.original_pipeline_estimated)
        result_df_opt_original_pipeline_importing_and_monkeypatching.append(
            analysis_result_opt.original_pipeline_importing_and_monkeypatching)
        result_df_opt_original_pipeline_without_importing_and_monkeypatching.append(
            analysis_result_opt.original_pipeline_without_importing_and_monkeypatching)
        result_df_opt_original_pipeline_train_shape.append(
            analysis_result_opt.original_pipeline_train_data_shape)
        result_df_opt_original_pipeline_test_shape.append(
            analysis_result_opt.original_pipeline_test_data_shape)
        result_df_opt_original_pipeline_model_training.append(analysis_result_opt.original_model_training)
        result_df_opt_what_if_plan_generation.append(analysis_result_opt.what_if_plan_generation)
        result_df_opt_what_if_query_optimization_duration.append(
            analysis_result_opt.what_if_query_optimization_duration)
        result_df_opt_what_if_execution.append(analysis_result_opt.what_if_execution)
        result_df_opt_what_if_execution_combined_model_training.append(
            analysis_result_opt.what_if_execution_combined_model_training)
        result_df_no_opt_what_if_plan_generation.append(analysis_result_no_opt.what_if_plan_generation)
        result_df_no_opt_what_if_query_optimization_duration.append(
            analysis_result_no_opt.what_if_query_optimization_duration)
        result_df_no_opt_what_if_execution.append(analysis_result_no_opt.what_if_execution)
        result_df_no_opt_what_if_execution_combined_model_training.append(
            analysis_result_no_opt.what_if_execution_combined_model_training)

        result_df_base_memory_per_sec_psutil_rss.append(memory_info_base[0])
        result_df_base_memory_per_sec_psutil_vms.append(memory_info_base[1])
        result_df_base_memory_max_psutil_rss.append(max(memory_info_base[0]))
        result_df_no_opt_memory_per_sec_psutil_rss.append(memory_info_no_opt[0])
        result_df_no_opt_memory_per_sec_psutil_vms.append(memory_info_no_opt[1])
        result_df_no_opt_memory_max_psutil_rss.append(max(memory_info_no_opt[0]))
        result_df_opt_memory_per_sec_psutil_rss.append(memory_info_opt[0])
        result_df_opt_memory_per_sec_psutil_vms.append(memory_info_opt[1])
        result_df_opt_memory_max_psutil_rss.append(max(memory_info_opt[0]))
        result_df_memory_inbetween_0.append(memory_inbetween_0)
        result_df_memory_inbetween_1.append(memory_inbetween_1)
        result_df_memory_inbetween_2.append(memory_inbetween_2)

        logger.info(f'# Finished -- repetition {repetition + 1} of {num_repetitions} ')
        # print(results_dict_items)

    result_df = pandas.DataFrame({'repetition': result_df_repetitions,
                                  'scenario': result_df_scenarios,
                                  'dataset': result_df_dataset,
                                  'data_loading': result_df_data_loading,
                                  'featurization': result_df_featurization,
                                  'model': result_df_model,
                                  'variant_factor': result_df_variant_factor,
                                  'variant_actual_count': result_df_variant_actual_count,
                                  'total_exec_duration_with_opt': result_df_total_exec_opt,
                                  'total_exec_duration_without_opt': result_df_total_exec_no_opt,
                                  'opt_what_if_optimized_estimated':
                                      result_df_opt_what_if_optimized_estimated,
                                  'opt_what_if_unoptimized_estimated': result_df_opt_what_if_unoptimized_estimated,
                                  'opt_what_if_optimization_saving_estimated':
                                      result_df_opt_what_if_optimization_saving_estimated,
                                  'opt_original_pipeline_estimated': result_df_opt_original_pipeline_estimated,
                                  'opt_original_pipeline_importing_and_monkeypatching':
                                      result_df_opt_original_pipeline_importing_and_monkeypatching,
                                  'opt_original_pipeline_without_importing_and_monkeypatching':
                                      result_df_opt_original_pipeline_without_importing_and_monkeypatching,
                                  'opt_original_pipeline_model_training':
                                      result_df_opt_original_pipeline_model_training,
                                  'opt_original_pipeline_train_data_shape': result_df_opt_original_pipeline_train_shape,
                                  'opt_original_pipeline_test_data_shape': result_df_opt_original_pipeline_test_shape,
                                  'opt_what_if_plan_generation': result_df_opt_what_if_plan_generation,
                                  'opt_what_if_query_optimization_duration':
                                      result_df_opt_what_if_query_optimization_duration,
                                  'opt_what_if_execution': result_df_opt_what_if_execution,
                                  'opt_what_if_execution_combined_model_training':
                                      result_df_opt_what_if_execution_combined_model_training,
                                  'no_opt_what_if_plan_generation': result_df_no_opt_what_if_plan_generation,
                                  'no_opt_what_if_query_optimization_duration':
                                      result_df_no_opt_what_if_query_optimization_duration,
                                  'no_opt_what_if_execution': result_df_no_opt_what_if_execution,
                                  'no_opt_what_if_execution_combined_model_training':
                                      result_df_no_opt_what_if_execution_combined_model_training,
                                  'result_df_base_memory_per_sec_psutil_rss': result_df_base_memory_per_sec_psutil_rss,
                                  'result_df_base_memory_per_sec_psutil_vms': result_df_base_memory_per_sec_psutil_vms,
                                  'result_df_base_memory_max_psutil_rss': result_df_base_memory_max_psutil_rss,
                                  'result_df_no_opt_memory_per_sec_psutil_rss': result_df_no_opt_memory_per_sec_psutil_rss,
                                  'result_df_no_opt_memory_per_sec_psutil_vms': result_df_no_opt_memory_per_sec_psutil_vms,
                                  'result_df_no_opt_memory_max_psutil_rss': result_df_no_opt_memory_max_psutil_rss,
                                  'result_df_opt_memory_per_sec_psutil_rss': result_df_opt_memory_per_sec_psutil_rss,
                                  'result_df_opt_memory_per_sec_psutil_vms': result_df_opt_memory_per_sec_psutil_vms,
                                  'result_df_opt_memory_max_psutil_rss': result_df_opt_memory_max_psutil_rss,
                                  'result_df_memory_inbetween_0': result_df_memory_inbetween_0,
                                  'result_df_memory_inbetween_1': result_df_memory_inbetween_1,
                                  'result_df_memory_inbetween_2': result_df_memory_inbetween_2
                                  })

    result_df_path = os.path.join(output_directory, f"results-{scenario_name}-"
                                                    f"{dataset_name}-{data_loading_name}-"
                                                    f"{featurization_name}-{model_name}-{exec_strategy}-"
                                                    f"{variant_count}.csv")
    result_df.to_csv(result_df_path, index=False)
    logger.info(f'# Final results after all repetitions')
    logger.info(result_df)
