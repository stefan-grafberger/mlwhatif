# pylint: disable=invalid-name,missing-module-docstring
import os
import random
import sys
import tempfile
import warnings

import numpy
import pandas

from example_pipelines import HEALTHCARE_PY
from example_pipelines.healthcare import custom_monkeypatching
from experiments.optimizations.common_subexpression_elimination import run_common_subexpression_elimination_benchmark
from experiments.optimizations.operator_deletion_filter_push_up import run_operator_deletion_filter_push_up_benchmark
from experiments.optimizations.simple_filter_addition_push_up import run_filter_addition_push_up_benchmark
from experiments.optimizations.simple_projection_push_up import run_projection_push_up_benchmark
from experiments.optimizations.udf_split_and_reuse import run_udf_split_and_reuse_benchmark
from mlwhatif import PipelineAnalyzer, AnalysisResults

# Make sure this code is not executed during imports
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    if len(sys.argv) < 5:
        print("optimization scenario variant_count scale_factor")

    optimization = sys.argv[1]
    scenario = sys.argv[2]
    variant_count = int(sys.argv[3])
    scale_factor = float(sys.argv[4])

    cmd_args = []
    if len(sys.argv) > 5:
        cmd_args = sys.argv[5:]

    optimizations_benchmark_funcs = {
        'common_subexpression_elimination': run_common_subexpression_elimination_benchmark,
        'operator_deletion_filter_push_up': run_operator_deletion_filter_push_up_benchmark,
        'filter_addition_push_up': run_filter_addition_push_up_benchmark,
        'projection_push_up': run_projection_push_up_benchmark,
        'udf_split_and_reuse': run_udf_split_and_reuse_benchmark,
    }

    print('optimization, scenario, variant_count, data_size')

    num_repetitions = 7
    seeds = [42, 43, 44, 45, 46, 47, 48]
    assert len(seeds) == num_repetitions

    # Warm-up run to ignore effect of imports
    _ = PipelineAnalyzer \
        .on_pipeline_from_py_file(HEALTHCARE_PY) \
        .add_custom_monkey_patching_modules([custom_monkeypatching]) \
        .execute()

    current_directory = os.getcwd()
    output_directory = os.path.join(current_directory, r'optimization-benchmark-results')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    result_df_repetitions = []
    result_df_optimizations = []
    result_df_scenarios = []
    result_df_variant_counts = []
    result_df_scale_factors = []
    result_df_original_pipeline = []
    result_df_metrics = {}
    for repetition, seed in enumerate(seeds):
        numpy.random.seed(seed)
        random.seed(seed)
        print(f'# Optimization benchmark for {optimization} with args [{sys.argv[2:]}] '
              f'-- repetition {repetition + 1} of {num_repetitions}')
        if optimization not in optimizations_benchmark_funcs:
            print(f"Valid optimization types: {optimizations_benchmark_funcs.keys()}")
            raise ValueError(f"Optimization type {optimization} is not one of them!")
        benchmark_func = optimizations_benchmark_funcs[optimization]
        with tempfile.TemporaryDirectory() as tmp_dir:
            results_dict_items = benchmark_func(scenario, variant_count, scale_factor, tmp_dir).items()
        results_dict_items = [(result_key, result_value)
                              for (result_key, result_value) in results_dict_items
                              if isinstance(result_value, AnalysisResults)]
        results_dict_metrics = [(result_key, result_value.runtime_info.what_if_execution)
                                for (result_key, result_value) in results_dict_items
                                if isinstance(result_value, AnalysisResults)]
        for result_key, result_value in results_dict_metrics:
            metric_value_list = result_df_metrics.get(result_key, [])
            metric_value_list.append(result_value)
            result_df_metrics[result_key] = metric_value_list
        result_df_repetitions.append(repetition)
        result_df_optimizations.append(optimization)
        result_df_scenarios.append(scenario)
        result_df_variant_counts.append(variant_count)
        result_df_scale_factors.append(scale_factor)
        result_df_original_pipeline.append(results_dict_items[0][1].runtime_info.original_pipeline_estimated)

        print(f'# Finished -- repetition {repetition + 1} of {num_repetitions} ')
        # print(results_dict_items)

    result_df = pandas.DataFrame({'repetition': result_df_repetitions,
                                  'optimization': result_df_optimizations,
                                  'scenario': result_df_scenarios,
                                  'variant_counts': result_df_variant_counts,
                                  'scale_factors': result_df_scale_factors,
                                  'original_pipeline': result_df_original_pipeline,
                                  **result_df_metrics})
    result_df_path = os.path.join(output_directory, f"results-{optimization}-{scenario}-{variant_count}-{scale_factor}.csv")
    result_df.to_csv(result_df_path, index=False)
    print(f'# Final results after all repetitions')
    print(result_df)
