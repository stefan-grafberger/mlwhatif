import os
import sys
import warnings

from example_pipelines import HEALTHCARE_PY
from example_pipelines.healthcare import custom_monkeypatching
from experiments.common_subexpression_elimination import run_common_subexpression_elimination_benchmark
from experiments.operator_deletion_filter_push_up import run_operator_deletion_filter_push_up_benchmark
from experiments.simple_filter_addition_push_up import run_filter_addition_push_up_benchmark
from experiments.simple_projection_push_up import run_projection_push_up_benchmark
from experiments.udf_split_and_reuse import run_udf_split_and_reuse_benchmark
from mlwhatif import PipelineAnalyzer

warnings.filterwarnings("ignore")

if len(sys.argv) < 5:
    print("optimization scenario variant_count data_size")

optimization = sys.argv[1]
scenario = sys.argv[2]
variant_count = int(sys.argv[3])
data_size = int(sys.argv[4])


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

# Warm-up run to ignore effect of imports
_ = PipelineAnalyzer \
    .on_pipeline_from_py_file(HEALTHCARE_PY) \
    .add_custom_monkey_patching_modules([custom_monkeypatching]) \
    .execute()


current_directory = os.getcwd()
output_directory = os.path.join(current_directory, r'optimization-benchmark-results')
if not os.path.exists(output_directory):
    os.makedirs(output_directory)


for repetition in range(num_repetitions):
    print(f'# Optimization benchmark for {optimization} with args [{sys.argv[2:]}] '
          f'-- repetition {repetition + 1} of {num_repetitions}')
    if optimization not in optimizations_benchmark_funcs:
        print(f"Valid optimization types: {optimizations_benchmark_funcs.keys()}")
        raise ValueError(f"Optimization type {optimization} is not one of them!")
    benchmark_func = optimizations_benchmark_funcs[optimization]
    results = benchmark_func(scenario, variant_count, data_size, output_directory)
    print(f'# Final results -- repetition {repetition + 1} of {num_repetitions} ')
    print(results)
