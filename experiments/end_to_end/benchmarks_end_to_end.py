# pylint: disable=invalid-name,missing-module-docstring
import os
import random
import sys
import time
import warnings
from unittest.mock import patch

import numpy
import pandas

from example_pipelines.healthcare import custom_monkeypatching
from mlwhatif import PipelineAnalyzer
# Make sure this code is not executed during imports
from mlwhatif.analysis._data_cleaning import DataCleaning, ErrorType
from mlwhatif.analysis._data_corruption import DataCorruption, CorruptionType
from mlwhatif.analysis._operator_impact import OperatorImpact
from mlwhatif.analysis._permutation_feature_importance import PermutationFeatureImportance
from mlwhatif.utils import get_project_root


def get_analysis_for_scenario_and_dataset(scenario_name, dataset_name):
    if scenario_name == 'data_corruption' and dataset_name == 'reviews':
        analysis = DataCorruption({'vine': CorruptionType.CATEGORICAL_SHIFT,
                                   'review_body': CorruptionType.BROKEN_CHARACTERS,
                                   'category': CorruptionType.CATEGORICAL_SHIFT,
                                   'total_votes': CorruptionType.SCALING,
                                   'star_rating': CorruptionType.GAUSSIAN_NOISE},
                                  corruption_percentages=[0.2, 0.4, 0.6, 0.8, 1.0])
    elif scenario_name == 'feature_importance' and dataset_name == 'reviews':
        analysis = PermutationFeatureImportance()
    elif scenario_name == 'data_cleaning' and dataset_name == 'reviews':
        analysis = DataCleaning({'category': ErrorType.CAT_MISSING_VALUES,
                                 'vine': ErrorType.CAT_MISSING_VALUES,
                                 'star_rating': ErrorType.NUM_MISSING_VALUES,
                                 'total_votes': ErrorType.OUTLIERS,
                                 'review_id': ErrorType.DUPLICATES,
                                 None: ErrorType.MISLABEL})
    elif scenario_name == 'operator_impact' and dataset_name == 'reviews':
        analysis = OperatorImpact(test_selections=True, restrict_to_linenos=[46, 52, 95])
    else:
        raise ValueError(f"Invalid scenario or dataset: {scenario_name} {dataset_name}!")
    return analysis


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    if len(sys.argv) < 6:
        print("scenario_name dataset_name data_loading_name featurization_name model_name")

    scenario_name = sys.argv[1]
    dataset_name = sys.argv[2]
    data_loading_name = sys.argv[3]
    featurization_name = sys.argv[4]
    model_name = sys.argv[5]

    num_repetitions = 7
    seeds = [42, 43, 44, 45, 46, 47, 48]
    assert len(seeds) == num_repetitions

    pipeline_run_file = os.path.join(str(get_project_root()), "experiments", "end_to_end", "run_pipeline.py")
    # Warm-up run to ignore effect of imports
    synthetic_cmd_args = ['mlwhatif']
    cmd_args = sys.argv[2:].copy()
    synthetic_cmd_args.extend(cmd_args)

    analysis = get_analysis_for_scenario_and_dataset(scenario_name, dataset_name)

    print(f'Patching sys.argv with {synthetic_cmd_args}')
    current_directory = os.getcwd()
    output_directory = os.path.join(current_directory, r'end-to-end-benchmark-results')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with patch.object(sys, 'argv', synthetic_cmd_args):
        analysis_result = PipelineAnalyzer \
            .on_pipeline_from_py_file(pipeline_run_file) \
            .add_custom_monkey_patching_modules([custom_monkeypatching]) \
            .execute()
        analysis_result.save_original_dag_to_path(os.path.join(output_directory, "test"))
        # FIXME: Remove this later

    result_df_repetitions = []
    result_df_scenarios = []
    result_df_dataset = []
    result_df_data_loading = []
    result_df_featurization = []
    result_df_model = []
    result_df_total_exec_opt = []
    result_df_total_exec_no_opt = []
    for repetition, seed in enumerate(seeds):
        numpy.random.seed(seed)
        random.seed(seed)
        print(f'# Optimization benchmark for {scenario_name} with args [{sys.argv[2:]}] '
              f'-- repetition {repetition + 1} of {num_repetitions}')

        # FIXME: We still need to get the what-if analyses and specify it

        total_opt_exec_start = time.time()
        with patch.object(sys, 'argv', synthetic_cmd_args):
            analysis_result_opt = PipelineAnalyzer \
                .on_pipeline_from_py_file(pipeline_run_file) \
                .add_custom_monkey_patching_modules([custom_monkeypatching]) \
                .skip_multi_query_optimization(False) \
                .add_what_if_analysis(analysis) \
                .execute()
        total_opt_exec_duration = (time.time() - total_opt_exec_start) * 1000

        total_no_opt_exec_start = time.time()
        with patch.object(sys, 'argv', synthetic_cmd_args):
            analysis_result_no_opt = PipelineAnalyzer \
                .on_pipeline_from_py_file(pipeline_run_file) \
                .add_custom_monkey_patching_modules([custom_monkeypatching]) \
                .skip_multi_query_optimization(True) \
                .add_what_if_analysis(analysis) \
                .execute()
        total_no_opt_exec_duration = (time.time() - total_no_opt_exec_start) * 1000

        result_df_repetitions.append(repetition)
        result_df_scenarios.append(scenario_name)
        result_df_dataset.append(dataset_name)
        result_df_data_loading.append(data_loading_name)
        result_df_featurization.append(featurization_name)
        result_df_model.append(model_name)
        result_df_total_exec_opt.append(total_opt_exec_duration)
        result_df_total_exec_no_opt.append(total_no_opt_exec_duration)

        print(f'# Finished -- repetition {repetition + 1} of {num_repetitions} ')
        # print(results_dict_items)

    result_df = pandas.DataFrame({'repetition': result_df_repetitions,
                                  'scenario': result_df_scenarios,
                                  'dataset': result_df_dataset,
                                  'data_loading': result_df_data_loading,
                                  'featurization': result_df_featurization,
                                  'model': result_df_model,
                                  'exec_duration_with_opt': result_df_total_exec_opt,
                                  'exec_duration_without_opt': result_df_total_exec_no_opt,
                                  })
    result_df_path = os.path.join(output_directory, f"results-{scenario_name}-"
                                                    f"{dataset_name}-{data_loading_name}-"
                                                    f"{featurization_name}-{model_name}.csv")
    result_df.to_csv(result_df_path, index=False)
    print(f'# Final results after all repetitions')
    print(result_df)
