# pylint: disable=invalid-name,missing-module-docstring
import logging
import os
import random
import sys
import time
import warnings
from unittest.mock import patch

import numpy
import pandas
from imgaug.augmenters import GaussianBlur
from jenga.corruptions.image import GaussianNoiseCorruption

from example_pipelines.healthcare import custom_monkeypatching
from mlwhatif import PipelineAnalyzer
# Make sure this code is not executed during imports
from mlwhatif.analysis._data_cleaning import DataCleaning, ErrorType
from mlwhatif.analysis._data_corruption import DataCorruption, CorruptionType
from mlwhatif.analysis._operator_impact import OperatorImpact
from mlwhatif.analysis._permutation_feature_importance import PermutationFeatureImportance
from mlwhatif.utils import get_project_root

logger = logging.getLogger(__name__)


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
        analysis = OperatorImpact(test_selections=True)
    elif scenario_name == 'data_corruption' and dataset_name == 'healthcare':
        def corruption(pandas_df):
            df_copy = pandas_df.copy()
            df_copy['num_children'] = 0
            return df_copy

        analysis = DataCorruption({'income': CorruptionType.SCALING,
                                   'num_children': corruption,
                                   'last_name': CorruptionType.BROKEN_CHARACTERS,
                                   'smoker': CorruptionType.CATEGORICAL_SHIFT,
                                   'county': CorruptionType.CATEGORICAL_SHIFT,
                                   'race': CorruptionType.CATEGORICAL_SHIFT
                                   },
                                  corruption_percentages=[0.2, 0.4, 0.6, 0.8, 1.0])
    elif scenario_name == 'feature_importance' and dataset_name == 'healthcare':
        analysis = PermutationFeatureImportance()
    elif scenario_name == 'data_cleaning' and dataset_name == 'healthcare':
        analysis = DataCleaning({'smoker': ErrorType.CAT_MISSING_VALUES,
                                 'county': ErrorType.CAT_MISSING_VALUES,
                                 'race': ErrorType.CAT_MISSING_VALUES,
                                 'num_children': ErrorType.NUM_MISSING_VALUES,
                                 'income': ErrorType.OUTLIERS,
                                 'full_name': ErrorType.DUPLICATES,
                                 None: ErrorType.MISLABEL})
    elif scenario_name == 'operator_impact' and dataset_name == 'healthcare':
        analysis = OperatorImpact(test_selections=True)
    elif scenario_name == 'data_corruption' and dataset_name == 'folktables':
        analysis = DataCorruption({'AGEP': CorruptionType.SCALING,
                                   'WKHP': CorruptionType.GAUSSIAN_NOISE,
                                   'COW': CorruptionType.CATEGORICAL_SHIFT,
                                   'SCHL': CorruptionType.BROKEN_CHARACTERS,
                                   'MAR': CorruptionType.CATEGORICAL_SHIFT,
                                   'OCCP': CorruptionType.MISSING_VALUES,
                                   'POBP': CorruptionType.MISSING_VALUES,
                                   'RELP': CorruptionType.MISSING_VALUES},
                                  corruption_percentages=[0.25, 0.5, 0.75, 1.0])
    elif scenario_name == 'feature_importance' and dataset_name == 'folktables':
        analysis = PermutationFeatureImportance()
    elif scenario_name == 'data_cleaning' and dataset_name == 'folktables':
        analysis = DataCleaning({'OCCP': ErrorType.CAT_MISSING_VALUES,
                                 'POBP': ErrorType.CAT_MISSING_VALUES,
                                 'RELP': ErrorType.CAT_MISSING_VALUES,
                                 'AGEP': ErrorType.OUTLIERS,
                                 'WKHP': ErrorType.NUM_MISSING_VALUES,
                                 'COW': ErrorType.CAT_MISSING_VALUES,
                                 'MAR': ErrorType.CAT_MISSING_VALUES,
                                 'SCHL': ErrorType.CAT_MISSING_VALUES,
                                 None: ErrorType.MISLABEL})
    elif scenario_name == 'operator_impact' and dataset_name == 'folktables':
        analysis = OperatorImpact(test_selections=True)
    elif scenario_name == 'data_corruption' and dataset_name == 'cardio':
        analysis = DataCorruption({'age': CorruptionType.SCALING,
                                   'height': CorruptionType.SCALING,
                                   'weight': CorruptionType.GAUSSIAN_NOISE,
                                   'ap_hi': CorruptionType.GAUSSIAN_NOISE,
                                   'ap_lo': CorruptionType.GAUSSIAN_NOISE,
                                   'gender': CorruptionType.MISSING_VALUES,
                                   'cholesterol': CorruptionType.BROKEN_CHARACTERS,
                                   'gluc': CorruptionType.MISSING_VALUES,
                                   'smoke': CorruptionType.MISSING_VALUES,
                                   'alco': CorruptionType.MISSING_VALUES,
                                   'active': CorruptionType.MISSING_VALUES})
    elif scenario_name == 'feature_importance' and dataset_name == 'cardio':
        analysis = PermutationFeatureImportance()
    elif scenario_name == 'data_cleaning' and dataset_name == 'cardio':
        analysis = DataCleaning({'smoke': ErrorType.CAT_MISSING_VALUES,
                                 'alco': ErrorType.CAT_MISSING_VALUES,
                                 'active': ErrorType.CAT_MISSING_VALUES,
                                 'height': ErrorType.OUTLIERS,
                                 'weight': ErrorType.OUTLIERS,
                                 'age': ErrorType.NUM_MISSING_VALUES,
                                 'gender': ErrorType.CAT_MISSING_VALUES,
                                 'cholesterol': ErrorType.CAT_MISSING_VALUES,
                                 'gluc': ErrorType.CAT_MISSING_VALUES,
                                 None: ErrorType.MISLABEL})
    elif scenario_name == 'operator_impact' and dataset_name == 'cardio':
        analysis = OperatorImpact(test_selections=True)
    elif scenario_name == 'data_corruption' and dataset_name == 'sneakers':
        def corruption(pandas_df):
            df_copy = pandas_df.copy()
            image_count = df_copy.shape[0]
            image_np_array = numpy.concatenate(df_copy['image'].values).reshape(image_count, 28, 28) \
                .astype(numpy.uint8)
            # corrupter = GaussianNoise(severity=3)
            corrupter = GaussianNoiseCorruption(fraction=1., severity=3)
            image_np_array = corrupter.transform(image_np_array)
            df_copy['image'] = list(image_np_array.reshape(image_count, -1))
            return df_copy

        analysis = DataCorruption({'image': corruption})
    elif scenario_name == 'data_cleaning' and dataset_name == 'sneakers':
        analysis = DataCleaning({None: ErrorType.MISLABEL})
    else:
        raise ValueError(f"Invalid scenario or dataset: {scenario_name} {dataset_name}!")
    return analysis


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    if len(sys.argv) < 6:
        logger.info("scenario_name dataset_name data_loading_name featurization_name model_name")

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

    logging.info(f'Patching sys.argv with {synthetic_cmd_args}')
    current_directory = os.getcwd()
    output_directory = os.path.join(current_directory, r'end-to-end-benchmark-results')
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
    result_df_scenarios = []
    result_df_dataset = []
    result_df_data_loading = []
    result_df_featurization = []
    result_df_model = []
    result_df_total_exec_opt = []
    result_df_total_exec_no_opt = []
    result_df_variant_counts = []

    result_df_opt_what_if_optimized_estimated = []
    result_df_opt_what_if_unoptimized_estimated = []
    result_df_opt_what_if_optimization_saving_estimated = []
    result_df_opt_original_pipeline_estimated = []
    result_df_opt_original_pipeline_importing_and_monkeypatching = []
    result_df_opt_original_pipeline_without_importing_and_monkeypatching = []
    result_df_opt_original_pipeline_model_training = []
    result_df_opt_what_if_plan_generation = []
    result_df_opt_what_if_query_optimization_duration = []
    result_df_opt_what_if_execution = []
    result_df_opt_what_if_execution_combined_model_training = []
    result_df_no_opt_what_if_plan_generation = []
    result_df_no_opt_what_if_query_optimization_duration = []
    result_df_no_opt_what_if_execution = []
    result_df_no_opt_what_if_execution_combined_model_training = []

    for repetition, seed in enumerate(seeds):
        numpy.random.seed(seed)
        random.seed(seed)
        logger.info(f'# Optimization benchmark for {scenario_name} with args [{sys.argv[2:]}] '
                    f'-- repetition {repetition + 1} of {num_repetitions}')

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
        result_df_variant_counts.append(len(analysis_result_no_opt.what_if_dags))

        result_df_opt_what_if_optimized_estimated.append(analysis_result_opt.runtime_info.what_if_optimized_estimated)
        result_df_opt_what_if_unoptimized_estimated.append(
            analysis_result_opt.runtime_info.what_if_unoptimized_estimated)
        result_df_opt_what_if_optimization_saving_estimated.append(
            analysis_result_opt.runtime_info.what_if_optimization_saving_estimated)
        result_df_opt_original_pipeline_estimated.append(analysis_result_opt.runtime_info.original_pipeline_estimated)
        result_df_opt_original_pipeline_importing_and_monkeypatching.append(
            analysis_result_opt.runtime_info.original_pipeline_importing_and_monkeypatching)
        result_df_opt_original_pipeline_without_importing_and_monkeypatching.append(
            analysis_result_opt.runtime_info.original_pipeline_without_importing_and_monkeypatching)
        result_df_opt_original_pipeline_model_training.append(analysis_result_opt.runtime_info.original_model_training)
        result_df_opt_what_if_plan_generation.append(analysis_result_opt.runtime_info.what_if_plan_generation)
        result_df_opt_what_if_query_optimization_duration.append(
            analysis_result_opt.runtime_info.what_if_query_optimization_duration)
        result_df_opt_what_if_execution.append(analysis_result_opt.runtime_info.what_if_execution)
        result_df_opt_what_if_execution_combined_model_training.append(
            analysis_result_opt.runtime_info.what_if_execution_combined_model_training)
        result_df_no_opt_what_if_plan_generation.append(analysis_result_no_opt.runtime_info.what_if_plan_generation)
        result_df_no_opt_what_if_query_optimization_duration.append(
            analysis_result_no_opt.runtime_info.what_if_query_optimization_duration)
        result_df_no_opt_what_if_execution.append(analysis_result_no_opt.runtime_info.what_if_execution)
        result_df_no_opt_what_if_execution_combined_model_training.append(
            analysis_result_no_opt.runtime_info.what_if_execution_combined_model_training)

        logger.info(f'# Finished -- repetition {repetition + 1} of {num_repetitions} ')
        # print(results_dict_items)

    result_df = pandas.DataFrame({'repetition': result_df_repetitions,
                                  'scenario': result_df_scenarios,
                                  'dataset': result_df_dataset,
                                  'data_loading': result_df_data_loading,
                                  'featurization': result_df_featurization,
                                  'model': result_df_model,
                                  'variant_count': result_df_variant_counts,
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
                                      result_df_no_opt_what_if_execution_combined_model_training
                                  })
    result_df_path = os.path.join(output_directory, f"results-{scenario_name}-"
                                                    f"{dataset_name}-{data_loading_name}-"
                                                    f"{featurization_name}-{model_name}.csv")
    result_df.to_csv(result_df_path, index=False)
    logger.info(f'# Final results after all repetitions')
    logger.info(result_df)
