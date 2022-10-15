"""
Tests whether the optimization works
"""
# pylint: disable=too-many-locals,invalid-name,missing-function-docstring
# TODO: Clean up these tests
import os
from inspect import cleandoc

from experiments.optimizations._benchmark_utils import get_test_df, WhatIfWrapper
from mlmq import PipelineAnalyzer
from mlmq.analysis._data_corruption import DataCorruption, CorruptionType
from mlmq.execution._pipeline_executor import singleton
from mlmq.optimization._simple_projection_push_up import SimpleProjectionPushUp
from mlmq.optimization._udf_split_and_reuse import UdfSplitAndReuse


def run_udf_split_and_reuse_benchmark(scenario, variant_count, data_size, csv_dir):
    if scenario not in scenario_funcs:
        print(f"Valid scenario names: {scenario_funcs.keys()}")
        raise ValueError(f"Scenario name {scenario} is not one of them!")
    return scenario_funcs[scenario](data_size, csv_dir, variant_count)


def execute_udf_split_and_reuse_ideal_case(scale_factor, tmpdir, variant_count):
    data_size = int(12000 * scale_factor)
    df_a_train, df_b_train = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "udf_split_and_reuse_df_a_ideal_case_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_b_path_train = os.path.join(tmpdir, "udf_split_and_reuse_df_b_ideal_case_train.csv")
    df_b_train.to_csv(df_b_path_train, index=False)
    df_a_test, df_b_test = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "udf_split_and_reuse_df_a_ideal_case_test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    df_b_path_test = os.path.join(tmpdir, "udf_split_and_reuse_df_b_ideal_case_test.csv")
    df_b_test.to_csv(df_b_path_test, index=False)
    test_code = cleandoc(f"""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder, FunctionTransformer
        from sklearn.dummy import DummyClassifier
        import numpy as np
        from sklearn.model_selection import train_test_split
        import fuzzy_pandas as fpd
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        
        df_train = pd.read_csv("{df_a_path_train}", usecols=['A', 'B', 'group_col_1', 'target_featurized'])
        df_test = pd.read_csv("{df_a_path_test}", usecols=['A', 'B', 'group_col_1', 'target_featurized'])
        
        column_transformer = ColumnTransformer(transformers=[
                   ('numeric', FunctionTransformer(accept_sparse=True, check_inverse=False), ['A', 'B']),
                   ('cat', OneHotEncoder(sparse=True, handle_unknown='ignore'), ['group_col_1'])
               ])
        pipeline = Pipeline(steps=[
           ('column_transformer', column_transformer),
           ('learner', DummyClassifier(strategy='constant', constant=0.))
        ])

        train_data = df_train[['A', 'B', 'group_col_1']]
        train_target = df_train['target_featurized']

        test_target = df_test['target_featurized']
        test_data = df_test[['A', 'B', 'group_col_1']]

        pipeline = pipeline.fit(train_data, train_target)
        test_score = pipeline.score(test_data, test_target)
        assert 0. <= test_score <= 1.
        """)
    corruption_percentages = []
    index_filter = []
    for variant_index in range(variant_count):
        corruption_percentages.append(variant_index * (0.1 / (variant_count - 1)) + 0.9)
        index_filter.append(1 + 2 * variant_index)
    data_corruption = WhatIfWrapper(
        DataCorruption([('group_col_1', CorruptionType.BROKEN_CHARACTERS)], also_corrupt_train=True,
                       corruption_percentages=corruption_percentages),
        index_filter=index_filter)
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info
    analysis_result_with_reuse_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([SimpleProjectionPushUp(singleton), UdfSplitAndReuse(singleton)]) \
        .execute()
    analysis_result_with_pushup_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([SimpleProjectionPushUp(singleton)]) \
        .execute()
    analysis_result_without_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([]) \
        .execute()
    analysis_result_without_any_opt = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .skip_multi_query_optimization(True) \
        .execute()

    return {'analysis': data_corruption,
            'analysis_result_with_reuse_opt_rule': analysis_result_with_reuse_opt_rule,
            'analysis_result_with_push_up_opt_rule': analysis_result_with_pushup_opt_rule,
            'analysis_result_without_opt_rule': analysis_result_without_opt_rule,
            'analysis_result_without_any_opt': analysis_result_without_any_opt}


def execute_udf_split_and_reuse_average_case(scale_factor, tmpdir, variant_count):
    data_size = int(215000 * scale_factor)
    df_a_train, df_b_train = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "projection_push_up_df_a_average_case_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_b_path_train = os.path.join(tmpdir, "projection_push_up_df_b_average_case_train.csv")
    df_b_train.to_csv(df_b_path_train, index=False)
    df_a_test, df_b_test = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "projection_push_up_df_a_average_case_test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    df_b_path_test = os.path.join(tmpdir, "projection_push_up_df_b_average_case_test.csv")
    df_b_test.to_csv(df_b_path_test, index=False)
    test_code = cleandoc(f"""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder, RobustScaler
        from sklearn.dummy import DummyClassifier
        import numpy as np
        from sklearn.model_selection import train_test_split
        import fuzzy_pandas as fpd
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer

        df_a_train = pd.read_csv("{df_a_path_train}")
        df_a_train = df_a_train[df_a_train['A'] >= 15]
        df_b_train = pd.read_csv("{df_b_path_train}")
        df_train = df_a_train.merge(df_b_train, on=['str_id'])

        df_a_test = pd.read_csv("{df_a_path_test}")
        df_a_test = df_a_test[df_a_test['A'] >= 15]
        df_b_test = pd.read_csv("{df_b_path_test}")
        df_test = df_a_test.merge(df_b_test, on=['str_id'])

        column_transformer = ColumnTransformer(transformers=[
                    ('numeric', RobustScaler(), ['A', 'B']),
                    ('cat', OneHotEncoder(sparse=True, handle_unknown='ignore'), ['group_col_1'])
                ])
        pipeline = Pipeline(steps=[
            ('column_transformer', column_transformer),
            ('learner', DummyClassifier(strategy='constant', constant=0.))
        ])
        
        train_data = df_train[['A', 'B', 'group_col_1']]
        train_target = df_train['target_featurized']
        
        test_target = df_test['target_featurized']
        test_data = df_test[['A', 'B', 'group_col_1']]
        
        pipeline = pipeline.fit(train_data, train_target)
        test_score = pipeline.score(test_data, test_target)
        assert 0. <= test_score <= 1.
        """)
    corruption_percentages = []
    index_filter = []
    for variant_index in range(variant_count):
        corruption_percentages.append((variant_index + 1) * (1.0 / variant_count))
        index_filter.append(1 + 2 * variant_index)
    data_corruption = WhatIfWrapper(
        DataCorruption([('B', CorruptionType.GAUSSIAN_NOISE)], also_corrupt_train=True,
                       corruption_percentages=corruption_percentages),
        index_filter=index_filter)
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info
    analysis_result_with_reuse_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([SimpleProjectionPushUp(singleton), UdfSplitAndReuse(singleton)]) \
        .execute()
    analysis_result_with_pushup_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([SimpleProjectionPushUp(singleton)]) \
        .execute()
    analysis_result_without_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([]) \
        .execute()
    analysis_result_without_any_opt = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .skip_multi_query_optimization(True) \
        .execute()
    return {'analysis': data_corruption,
            'analysis_result_with_reuse_opt_rule': analysis_result_with_reuse_opt_rule,
            'analysis_result_with_push_up_opt_rule': analysis_result_with_pushup_opt_rule,
            'analysis_result_without_opt_rule': analysis_result_without_opt_rule,
            'analysis_result_without_any_opt': analysis_result_without_any_opt}


def execute_udf_split_and_reuse_worst_case_with_selectivity_safety_active(scale_factor, tmpdir, variant_count):
    data_size = int(60000 * scale_factor)
    df_a_train, _ = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "udf_split_and_reuse_df_a_worst_case_with_safety_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_a_test, _ = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "udf_split_and_reuse_df_a_worst_case_with_safety_test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    test_code = cleandoc(f"""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder, FunctionTransformer
        from sklearn.dummy import DummyClassifier
        import numpy as np
        from sklearn.model_selection import train_test_split
        import fuzzy_pandas as fpd
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        
        df_train = pd.read_csv("{df_a_path_train}", usecols=['A', 'B', 'group_col_1', 'target_featurized'])
        df_test = pd.read_csv("{df_a_path_test}", usecols=['A', 'B', 'group_col_1', 'target_featurized'])
        
        column_transformer = ColumnTransformer(transformers=[
                   ('numeric', FunctionTransformer(accept_sparse=True, check_inverse=False), ['A', 'B']),
                   ('cat', OneHotEncoder(sparse=True, handle_unknown='ignore'), ['group_col_1'])
               ])
        pipeline = Pipeline(steps=[
           ('column_transformer', column_transformer),
           ('learner', DummyClassifier(strategy='constant', constant=0.))
        ])

        train_data = df_train[['A', 'B', 'group_col_1']]
        train_target = df_train['target_featurized']

        test_target = df_test['target_featurized']
        test_data = df_test[['A', 'B', 'group_col_1']]

        pipeline = pipeline.fit(train_data, train_target)
        test_score = pipeline.score(test_data, test_target)
        assert 0. <= test_score <= 1.
        """)
    corruption_percentages = []
    index_filter = []
    for variant_index in range(variant_count):
        corruption_percentages.append(variant_index * (0.1 / (variant_count - 1)))
        index_filter.append(1 + 2 * variant_index)
    data_corruption = WhatIfWrapper(
        DataCorruption([('A', CorruptionType.BROKEN_CHARACTERS)], also_corrupt_train=True,
                       corruption_percentages=corruption_percentages),
        index_filter=index_filter)
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info
    analysis_result_with_reuse_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([SimpleProjectionPushUp(singleton), UdfSplitAndReuse(singleton)]) \
        .execute()
    analysis_result_with_pushup_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([SimpleProjectionPushUp(singleton)]) \
        .execute()
    analysis_result_without_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([]) \
        .execute()
    analysis_result_without_any_opt = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .skip_multi_query_optimization(True) \
        .execute()
    return {'analysis': data_corruption,
            'analysis_result_with_reuse_opt_rule': analysis_result_with_reuse_opt_rule,
            'analysis_result_with_push_up_opt_rule': analysis_result_with_pushup_opt_rule,
            'analysis_result_without_opt_rule': analysis_result_without_opt_rule,
            'analysis_result_without_any_opt': analysis_result_without_any_opt}


def execute_udf_split_and_reuse_worst_case_with_selectivity_inactive(scale_factor, tmpdir, variant_count):
    data_size = int(13000 * scale_factor)
    df_a_train, _ = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "udf_split_and_reuse_df_a_worst_case_without_safety_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_a_test, _ = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "udf_split_and_reuse_df_a_worst_case_without_safety_test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    test_code = cleandoc(f"""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder, FunctionTransformer
        from sklearn.dummy import DummyClassifier
        import numpy as np
        from sklearn.model_selection import train_test_split
        import fuzzy_pandas as fpd
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        
        df_train = pd.read_csv("{df_a_path_train}", usecols=['A', 'B', 'group_col_1', 'target_featurized'])
        df_test = pd.read_csv("{df_a_path_test}", usecols=['A', 'B', 'group_col_1', 'target_featurized'])
        
        column_transformer = ColumnTransformer(transformers=[
                   ('numeric', FunctionTransformer(accept_sparse=True, check_inverse=False), ['A', 'B']),
                   ('cat', OneHotEncoder(sparse=True, handle_unknown='ignore'), ['group_col_1'])
               ])
        pipeline = Pipeline(steps=[
           ('column_transformer', column_transformer),
           ('learner', DummyClassifier(strategy='constant', constant=0.))
        ])

        train_data = df_train[['A', 'B', 'group_col_1']]
        train_target = df_train['target_featurized']

        test_target = df_test['target_featurized']
        test_data = df_test[['A', 'B', 'group_col_1']]

        pipeline = pipeline.fit(train_data, train_target)
        test_score = pipeline.score(test_data, test_target)
        assert 0. <= test_score <= 1.
        """)
    corruption_percentages = []
    index_filter = []
    for variant_index in range(variant_count):
        corruption_percentages.append(variant_index * (0.1 / (variant_count - 1)))
        index_filter.append(1 + 2 * variant_index)
    data_corruption = WhatIfWrapper(
        DataCorruption([('A', CorruptionType.BROKEN_CHARACTERS)], also_corrupt_train=True,
                       corruption_percentages=corruption_percentages),
        index_filter=index_filter)
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info
    analysis_result_with_reuse_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([SimpleProjectionPushUp(singleton),
                                       UdfSplitAndReuse(singleton, disable_selectivity_safety=True)]) \
        .execute()
    analysis_result_with_pushup_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([SimpleProjectionPushUp(singleton)]) \
        .execute()
    analysis_result_without_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([]) \
        .execute()
    analysis_result_without_any_opt = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .skip_multi_query_optimization(True) \
        .execute()
    return {'analysis': data_corruption,
            'analysis_result_with_reuse_opt_rule': analysis_result_with_reuse_opt_rule,
            'analysis_result_with_push_up_opt_rule': analysis_result_with_pushup_opt_rule,
            'analysis_result_without_opt_rule': analysis_result_without_opt_rule,
            'analysis_result_without_any_opt': analysis_result_without_any_opt}


def execute_udf_split_and_reuse_worst_case_with_constant(scale_factor, tmpdir, variant_count):
    data_size = int(1050000 * scale_factor)
    df_a_train, _ = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "udf_split_and_reuse_df_a_worst_case_constant_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_a_test, _ = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "udf_split_and_reuse_df_a_worst_case_constant_test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    test_code = cleandoc(f"""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder
        from sklearn.dummy import DummyClassifier
        import numpy as np
        from sklearn.model_selection import train_test_split
        import fuzzy_pandas as fpd
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        
        df_train = pd.read_csv("{df_a_path_train}", usecols=['A', 'B', 'target_featurized'])
        df_test = pd.read_csv("{df_a_path_test}", usecols=['A', 'B', 'target_featurized'])
        
        clf = DummyClassifier(strategy='constant', constant=0.)

        train_data = df_train[['A', 'B']]
        train_target = df_train['target_featurized']

        test_target = df_test['target_featurized']
        test_data = df_test[['A', 'B']]

        clf = clf.fit(train_data, train_target)
        test_score = clf.score(test_data, test_target)
        assert 0. <= test_score <= 1.
        """)
    corruption_percentages = []
    index_filter = []
    for variant_index in range(variant_count):
        # set is used, so they need to be distinct, but the difference should be small enough not to make a
        #  difference at all
        corruption_percentages.append((1.0 / variant_count) - variant_index * 0.00000001)
        index_filter.append(1 + 2 * variant_index)

    def corruption(pandas_df):
        pandas_df['B'] = 0
        return pandas_df

    data_corruption = WhatIfWrapper(
        DataCorruption([('B', corruption)], also_corrupt_train=True,
                       corruption_percentages=corruption_percentages),
        index_filter=index_filter)
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info
    analysis_result_with_reuse_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([SimpleProjectionPushUp(singleton), UdfSplitAndReuse(singleton)]) \
        .execute()
    analysis_result_with_pushup_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([SimpleProjectionPushUp(singleton)]) \
        .execute()
    analysis_result_without_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([]) \
        .execute()
    analysis_result_without_any_opt = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .skip_multi_query_optimization(True) \
        .execute()
    return {'analysis': data_corruption,
            'analysis_result_with_reuse_opt_rule': analysis_result_with_reuse_opt_rule,
            'analysis_result_with_push_up_opt_rule': analysis_result_with_pushup_opt_rule,
            'analysis_result_without_opt_rule': analysis_result_without_opt_rule,
            'analysis_result_without_any_opt': analysis_result_without_any_opt}


scenario_funcs = {
    'ideal': execute_udf_split_and_reuse_ideal_case,
    'average': execute_udf_split_and_reuse_average_case,
    'worst_w_safety': execute_udf_split_and_reuse_worst_case_with_selectivity_safety_active,
    'worst_wo_safety': execute_udf_split_and_reuse_worst_case_with_selectivity_inactive,
    'worst_constant': execute_udf_split_and_reuse_worst_case_with_constant
}
