"""
Tests whether the optimization works
"""
# pylint: disable=too-many-locals,invalid-name,missing-function-docstring,unused-argument
# TODO: Clean up these tests
import os
from functools import partial
from inspect import cleandoc

from experiments._benchmark_utils import get_test_df
from mlwhatif import PipelineAnalyzer
from mlwhatif.execution._pipeline_executor import singleton
from mlwhatif.optimization._simple_filter_addition_push_up import SimpleFilterAdditionPushUp
from mlwhatif.testing._data_filter_variants import DataFilterVariants


def run_filter_addition_push_up_benchmark(scenario, variant_count, data_size, csv_dir):
    if scenario not in scenario_funcs:
        print(f"Valid scenario names: {scenario_funcs.keys()}")
        raise ValueError(f"Scenario name {scenario} is not one of them!")
    return scenario_funcs[scenario](data_size, csv_dir, variant_count)


def execute_filter_addition_push_up_ideal_case(scale_factor, tmpdir, variant_count):
    data_size = int(4000 * scale_factor)
    df_a_train, df_b_train = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "filter_push_up_df_a_ideal_case_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_b_path_train = os.path.join(tmpdir, "filter_push_up_df_b_ideal_case_train.csv")
    df_b_train.to_csv(df_b_path_train, index=False)
    df_a_test, df_b_test = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "filter_push_up_df_a_ideal_case_test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    df_b_path_test = os.path.join(tmpdir, "filter_push_up_df_b_ideal_case_test.csv")
    df_b_test.to_csv(df_b_path_test, index=False)
    test_code = cleandoc(f"""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler
        from sklearn.dummy import DummyClassifier
        import numpy as np
        from sklearn.model_selection import train_test_split
        import fuzzy_pandas as fpd
        
        df_a_train = pd.read_csv("{df_a_path_train}", engine='python')
        df_a_train = df_a_train[df_a_train['A'] >= 95]
        df_b_train = pd.read_csv("{df_b_path_train}", engine='python')
        df_train = fpd.fuzzy_merge(df_a_train, df_b_train, on='str_id', method='levenshtein', keep_right=['C', 'D'],
            threshold=0.99)
        
        df_a_test = pd.read_csv("{df_a_path_test}", engine='python')
        df_a_test = df_a_test[df_a_test['A'] >= 95]
        df_b_test = pd.read_csv("{df_b_path_test}", engine='python')
        df_test = fpd.fuzzy_merge(df_a_test, df_b_test, on='str_id', method='levenshtein', keep_right=['C', 'D'],
            threshold=0.99)
        
        train_target = df_train['target_featurized']
        train_data = df_train[['A', 'B']]
        
        test_target = df_test['target_featurized']
        test_data = df_test[['A', 'B']]
        
        clf = DummyClassifier(strategy='constant', constant=0.)
        clf = clf.fit(train_data, train_target)
        test_score = clf.score(test_data, test_target)
        assert 0. <= test_score <= 1.
        """)
    filter_variants = {}
    for variant_index in range(variant_count - 1):
        def filter_func(df, bound_variant_index):
            return df[df['B'] <= (99 - bound_variant_index)]

        filter_variants[f"filter_{variant_index}"] = ("B", partial(filter_func, bound_variant_index=variant_index))
    data_corruption = DataFilterVariants(filter_variants, True)
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info
    analysis_result_with_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([SimpleFilterAdditionPushUp(singleton)]) \
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
            'analysis_result_with_opt_rule': analysis_result_with_opt_rule,
            'analysis_result_without_opt_rule': analysis_result_without_opt_rule,
            'analysis_result_without_any_opt': analysis_result_without_any_opt}


def execute_filter_addition_push_up_average_case(scale_factor, tmpdir, variant_count):
    data_size = int(250000 * scale_factor)
    df_a_train, df_b_train = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "filter_push_up_df_a_average_case_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_b_path_train = os.path.join(tmpdir, "filter_push_up_df_b_average_case_train.csv")
    df_b_train.to_csv(df_b_path_train, index=False)
    df_a_test, df_b_test = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "filter_push_up_df_a_average_case_test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    df_b_path_test = os.path.join(tmpdir, "filter_push_up_df_b_average_case_test.csv")
    df_b_test.to_csv(df_b_path_test, index=False)
    test_code = cleandoc(f"""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder
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
                    ('numeric', StandardScaler(), ['A', 'B']),
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
    filter_variants = {}
    for variant_index in range(variant_count - 1):
        def filter_func(df, bound_variant_index):
            return df[df['B'] <= (80 - bound_variant_index)]

        filter_variants[f"filter_{variant_index}"] = ("B", partial(filter_func, bound_variant_index=variant_index))
    data_corruption = DataFilterVariants(filter_variants, True)
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info
    analysis_result_with_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([SimpleFilterAdditionPushUp(singleton)]) \
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
            'analysis_result_with_opt_rule': analysis_result_with_opt_rule,
            'analysis_result_without_opt_rule': analysis_result_without_opt_rule,
            'analysis_result_without_any_opt': analysis_result_without_any_opt}


def execute_filter_addition_push_up_worst_case_no_original_pipeline(scale_factor, tmpdir, variant_count):
    data_size = int(4000 * scale_factor)
    df_a_train, df_b_train = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "filter_push_up_df_a_worst_case_no_original_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_b_path_train = os.path.join(tmpdir, "filter_push_up_df_b_worst_case_no_original_train.csv")
    df_b_train.to_csv(df_b_path_train, index=False)
    df_a_test, df_b_test = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "filter_push_up_df_a_worst_case_no_original_test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    df_b_path_test = os.path.join(tmpdir, "filter_push_up_df_b_worst_case_no_original_test.csv")
    df_b_test.to_csv(df_b_path_test, index=False)
    test_code = cleandoc(f"""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler
        from sklearn.dummy import DummyClassifier
        import numpy as np
        from sklearn.model_selection import train_test_split
        import fuzzy_pandas as fpd
        
        df_a_train = pd.read_csv("{df_a_path_train}", engine='python')
        df_a_train = df_a_train[df_a_train['A'] >= 95]
        df_b_train = pd.read_csv("{df_b_path_train}", engine='python')
        df_train = fpd.fuzzy_merge(df_a_train, df_b_train, on='str_id', method='levenshtein', keep_right=['C', 'D'],
            threshold=0.99)
        
        df_a_test = pd.read_csv("{df_a_path_test}", engine='python')
        df_a_test = df_a_test[df_a_test['A'] >= 95]
        df_b_test = pd.read_csv("{df_b_path_test}", engine='python')
        df_test = fpd.fuzzy_merge(df_a_test, df_b_test, on='str_id', method='levenshtein', keep_right=['C', 'D'],
            threshold=0.99)
        
        train_target = df_train['target_featurized']
        train_data = df_train[['A', 'B']]
        
        test_target = df_test['target_featurized']
        test_data = df_test[['A', 'B']]
        
        clf = DummyClassifier(strategy='constant', constant=0.)
        clf = clf.fit(train_data, train_target)
        test_score = clf.score(test_data, test_target)
        assert 0. <= test_score <= 1.
        """)
    filter_variants = {}
    for variant_index in range(variant_count):
        def filter_func(df, bound_variant_index):
            return df[df['B'] <= 3]

        filter_variants[f"filter_{variant_index}"] = ("B", partial(filter_func, bound_variant_index=variant_index))
    data_corruption = DataFilterVariants(filter_variants, False)
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info
    analysis_result_with_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([SimpleFilterAdditionPushUp(singleton)]) \
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
            'analysis_result_with_opt_rule': analysis_result_with_opt_rule,
            'analysis_result_without_opt_rule': analysis_result_without_opt_rule,
            'analysis_result_without_any_opt': analysis_result_without_any_opt}


def execute_filter_addition_push_up_worst_case_original_pipeline(scale_factor, tmpdir, variant_count):
    data_size = int(4000 * scale_factor)
    df_a_train, df_b_train = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "filter_push_up_df_a_worst_case_no_original_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_b_path_train = os.path.join(tmpdir, "filter_push_up_df_b_worst_case_no_original_train.csv")
    df_b_train.to_csv(df_b_path_train, index=False)
    df_a_test, df_b_test = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "filter_push_up_df_a_worst_case_no_original_test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    df_b_path_test = os.path.join(tmpdir, "filter_push_up_df_b_worst_case_no_original_test.csv")
    df_b_test.to_csv(df_b_path_test, index=False)
    test_code = cleandoc(f"""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler
        from sklearn.dummy import DummyClassifier
        import numpy as np
        from sklearn.model_selection import train_test_split
        import fuzzy_pandas as fpd

        df_a_train = pd.read_csv("{df_a_path_train}", engine='python')
        df_a_train = df_a_train[df_a_train['A'] >= 95]
        df_b_train = pd.read_csv("{df_b_path_train}", engine='python')
        df_train = fpd.fuzzy_merge(df_a_train, df_b_train, on='str_id', method='levenshtein', keep_right=['C', 'D'],
            threshold=0.99)

        df_a_test = pd.read_csv("{df_a_path_test}", engine='python')
        df_a_test = df_a_test[df_a_test['A'] >= 95]
        df_b_test = pd.read_csv("{df_b_path_test}", engine='python')
        df_test = fpd.fuzzy_merge(df_a_test, df_b_test, on='str_id', method='levenshtein', keep_right=['C', 'D'],
            threshold=0.99)

        train_target = df_train['target_featurized']
        train_data = df_train[['A', 'B']]

        test_target = df_test['target_featurized']
        test_data = df_test[['A', 'B']]

        clf = DummyClassifier(strategy='constant', constant=0.)
        clf = clf.fit(train_data, train_target)
        test_score = clf.score(test_data, test_target)
        assert 0. <= test_score <= 1.
        """)
    filter_variants = {}
    for variant_index in range(variant_count - 1):
        def filter_func(df, bound_variant_index):
            return df[df['B'] <= 3]

        filter_variants[f"filter_{variant_index}"] = ("B", partial(filter_func, bound_variant_index=variant_index))
    data_corruption = DataFilterVariants(filter_variants, True)
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info
    analysis_result_with_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([SimpleFilterAdditionPushUp(singleton)]) \
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
            'analysis_result_with_opt_rule': analysis_result_with_opt_rule,
            'analysis_result_without_opt_rule': analysis_result_without_opt_rule,
            'analysis_result_without_any_opt': analysis_result_without_any_opt}


scenario_funcs = {
    'ideal': execute_filter_addition_push_up_ideal_case,
    'average': execute_filter_addition_push_up_average_case,
    'worst_wo_original': execute_filter_addition_push_up_worst_case_no_original_pipeline,
    'worst_w_original': execute_filter_addition_push_up_worst_case_original_pipeline
}
