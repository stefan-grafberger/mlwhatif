"""
Tests whether the optimization works
"""
# pylint: disable=too-many-locals,invalid-name,missing-function-docstring
# TODO: Clean up these tests
import os
from functools import partial
from inspect import cleandoc

from sklearn.dummy import DummyClassifier

from experiments._benchmark_utils import get_test_df
from mlwhatif import PipelineAnalyzer
from mlwhatif.analysis._model_variants import ModelVariants


def run_common_subexpression_elimination_benchmark(scenario, variant_count, data_size, csv_dir):
    if scenario not in scenario_funcs:
        print(f"Valid scenario names: {scenario_funcs.keys()}")
        raise ValueError(f"Scenario name {scenario} is not one of them!")
    return scenario_funcs[scenario](data_size, csv_dir, variant_count)



def execute_common_subexpression_elimination_ideal_case(data_size, tmpdir, variant_count):
    df_a_train, df_b_train = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "common_subexpression_elimination_df_a_ideal_case_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_b_path_train = os.path.join(tmpdir, "common_subexpression_elimination_df_b_ideal_case_train.csv")
    df_b_train.to_csv(df_b_path_train, index=False)
    df_a_test, df_b_test = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "common_subexpression_elimination_df_a_ideal_case_test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    df_b_path_test = os.path.join(tmpdir, "common_subexpression_elimination_df_b_ideal_case_test.csv")
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
    variants = []
    for variant_index in range(variant_count):
        variants.append((f'dummy_classifier_{variant_index}',
                         partial(DummyClassifier, strategy='constant', constant=0.)))
    analysis = ModelVariants(variants)
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info
    analysis_result_with_opt = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(analysis) \
        .overwrite_optimization_rules([]) \
        .execute()
    analysis_result_without_any_opt = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(analysis) \
        .skip_multi_query_optimization(True) \
        .execute()
    return analysis, analysis_result_with_opt, analysis_result_without_any_opt


def execute_common_subexpression_elimunation_average_case(data_size, tmpdir, variant_count):
    df_a_train, df_b_train = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "common_subexpression_elimination_df_a_average_case_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_b_path_train = os.path.join(tmpdir, "common_subexpression_elimination_df_b_average_case_train.csv")
    df_b_train.to_csv(df_b_path_train, index=False)
    df_a_test, df_b_test = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "common_subexpression_elimination_df_a_average_case_test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    df_b_path_test = os.path.join(tmpdir, "common_subexpression_elimination_df_b_average_case_test.csv")
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
    variants = []
    for variant_index in range(variant_count):
        variants.append((f'dummy_classifier_{variant_index}',
                         partial(DummyClassifier, strategy='constant', constant=0.)))
    analysis = ModelVariants(variants)
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info
    analysis_result_with_opt = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(analysis) \
        .overwrite_optimization_rules([]) \
        .execute()
    analysis_result_without_any_opt = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(analysis) \
        .skip_multi_query_optimization(True) \
        .execute()
    return analysis, analysis_result_with_opt, analysis_result_without_any_opt


def execute_common_subexpression_elimination_worst_case(data_size, tmpdir, variant_count):
    df_a_train, _ = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "common_subexpression_elimination_df_a_worst_case_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_a_test, _ = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "common_subexpression_elimination_df_a_worst_case_test.csv")
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

        df_train = pd.read_csv("{df_a_path_train}")
        df_test = pd.read_csv("{df_a_path_test}")

        clf = DummyClassifier(strategy='constant', constant=0.)

        train_data = df_train[['A', 'B', 'group_col_1']]
        train_target = df_train['target_featurized']

        test_target = df_test['target_featurized']
        test_data = df_test[['A', 'B', 'group_col_1']]

        clf = clf.fit(train_data, train_target)
        test_score = clf.score(test_data, test_target)
        assert 0. <= test_score <= 1.
        """)
    variants = []
    for variant_index in range(variant_count):
        variants.append((f'dummy_classifier_{variant_index}',
                         partial(DummyClassifier, strategy='constant', constant=0.)))
    analysis = ModelVariants(variants)
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info
    analysis_result_with_opt = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(analysis) \
        .overwrite_optimization_rules([]) \
        .execute()
    analysis_result_without_any_opt = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(analysis) \
        .skip_multi_query_optimization(True) \
        .execute()
    return analysis, analysis_result_with_opt, analysis_result_without_any_opt


scenario_funcs = {
    'ideal': execute_common_subexpression_elimination_ideal_case,
    'average': execute_common_subexpression_elimunation_average_case,
    'worst': execute_common_subexpression_elimination_worst_case
}
