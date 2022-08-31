"""
Tests whether the optimization works
"""
# pylint: disable=too-many-locals,invalid-name
# TODO: Clean up these tests
import os
from inspect import cleandoc

from experiments._benchmark_utils import get_test_df
from mlwhatif import PipelineAnalyzer
from mlwhatif.analysis._operator_fairness import OperatorFairness
from mlwhatif.execution._pipeline_executor import singleton
from mlwhatif.optimization._operator_deletion_filter_push_up import OperatorDeletionFilterPushUp


def execute_operator_deletion_filter_push_up_ideal_case(data_size, tmpdir, variant_count):
    df_a_train, df_b_train = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_a_ideal_case_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_b_path_train = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_b_ideal_case_train.csv")
    df_b_train.to_csv(df_b_path_train, index=False)
    df_a_test, df_b_test = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_a_ideal_case_test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    df_b_path_test = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_b_ideal_case_test.csv")
    df_b_test.to_csv(df_b_path_test, index=False)
    filter_lines_train = []
    filter_lines_test = []
    for variant_index in range(variant_count):
        filter_lines_train.append(f"df_a_train = df_a_train[df_a_train['A'] != {95 - variant_index}]")
        filter_lines_test.append(f"df_a_test = df_a_test[df_a_test['A'] != {95 - variant_index}]")
    filter_line_train = '\n        '.join(filter_lines_train)
    filter_line_test = '\n        '.join(filter_lines_test)
    test_code = cleandoc(f"""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler
        from sklearn.dummy import DummyClassifier
        import numpy as np
        from sklearn.model_selection import train_test_split
        import fuzzy_pandas as fpd
        
        df_a_train = pd.read_csv("{df_a_path_train}", engine='python')
        {filter_line_train}
        df_b_train = pd.read_csv("{df_b_path_train}", engine='python')
        df_train = fpd.fuzzy_merge(df_a_train, df_b_train, on='str_id', method='levenshtein', keep_right=['C', 'D'],
            threshold=0.99)
        
        df_a_test = pd.read_csv("{df_a_path_test}", engine='python')
        {filter_line_test}
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
    data_corruption = OperatorFairness(test_transformers=False, test_selections=True)
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info
    analysis_result_with_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([OperatorDeletionFilterPushUp(singleton)]) \
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
    return analysis_result_with_opt_rule, analysis_result_without_any_opt, analysis_result_without_opt_rule, \
        data_corruption


def execute_operator_deletion_filter_push_up_worst_case_safety_inactive(data_size, tmpdir, variant_count):
    df_a_train, df_b_train = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_a_worst_case_safety_inactive_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_b_path_train = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_b_worst_case_safety_inactive_train.csv")
    df_b_train.to_csv(df_b_path_train, index=False)
    df_a_test, df_b_test = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_a_worst_case_safety_inactive_test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    df_b_path_test = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_b_worst_case_safety_inactive_test.csv")
    df_b_test.to_csv(df_b_path_test, index=False)
    filter_lines_train = []
    filter_lines_test = []
    for variant_index in range(variant_count):
        filter_lines_train.append(f"df_a_train = df_a_train[df_a_train['A'] >= {99 - variant_index}]")
        filter_lines_test.append(f"df_a_test = df_a_test[df_a_test['A'] >= {99 - variant_index}]")
    filter_line_train = '\n        '.join(filter_lines_train)
    filter_line_test = '\n        '.join(filter_lines_test)
    test_code = cleandoc(f"""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler
        from sklearn.dummy import DummyClassifier
        import numpy as np
        from sklearn.model_selection import train_test_split
        import fuzzy_pandas as fpd

        df_a_train = pd.read_csv("{df_a_path_train}")
        {filter_line_train}
        df_b_train = pd.read_csv("{df_b_path_train}")
        df_train = fpd.fuzzy_merge(df_a_train, df_b_train, on='str_id', method='levenshtein', keep_right=['C', 'D'],
            threshold=0.99)

        df_a_test = pd.read_csv("{df_a_path_test}")
        {filter_line_test}
        df_b_test = pd.read_csv("{df_b_path_test}")
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
    data_corruption = OperatorFairness(test_transformers=False, test_selections=True)
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info
    analysis_result_with_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([OperatorDeletionFilterPushUp(singleton, disable_selectivity_safety=True)]) \
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
    return analysis_result_with_opt_rule, analysis_result_without_any_opt, analysis_result_without_opt_rule, \
        data_corruption


def execute_operator_deletion_filter_push_up_worst_case_safety_active(data_size, tmpdir, variant_count):
    df_a_train, df_b_train = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_a_worst_case_safety_"
                                           "active_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_b_path_train = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_b_worst_case_safety_"
                                           "active_train.csv")
    df_b_train.to_csv(df_b_path_train, index=False)
    df_a_test, df_b_test = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_a_worst_case_safety_active_"
                                          "test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    df_b_path_test = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_b_worst_case_safety_active_"
                                          "test.csv")
    df_b_test.to_csv(df_b_path_test, index=False)
    filter_lines_train = []
    filter_lines_test = []
    for variant_index in range(variant_count):
        filter_lines_train.append(f"df_a_train = df_a_train[df_a_train['A'] >= {99 - variant_index}]")
        filter_lines_test.append(f"df_a_test = df_a_test[df_a_test['A'] >= {99 - variant_index}]")
    filter_line_train = '\n        '.join(filter_lines_train)
    filter_line_test = '\n        '.join(filter_lines_test)
    test_code = cleandoc(f"""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler
        from sklearn.dummy import DummyClassifier
        import numpy as np
        from sklearn.model_selection import train_test_split
        import fuzzy_pandas as fpd

        df_a_train = pd.read_csv("{df_a_path_train}")
        {filter_line_train}
        df_b_train = pd.read_csv("{df_b_path_train}")
        df_train = fpd.fuzzy_merge(df_a_train, df_b_train, on='str_id', method='levenshtein', keep_right=['C', 'D'],
            threshold=0.99)

        df_a_test = pd.read_csv("{df_a_path_test}")
        {filter_line_test}
        df_b_test = pd.read_csv("{df_b_path_test}")
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
    data_corruption = OperatorFairness(test_transformers=False, test_selections=True)
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info
    analysis_result_with_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([OperatorDeletionFilterPushUp(singleton)]) \
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
    return analysis_result_with_opt_rule, analysis_result_without_any_opt, analysis_result_without_opt_rule, \
        data_corruption


def execute_operator_deletion_filter_push_up_average_case(data_size, tmpdir, variant_count):
    df_a_train, df_b_train = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_a_average_case_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_b_path_train = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_b_average_case_train.csv")
    df_b_train.to_csv(df_b_path_train, index=False)
    df_a_test, df_b_test = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_a_average_case_test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    df_b_path_test = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_b_average_case_test.csv")
    df_b_test.to_csv(df_b_path_test, index=False)
    filter_lines_train = []
    filter_lines_test = []
    for variant_index in range(variant_count):
        lower_bound = 80 + 20 / variant_count * variant_index
        upper_bound = 80 + 20 / variant_count * (variant_index + 1)

        filter_lines_train.append(f"df_a_train = df_a_train[(df_a_train['A'] <= {lower_bound}) "
                                  f"| (df_a_train['A'] >= {upper_bound})]")
        filter_lines_test.append(f"df_a_test = df_a_test[(df_a_test['A'] <= {lower_bound}) "
                                 f"| (df_a_test['A'] >= {upper_bound})]")
    filter_line_train = '\n        '.join(filter_lines_train)
    filter_line_test = '\n        '.join(filter_lines_test)
    test_code = cleandoc(f"""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler, OneHotEncoder
        from sklearn.dummy import DummyClassifier
        from sklearn.pipeline import Pipeline
        import numpy as np
        from sklearn.model_selection import train_test_split
        import fuzzy_pandas as fpd
        from sklearn.compose import ColumnTransformer

        df_a_train = pd.read_csv("{df_a_path_train}")
        {filter_line_train}
        df_b_train = pd.read_csv("{df_b_path_train}")
        df_train = df_a_train.merge(df_b_train, on=['str_id'])

        df_a_test = pd.read_csv("{df_a_path_test}")
        {filter_line_test}
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

        clf = DummyClassifier(strategy='constant', constant=0.)
        clf = clf.fit(train_data, train_target)
        test_score = clf.score(test_data, test_target)
        assert 0. <= test_score <= 1.
        """)
    data_corruption = OperatorFairness(test_transformers=False, test_selections=True)
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info
    analysis_result_with_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([OperatorDeletionFilterPushUp(singleton, disable_selectivity_safety=True)]) \
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
    return analysis_result_with_opt_rule, analysis_result_without_any_opt, analysis_result_without_opt_rule, \
        data_corruption


def execute_operator_deletion_filter_push_up_worst_case_safety_too_defensive_for_scenario(data_size, tmpdir, variant_count):
    df_a_train, df_b_train = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_a_worst_case_safety_failing_"
                                           "train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_b_path_train = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_b_worst_case_safety_failing_"
                                           "train.csv")
    df_b_train.to_csv(df_b_path_train, index=False)
    df_a_test, df_b_test = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_a_worst_case_safety_failing_"
                                          "test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    df_b_path_test = os.path.join(tmpdir, "operator_deletion_filter_deletion_push_up_df_b_worst_case_safety_failing_"
                                          "test.csv")
    df_b_test.to_csv(df_b_path_test, index=False)
    filter_lines_train = []
    filter_lines_test = []
    for variant_index in range(variant_count - 1):
        filter_lines_train.append(f"df_a_train = df_a_train[df_a_train['A'] != {95 - variant_index}]")
        filter_lines_test.append(f"df_a_test = df_a_test[df_a_test['A'] != {95 - variant_index}]")
    barely_fail_safety_check_threshold = int(103 - 100 / variant_count)
    if variant_count >= 1:
        # FIXME: For filters like this that basically all do the same, the performance can heavily decrease!
        #  The optimization needs a special case for this to not do the optimization then.
        filter_lines_train.append(f"df_a_train = df_a_train[df_a_train['A'] >= {barely_fail_safety_check_threshold}]")
        filter_lines_test.append(f"df_a_test = df_a_test[df_a_test['A'] >= {barely_fail_safety_check_threshold}]")
    filter_line_train = '\n        '.join(filter_lines_train)
    filter_line_test = '\n        '.join(filter_lines_test)
    test_code = cleandoc(f"""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler
        from sklearn.dummy import DummyClassifier
        import numpy as np
        from sklearn.model_selection import train_test_split
        import fuzzy_pandas as fpd

        df_a_train = pd.read_csv("{df_a_path_train}")
        {filter_line_train}
        df_b_train = pd.read_csv("{df_b_path_train}")
        df_train = fpd.fuzzy_merge(df_a_train, df_b_train, on='str_id', method='levenshtein', keep_right=['C', 'D'],
            threshold=0.99)

        df_a_test = pd.read_csv("{df_a_path_test}")
        {filter_line_test}
        df_b_test = pd.read_csv("{df_b_path_test}")
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
    data_corruption = OperatorFairness(test_transformers=False, test_selections=True)
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info
    analysis_result_with_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([OperatorDeletionFilterPushUp(singleton)]) \
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
    return analysis_result_with_opt_rule, analysis_result_without_any_opt, analysis_result_without_opt_rule, data_corruption
