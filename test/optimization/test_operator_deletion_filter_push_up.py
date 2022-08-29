"""
Tests whether the optimization works
"""
# pylint: disable=too-many-locals,invalid-name
# TODO: Clean up these tests
import os
from inspect import cleandoc

from mlwhatif import PipelineAnalyzer
from mlwhatif.analysis._operator_fairness import OperatorFairness
from mlwhatif.execution._pipeline_executor import singleton
from mlwhatif.optimization._operator_deletion_filter_push_up import OperatorDeletionFilterPushUp
from mlwhatif.testing._testing_helper_utils import WhatIfWrapper, get_test_df


def test_filter_push_up_ideal_case(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    data_size = 1000
    variant_count = 10

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

    index_filter = []
    filter_lines_train = []
    filter_lines_test = []
    for variant_index in range(variant_count):
        index_filter.append(variant_index)
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
        
        train_target = label_binarize(df_train['target'], classes=['no', 'yes'])
        train_data = df_train[['A', 'B']]
        
        test_target = label_binarize(df_test['target'], classes=['no', 'yes'])
        test_data = df_test[['A', 'B']]
        
        clf = DummyClassifier(strategy='constant', constant=0.)
        clf = clf.fit(train_data, train_target)
        test_score = clf.score(test_data, test_target)
        assert 0. <= test_score <= 1.
        """)

    data_corruption = WhatIfWrapper(
        OperatorFairness(test_transformers=False, test_selections=True),
        index_filter=index_filter)

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

    assert analysis_result_with_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)

    assert analysis_result_with_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    analysis_result_with_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-opt-orig"))
    analysis_result_with_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if"))
    analysis_result_with_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "with-opt-what-if-optimised"))

    analysis_result_without_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "without-opt-orig"))
    analysis_result_without_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-opt-what-if"))
    analysis_result_without_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-opt-what-if-optimised"))

    analysis_result_without_any_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "without-any-opt-orig"))
    analysis_result_without_any_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-any-opt-what-if"))
    analysis_result_without_any_opt.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-any-opt-what-if-optimised"))


def test_filter_push_up_worst_case_safety_inactive(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    data_size = 2000
    variant_count = 2

    df_a_train, df_b_train = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "filter_push_up_df_a_worst_case_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_b_path_train = os.path.join(tmpdir, "filter_push_up_df_b_worst_case_train.csv")
    df_b_train.to_csv(df_b_path_train, index=False)

    df_a_test, df_b_test = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "filter_push_up_df_a_worst_case_test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    df_b_path_test = os.path.join(tmpdir, "filter_push_up_df_b_worst_case_test.csv")
    df_b_test.to_csv(df_b_path_test, index=False)

    index_filter = []
    filter_lines_train = []
    filter_lines_test = []
    for variant_index in range(variant_count):
        index_filter.append(variant_index)
        # FIXME: For filters like this that basically all do the same, the performance can heavily decrease!
        #  The optimization needs a special case for this to not do the optimization then.
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

        train_target = label_binarize(df_train['target'], classes=['no', 'yes'])
        train_data = df_train[['A', 'B']]

        test_target = label_binarize(df_test['target'], classes=['no', 'yes'])
        test_data = df_test[['A', 'B']]

        clf = DummyClassifier(strategy='constant', constant=0.)
        clf = clf.fit(train_data, train_target)
        test_score = clf.score(test_data, test_target)
        assert 0. <= test_score <= 1.
        """)

    data_corruption = WhatIfWrapper(
        OperatorFairness(test_transformers=False, test_selections=True),
        index_filter=index_filter)

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

    assert analysis_result_with_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)

    assert analysis_result_with_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    analysis_result_with_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-opt-orig"))
    analysis_result_with_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if"))
    analysis_result_with_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "with-opt-what-if-optimised"))

    analysis_result_without_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "without-opt-orig"))
    analysis_result_without_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-opt-what-if"))
    analysis_result_without_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-opt-what-if-optimised"))

    analysis_result_without_any_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "without-any-opt-orig"))
    analysis_result_without_any_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-any-opt-what-if"))
    analysis_result_without_any_opt.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-any-opt-what-if-optimised"))


def test_filter_push_up_worst_case_safety_active(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    data_size = 2000
    variant_count = 2

    df_a_train, df_b_train = get_test_df(int(data_size * 0.8))
    df_a_path_train = os.path.join(tmpdir, "filter_push_up_df_a_worst_case_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_b_path_train = os.path.join(tmpdir, "filter_push_up_df_b_worst_case_train.csv")
    df_b_train.to_csv(df_b_path_train, index=False)

    df_a_test, df_b_test = get_test_df(int(data_size * 0.2))
    df_a_path_test = os.path.join(tmpdir, "filter_push_up_df_a_worst_case_test.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    df_b_path_test = os.path.join(tmpdir, "filter_push_up_df_b_worst_case_test.csv")
    df_b_test.to_csv(df_b_path_test, index=False)

    index_filter = []
    filter_lines_train = []
    filter_lines_test = []
    for variant_index in range(variant_count):
        index_filter.append(variant_index)
        # FIXME: For filters like this that basically all do the same, the performance can heavily decrease!
        #  The optimization needs a special case for this to not do the optimization then.
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

        train_target = label_binarize(df_train['target'], classes=['no', 'yes'])
        train_data = df_train[['A', 'B']]

        test_target = label_binarize(df_test['target'], classes=['no', 'yes'])
        test_data = df_test[['A', 'B']]

        clf = DummyClassifier(strategy='constant', constant=0.)
        clf = clf.fit(train_data, train_target)
        test_score = clf.score(test_data, test_target)
        assert 0. <= test_score <= 1.
        """)

    data_corruption = WhatIfWrapper(
        OperatorFairness(test_transformers=False, test_selections=True),
        index_filter=index_filter)

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

    assert analysis_result_with_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (variant_count + 1, 2)

    assert analysis_result_with_opt_rule.runtime_info.what_if_optimized_estimated <= \
           analysis_result_without_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    analysis_result_with_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "with-opt-orig"))
    analysis_result_with_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if"))
    analysis_result_with_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "with-opt-what-if-optimised"))

    analysis_result_without_opt_rule.save_original_dag_to_path(os.path.join(str(tmpdir), "without-opt-orig"))
    analysis_result_without_opt_rule.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-opt-what-if"))
    analysis_result_without_opt_rule.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-opt-what-if-optimised"))

    analysis_result_without_any_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "without-any-opt-orig"))
    analysis_result_without_any_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "without-any-opt-what-if"))
    analysis_result_without_any_opt.save_optimised_what_if_dags_to_path(
        os.path.join(str(tmpdir), "without-any-opt-what-if-optimised"))
