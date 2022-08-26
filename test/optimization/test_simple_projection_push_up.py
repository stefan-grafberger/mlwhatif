"""
Tests whether the optimization works
"""
import os
from inspect import cleandoc

import networkx
from testfixtures import compare

from example_pipelines import ADULT_SIMPLE_PY
from mlwhatif import PipelineAnalyzer
from mlwhatif.analysis._data_corruption import DataCorruption, CorruptionType
from mlwhatif.instrumentation._pipeline_executor import singleton
from mlwhatif.optimization._simple_projection_push_up import SimpleProjectionPushUp
from mlwhatif.testing._testing_helper_utils import get_expected_dag_adult_easy

import pandas as pd
import numpy as np
from numpy.random import randint, shuffle
import random


def test_projection_push_up_ideal_case(tmpdir):
    """
    Tests whether the .py version of the inspector works
    """
    tmpdir.mkdir("projection_push_up")

    df_a_train, df_b_train = get_test_df_ideal_case(400)
    df_a_path_train = os.path.join(tmpdir, "projection_push_up", "projection_push_up_df_a_ideal_case_train.csv")
    df_a_train.to_csv(df_a_path_train, index=False)
    df_b_path_train = os.path.join(tmpdir, "projection_push_up", "projection_push_up_df_b_ideal_case_train.csv")
    df_b_train.to_csv(df_b_path_train, index=False)

    df_a_test, df_b_test = get_test_df_ideal_case(50)
    df_a_path_test = os.path.join(tmpdir, "projection_push_up", "projection_push_up_df_a_ideal_case_train.csv")
    df_a_test.to_csv(df_a_path_test, index=False)
    df_b_path_test = os.path.join(tmpdir, "projection_push_up", "projection_push_up_df_b_ideal_case_train.csv")
    df_b_test.to_csv(df_b_path_test, index=False)

    test_code = cleandoc(f"""
        import pandas as pd
        from sklearn.preprocessing import label_binarize, StandardScaler
        from sklearn.dummy import DummyClassifier
        import numpy as np
        from sklearn.model_selection import train_test_split
        import fuzzy_pandas as fpd
        
        df_a_train = pd.read_csv("{df_a_path_train}", engine='python')
        df_b_train = pd.read_csv("{df_b_path_train}", engine='python')
        df_train = fpd.fuzzy_merge(df_a_train, df_b_train, on='str_id', method='levenshtein', keep_right=['C', 'D'],
            threshold=0.99)
        df_train = df_train[df_train['A'] >= 95]
        
        df_a_test = pd.read_csv("{df_a_path_test}", engine='python')
        df_b_test = pd.read_csv("{df_b_path_test}", engine='python')
        df_test = fpd.fuzzy_merge(df_a_test, df_b_test, on='str_id', method='levenshtein', keep_right=['C', 'D'],
            threshold=0.99)
        df_test = df_train[df_train['A'] >= 95]
        
        train_target = label_binarize(df_train['target'], classes=['no', 'yes'])
        train_data = df_train[['A', 'B']]
        
        test_target = label_binarize(df_test['target'], classes=['no', 'yes'])
        test_data = df_test[['A', 'B']]
        
        clf = DummyClassifier(strategy='constant', constant=0.)
        clf = clf.fit(train_data, train_target)
        test_score = clf.score(test_data, test_target)
        assert 0. <= test_score <= 1.
        """)

    def corruption(pandas_df):
        pandas_df['B'] = 0
        return pandas_df

    data_corruption = DataCorruption({'A': CorruptionType.SCALING, 'B': corruption}, also_corrupt_train=True,
                                     corruption_percentages=[1.])
    dag_extraction_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .execute() \
        .dag_extraction_info

    analysis_result_with_opt_rule = PipelineAnalyzer \
        .on_previously_extracted_pipeline(dag_extraction_result) \
        .add_what_if_analysis(data_corruption) \
        .overwrite_optimization_rules([SimpleProjectionPushUp(singleton)])  \
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

    assert analysis_result_with_opt_rule.analysis_to_result_reports[data_corruption].shape == (3, 4)
    assert analysis_result_without_opt_rule.analysis_to_result_reports[data_corruption].shape == (3, 4)
    assert analysis_result_without_any_opt.analysis_to_result_reports[data_corruption].shape == (3, 4)

    assert analysis_result_with_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_opt_rule.runtime_info.what_if_optimized_estimated < \
           analysis_result_without_any_opt.runtime_info.what_if_unoptimized_estimated

    intermediate_extraction_orig_path = os.path.join(str(tmpdir), "projection_push_up", "data_corruption-orig")
    intermediate_extraction_generated_path = os.path.join(str(tmpdir), "projection_push_up", "data_corruption-what-if")
    intermediate_extraction_optimised_path = os.path.join(str(tmpdir), "projection_push_up",
                                                          "data_corruption-what-if-optimised")

    analysis_result_without_opt_rule.save_original_dag_to_path(intermediate_extraction_orig_path)
    analysis_result_without_opt_rule.save_what_if_dags_to_path(intermediate_extraction_generated_path)
    analysis_result_without_opt_rule.save_optimised_what_if_dags_to_path(intermediate_extraction_optimised_path)


def get_test_df_ideal_case(data_frame_rows):
    """Get some test data """
    sizes_before_join = int(data_frame_rows * 1.1)
    start_with_offset = int(data_frame_rows * 0.1)
    end_with_offset = start_with_offset + sizes_before_join
    assert sizes_before_join - start_with_offset == data_frame_rows

    id_a = np.arange(sizes_before_join)
    shuffle(id_a)
    a = randint(0, 100, size=sizes_before_join)
    b = randint(0, 100, size=sizes_before_join)
    categories = ['cat_a', 'cat_b', 'cat_c']
    group_col_1 = pd.Series(random.choices(categories, k=sizes_before_join))
    group_col_2 = pd.Series(random.choices(categories, k=sizes_before_join))
    group_col_3 = pd.Series(random.choices(categories, k=sizes_before_join))
    target = pd.Series(random.choices(["yes", "no"], k=sizes_before_join))
    id_b = np.arange(start_with_offset, end_with_offset)
    shuffle(id_b)
    c = randint(0, 100, size=sizes_before_join)
    d = randint(0, 100, size=sizes_before_join)
    df_a = pd.DataFrame(zip(id_a, a, b, group_col_1, group_col_2, group_col_3, target),
                        columns=['id', 'A', 'B', 'group_col_1', 'group_col_2', 'group_col_3', 'target'])
    df_a["str_id"] = "id_" + df_a["id"].astype(str)
    df_b = pd.DataFrame(zip(id_b, c, d), columns=['id', 'C', 'D'])
    df_b["str_id"] = "id_" + df_b["id"].astype(str)
    return df_a, df_b
