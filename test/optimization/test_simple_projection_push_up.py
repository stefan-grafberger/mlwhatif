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

    df_a, df_b = get_test_df_ideal_case(5000)
    df_a_path = os.path.join(tmpdir, "projection_push_up", "projection_push_up_df_a_ideal_case.csv")
    df_a.to_csv(df_a_path, index=False)
    df_b_path = os.path.join(tmpdir, "projection_push_up", "projection_push_up_df_b_ideal_case.csv")
    df_b.to_csv(df_b_path, index=False)

    test_code = cleandoc(f"""
            import pandas as pd
            from sklearn.preprocessing import label_binarize, StandardScaler
            from sklearn.dummy import DummyClassifier
            import numpy as np
            from sklearn.model_selection import train_test_split
            import fuzzy_pandas as fpd

            df_a = pd.read_csv("{df_a_path}", engine='python')
            df_b = pd.read_csv("{df_b_path}", engine='python')
            pandas_df = fpd.fuzzy_merge(df_a, df_b, on='str_id', method='levenshtein', keep_right=['C', 'D'],
                threshold=0.99)
            pandas_df = pandas_df[pandas_df['A'] >= 95]
            train_data = pandas_df[pandas_df['B'] < 80]
            test_data = pandas_df[pandas_df['B'] >= 80]

            target = label_binarize(train_data['target'], classes=['no', 'yes'])
            train_data = train_data[['A', 'B']]

            clf = DummyClassifier(strategy='constant', constant=0.)
            clf = clf.fit(train_data, target)

            test_labels = label_binarize(test_data['target'], classes=['no', 'yes'])
            test_data = test_data[['A', 'B']]
            test_score = clf.score(test_data, test_labels)
            assert 0. <= test_score <= 1.
            """)

    def corruption(pandas_df):
        pandas_df['B'] = 0
        return pandas_df

    data_corruption = DataCorruption({'A': CorruptionType.SCALING, 'B': corruption}, also_corrupt_train=True)

    analysis_result = PipelineAnalyzer \
        .on_pipeline_from_string(test_code) \
        .add_what_if_analysis(data_corruption) \
        .execute()

    report = analysis_result.analysis_to_result_reports[data_corruption]
    assert report.shape == (7, 4)

    intermediate_extraction_orig_path = os.path.join(str(tmpdir), "projection_push_up", "data_corruption-orig")
    intermediate_extraction_generated_path = os.path.join(str(tmpdir), "projection_push_up", "data_corruption-what-if")
    intermediate_extraction_optimised_path = os.path.join(str(tmpdir), "projection_push_up",
                                                          "data_corruption-what-if-optimised")

    analysis_result.save_original_dag_to_path(intermediate_extraction_orig_path)
    analysis_result.save_what_if_dags_to_path(intermediate_extraction_generated_path)
    analysis_result.save_optimised_what_if_dags_to_path(intermediate_extraction_optimised_path)


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
