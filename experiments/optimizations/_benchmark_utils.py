"""
Some util functions used in other tests
"""
import random
from typing import List, Iterable, Dict

import networkx
import numpy
import pandas
from numpy.random.mtrand import shuffle, randint

from mlmq.analysis._what_if_analysis import WhatIfAnalysis
from mlmq.execution._patches import PipelinePatch
from mlmq.execution._pipeline_executor import singleton


class WhatIfWrapper(WhatIfAnalysis):
    """A simple wrapper to filter plans to try for benchmarking specific optimizations"""

    def __init__(self, what_if_analysis: WhatIfAnalysis, index_filter: List[int]):
        self.what_if_analysis = what_if_analysis
        self.index_filter = index_filter

    def generate_plans_to_try(self, dag: networkx.DiGraph) -> Iterable[Iterable[PipelinePatch]]:
        unfiltered_plans = list(self.what_if_analysis.generate_plans_to_try(dag))
        filtered_plans = [unfiltered_plans[index] for index in self.index_filter]
        return filtered_plans

    def generate_final_report(self, extracted_plan_results: Dict[str, any]) -> any:
        keys_and_labels = singleton.labels_to_extracted_plan_results.items()
        return pandas.DataFrame(keys_and_labels, columns=['result_key', 'result_value'])


def get_test_df(data_frame_rows):
    """Get some test data """
    # pylint: disable=too-many-locals,invalid-name
    sizes_before_join = int(data_frame_rows * 1.1)
    start_with_offset = int(data_frame_rows * 0.1)
    end_with_offset = start_with_offset + sizes_before_join
    assert sizes_before_join - start_with_offset == data_frame_rows

    id_a = numpy.arange(sizes_before_join)
    shuffle(id_a)
    a = randint(0, 100, size=sizes_before_join)
    b = randint(0, 100, size=sizes_before_join)
    categories = ['cat_a', 'cat_b', 'cat_c']
    group_col_1 = pandas.Series(random.choices(categories, k=sizes_before_join))
    group_col_2 = pandas.Series(random.choices(categories, k=sizes_before_join))
    group_col_3 = pandas.Series(random.choices(categories, k=sizes_before_join))
    target = pandas.Series(random.choices(["yes", "no"], k=sizes_before_join))
    target_featurized = pandas.Series(random.choices([0., 1.], k=sizes_before_join))
    id_b = numpy.arange(start_with_offset, end_with_offset)
    shuffle(id_b)
    c = randint(0, 100, size=sizes_before_join)
    d = randint(0, 100, size=sizes_before_join)
    df_a = pandas.DataFrame(zip(id_a, a, b, group_col_1, group_col_2, group_col_3, target, target_featurized),
                            columns=['id', 'A', 'B', 'group_col_1', 'group_col_2', 'group_col_3', 'target',
                                     'target_featurized'])
    df_a["str_id"] = "id_" + df_a["id"].astype(str)
    df_b = pandas.DataFrame(zip(id_b, c, d), columns=['id', 'C', 'D'])
    df_b["str_id"] = "id_" + df_b["id"].astype(str)
    return df_a, df_b
