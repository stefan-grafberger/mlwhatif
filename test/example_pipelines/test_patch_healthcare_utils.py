"""
Tests whether the monkey patching works for all patched sklearn methods
"""
from functools import partial
from inspect import cleandoc
from types import FunctionType

import networkx
import pandas
from testfixtures import compare, Comparison, RangeComparison

from example_pipelines.healthcare import custom_monkeypatching
from mlmq import OperatorContext, FunctionInfo, OperatorType
from mlmq.execution import _pipeline_executor
from mlmq.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo, OptimizerInfo


def test_my_word_to_vec_transformer():
    """
    Tests whether the monkey patching of ('example_pipelines.healthcare.healthcare_utils', 'MyW2VTransformer') works
    """
    test_code = cleandoc("""
                import pandas as pd
                from example_pipelines.healthcare.healthcare_utils import MyW2VTransformer
                import numpy as np

                df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                word_to_vec = MyW2VTransformer(min_count=2, size=2, workers=1)
                encoded_data = word_to_vec.fit_transform(df)
                assert encoded_data.shape == (4, 2)
                test_df = pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})
                encoded_data = word_to_vec.transform(test_df)
                """)
    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True,
                                                        custom_monkey_patching=[custom_monkeypatching])

    expected_dag = networkx.DiGraph()
    expected_data_source = DagNode(0,
                                   BasicCodeLocation("<string-source>", 5),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.core.frame', 'DataFrame')),
                                   DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 2000), (4, 1),
                                                                             RangeComparison(0, 800))),
                                   OptionalCodeInfo(CodeReference(5, 5, 5, 62),
                                                    "pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})"),
                                   Comparison(partial))
    expected_transformer = DagNode(1,
                                   BasicCodeLocation("<string-source>", 6),
                                   OperatorContext(OperatorType.TRANSFORMER,
                                                   FunctionInfo('example_pipelines.healthcare.healthcare_utils',
                                                                'MyW2VTransformer')),
                                   DagNodeDetails('Word2Vec: fit_transform', ['array'],
                                                  OptimizerInfo(RangeComparison(0, 4000), (4, 2),
                                                                RangeComparison(0, 10000))),
                                   OptionalCodeInfo(CodeReference(6, 14, 6, 62),
                                                    'MyW2VTransformer(min_count=2, size=2, workers=1)'),
                                   Comparison(FunctionType))
    expected_dag.add_edge(expected_data_source, expected_transformer, arg_index=0)
    expected_data_source_two = DagNode(2,
                                       BasicCodeLocation("<string-source>", 9),
                                       OperatorContext(OperatorType.DATA_SOURCE,
                                                       FunctionInfo('pandas.core.frame', 'DataFrame')),
                                       DagNodeDetails(None, ['A'], OptimizerInfo(RangeComparison(0, 2000), (4, 1),
                                                                                 RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(9, 10, 9, 67),
                                                        "pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})"),
                                       Comparison(partial))
    expected_transformer_two = DagNode(3,
                                       BasicCodeLocation("<string-source>", 6),
                                       OperatorContext(OperatorType.TRANSFORMER,
                                                       FunctionInfo('example_pipelines.healthcare.healthcare_utils',
                                                                    'MyW2VTransformer')),
                                       DagNodeDetails('Word2Vec: transform', ['array'],
                                                      OptimizerInfo(RangeComparison(0, 2000), (4, 2),
                                                                    RangeComparison(0, 800))),
                                       OptionalCodeInfo(CodeReference(6, 14, 6, 62),
                                                        'MyW2VTransformer(min_count=2, size=2, workers=1)'),
                                       Comparison(FunctionType))
    expected_dag.add_edge(expected_transformer, expected_transformer_two, arg_index=0)
    expected_dag.add_edge(expected_data_source_two, expected_transformer_two, arg_index=1)
    compare(networkx.to_dict_of_dicts(inspector_result.original_dag), networkx.to_dict_of_dicts(expected_dag))

    fit_transform_node = list(inspector_result.original_dag.nodes)[1]
    transform_node = list(inspector_result.original_dag.nodes)[3]
    pandas_df = pandas.DataFrame({'A': ['cat_a', 'cat_b', 'cat_b', 'cat_c']})
    fit_transformed_result = fit_transform_node.processing_func(pandas_df)
    assert fit_transformed_result.shape == (4, 2)

    test_df = pandas.DataFrame({'A': ['cat_a', 'cat_b', 'cat_c', 'cat_c']})
    encoded_data = transform_node.processing_func(fit_transformed_result, test_df)
    assert encoded_data.shape == (4, 2)
