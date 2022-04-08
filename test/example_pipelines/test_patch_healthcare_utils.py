"""
Tests whether the monkey patching works for all patched sklearn methods
"""
from inspect import cleandoc

import networkx
from testfixtures import compare

from example_pipelines.healthcare import custom_monkeypatching
from mlwhatif import OperatorContext, FunctionInfo, OperatorType
from mlwhatif.instrumentation import _pipeline_executor
from mlwhatif.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo


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
                                   DagNodeDetails(None, ['A']),
                                   OptionalCodeInfo(CodeReference(5, 5, 5, 62),
                                                    "pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})"))
    expected_estimator = DagNode(1,
                                 BasicCodeLocation("<string-source>", 6),
                                 OperatorContext(OperatorType.TRANSFORMER,
                                                 FunctionInfo('example_pipelines.healthcare.healthcare_utils',
                                                              'MyW2VTransformer')),
                                 DagNodeDetails('Word2Vec: fit_transform', ['array']),
                                 OptionalCodeInfo(CodeReference(6, 14, 6, 62),
                                                  'MyW2VTransformer(min_count=2, size=2, workers=1)'))
    expected_dag.add_edge(expected_data_source, expected_estimator)
    expected_data_source_two = DagNode(2,
                                       BasicCodeLocation("<string-source>", 9),
                                       OperatorContext(OperatorType.DATA_SOURCE,
                                                       FunctionInfo('pandas.core.frame', 'DataFrame')),
                                       DagNodeDetails(None, ['A']),
                                       OptionalCodeInfo(CodeReference(9, 10, 9, 67),
                                                        "pd.DataFrame({'A': ['cat_a', 'cat_b', 'cat_a', 'cat_c']})"))
    expected_estimator_two = DagNode(3,
                                     BasicCodeLocation("<string-source>", 6),
                                     OperatorContext(OperatorType.TRANSFORMER,
                                                     FunctionInfo('example_pipelines.healthcare.healthcare_utils',
                                                                  'MyW2VTransformer')),
                                     DagNodeDetails('Word2Vec: transform', ['array']),
                                     OptionalCodeInfo(CodeReference(6, 14, 6, 62),
                                                      'MyW2VTransformer(min_count=2, size=2, workers=1)'))
    expected_dag.add_edge(expected_data_source_two, expected_estimator_two)
    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))


def test_arg_capturing_my_word_to_vec_transformer():
    """
    Tests whether ArgumentCapturing works for MyW2VTransformer
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
    fit_transform_node = list(inspector_result.dag.nodes)[1]
    transform_node = list(inspector_result.dag.nodes)[3]

    expected_fit_transform = DagNode(1,
                                     BasicCodeLocation("<string-source>", 6),
                                     OperatorContext(OperatorType.TRANSFORMER,
                                                     FunctionInfo('example_pipelines.healthcare.healthcare_utils',
                                                                  'MyW2VTransformer')),
                                     DagNodeDetails('Word2Vec: fit_transform', ['array']),
                                     OptionalCodeInfo(CodeReference(6, 14, 6, 62),
                                                      'MyW2VTransformer(min_count=2, size=2, workers=1)'))
    expected_transform = DagNode(3,
                                 BasicCodeLocation("<string-source>", 6),
                                 OperatorContext(OperatorType.TRANSFORMER,
                                                 FunctionInfo('example_pipelines.healthcare.healthcare_utils',
                                                              'MyW2VTransformer')),
                                 DagNodeDetails('Word2Vec: transform', ['array']),
                                 OptionalCodeInfo(CodeReference(6, 14, 6, 62),
                                                  'MyW2VTransformer(min_count=2, size=2, workers=1)'))

    compare(fit_transform_node, expected_fit_transform)
    compare(transform_node, expected_transform)
