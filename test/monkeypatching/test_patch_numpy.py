"""
Tests whether the monkey patching works for all patched numpy methods
"""
from functools import partial
from inspect import cleandoc

from testfixtures import compare, Comparison, RangeComparison

from mlmq import OperatorContext, FunctionInfo, OperatorType
from mlmq.execution import _pipeline_executor
from mlmq.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo, OptimizerInfo


def test_numpy_random():
    """
    Tests whether the monkey patching of ('numpy.random', 'random') works
    """
    test_code = cleandoc("""
        import numpy as np
        np.random.seed(42)
        test = np.random.random(100)
        assert len(test) == 100
        """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    extracted_node: DagNode = list(inspector_result.original_dag.nodes)[0]

    expected_node = DagNode(0,
                            BasicCodeLocation("<string-source>", 3),
                            OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('numpy.random', 'random')),
                            DagNodeDetails('random', ['array'], OptimizerInfo(RangeComparison(0, 800), (100, 1),
                                                                              RangeComparison(0, 2000))),
                            OptionalCodeInfo(CodeReference(3, 7, 3, 28), "np.random.random(100)"),
                            Comparison(partial))
    compare(extracted_node, expected_node)

    assert len(extracted_node.processing_func()) == 100
