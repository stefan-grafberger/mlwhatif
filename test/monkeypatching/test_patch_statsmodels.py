"""
Tests whether the monkey patching works for all patched statsmodels methods
"""
from inspect import cleandoc

import networkx
from testfixtures import compare

from mlwhatif import OperatorContext, FunctionInfo, OperatorType
from mlwhatif.instrumentation import _pipeline_executor
from mlwhatif.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo


def test_statsmodels_add_constant():
    """
    Tests whether the monkey patching of ('statsmodel.api', 'add_constant') works
    """
    test_code = cleandoc("""
        import numpy as np
        import statsmodels.api as sm
        np.random.seed(42)
        test = np.random.random(100)
        test = sm.add_constant(test)
        assert len(test) == 100
        """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    expected_dag = networkx.DiGraph()
    expected_random = DagNode(0,
                              BasicCodeLocation("<string-source>", 4),
                              OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('numpy.random', 'random')),
                              DagNodeDetails('random', ['array']),
                              OptionalCodeInfo(CodeReference(4, 7, 4, 28), "np.random.random(100)"))

    expected_constant = DagNode(1,
                                BasicCodeLocation("<string-source>", 5),
                                OperatorContext(OperatorType.PROJECTION_MODIFY,
                                                FunctionInfo('statsmodel.api', 'add_constant')),
                                DagNodeDetails('Adds const column', ['array']),
                                OptionalCodeInfo(CodeReference(5, 7, 5, 28), "sm.add_constant(test)"))
    expected_dag.add_edge(expected_random, expected_constant, arg_index=0)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))


def test_get_rdataset():
    """
    Tests whether the monkey patching of ('statsmodels.datasets', 'get_rdataset') works
    """
    test_code = cleandoc("""
        import statsmodels.api as sm

        dat = sm.datasets.get_rdataset("Guerry", "HistData").data
        assert len(dat) == 86
        """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)

    extracted_node: DagNode = list(inspector_result.dag.nodes)[0]
    expected_node = DagNode(0,
                            BasicCodeLocation("<string-source>", 3),
                            OperatorContext(OperatorType.DATA_SOURCE, FunctionInfo('statsmodels.datasets',
                                                                                   'get_rdataset')),
                            DagNodeDetails('Data from A.-M. Guerry, "Essay on the Moral Statistics of France"',
                                           ['dept', 'Region', 'Department', 'Crime_pers', 'Crime_prop', 'Literacy',
                                            'Donations', 'Infants', 'Suicides', 'MainCity', 'Wealth', 'Commerce',
                                            'Clergy', 'Crime_parents', 'Infanticide', 'Donation_clergy', 'Lottery',
                                            'Desertion', 'Instruction', 'Prostitutes', 'Distance', 'Area', 'Pop1831']),
                            OptionalCodeInfo(CodeReference(3, 6, 3, 52),
                                             """sm.datasets.get_rdataset("Guerry", "HistData")"""))
    compare(extracted_node, expected_node)


def test_ols_fit():
    """
    Tests whether the monkey patching of ('statsmodels.regression.linear_model.OLS', 'fit') works
    """
    test_code = cleandoc("""
        import numpy as np
        import statsmodels.api as sm
        np.random.seed(42)
        nobs = 100
        X = np.random.random((nobs, 2))
        X = sm.add_constant(X)
        beta = [1, .1, .5]
        e = np.random.random(nobs)
        y = np.dot(X, beta) + e
        results = sm.OLS(y, X).fit()
        assert results.summary() is not None
        """)

    inspector_result = _pipeline_executor.singleton.run(python_code=test_code, track_code_references=True)
    inspector_result.dag.remove_nodes_from(list(inspector_result.dag.nodes)[0:4])
    inspector_result.dag.remove_node(list(inspector_result.dag.nodes)[1])

    expected_dag = networkx.DiGraph()
    expected_train_data = DagNode(3,
                                  BasicCodeLocation("<string-source>", 10),
                                  OperatorContext(OperatorType.TRAIN_DATA,
                                                  FunctionInfo('statsmodel.api.OLS', 'fit')),
                                  DagNodeDetails(None, ['array']),
                                  OptionalCodeInfo(CodeReference(10, 10, 10, 22), 'sm.OLS(y, X)'))
    expected_train_labels = DagNode(4,
                                    BasicCodeLocation("<string-source>", 10),
                                    OperatorContext(OperatorType.TRAIN_LABELS,
                                                    FunctionInfo('statsmodel.api.OLS', 'fit')),
                                    DagNodeDetails(None, ['array']),
                                    OptionalCodeInfo(CodeReference(10, 10, 10, 22), 'sm.OLS(y, X)'))
    expected_ols = DagNode(5,
                           BasicCodeLocation("<string-source>", 10),
                           OperatorContext(OperatorType.ESTIMATOR,
                                           FunctionInfo('statsmodel.api.OLS', 'fit')),
                           DagNodeDetails('Decision Tree', []),
                           OptionalCodeInfo(CodeReference(10, 10, 10, 22), 'sm.OLS(y, X)'))
    expected_dag.add_edge(expected_train_data, expected_ols, arg_index=0)
    expected_dag.add_edge(expected_train_labels, expected_ols, arg_index=1)

    compare(networkx.to_dict_of_dicts(inspector_result.dag), networkx.to_dict_of_dicts(expected_dag))
