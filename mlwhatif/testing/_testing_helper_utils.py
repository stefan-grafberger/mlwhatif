"""
Some util functions used in other tests
"""
import os
import sys
from functools import partial
from inspect import cleandoc
from types import FunctionType
from unittest.mock import patch

import networkx
from pandas import DataFrame
from testfixtures import Comparison, RangeComparison

from example_pipelines.healthcare import custom_monkeypatching
from experiments.end_to_end.benchmarks_end_to_end import get_analysis_for_scenario_and_dataset
from mlwhatif import OperatorContext, FunctionInfo, OperatorType
from mlwhatif._pipeline_analyzer import PipelineAnalyzer
from mlwhatif.instrumentation._dag_node import DagNode, CodeReference, BasicCodeLocation, DagNodeDetails, \
    OptionalCodeInfo, OptimizerInfo
from mlwhatif.utils import get_project_root
from mlwhatif.visualisation._visualisation import save_fig_to_path


def get_expected_dag_adult_easy(caller_filename: str, line_offset: int = 0, with_code_references=True):
    """
    Get the expected DAG for the adult_easy pipeline
    """
    # pylint: disable=too-many-locals
    # The line numbers differ slightly between the .py file and the.ipynb file
    expected_graph = networkx.DiGraph()

    expected_data_source = DagNode(0,
                                   BasicCodeLocation(caller_filename, 12 + line_offset),
                                   OperatorContext(OperatorType.DATA_SOURCE,
                                                   FunctionInfo('pandas.io.parsers', 'read_csv')),
                                   DagNodeDetails('adult_train.csv',
                                                  ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                                   'marital-status', 'occupation', 'relationship', 'race',
                                                   'sex',
                                                   'capital-gain', 'capital-loss', 'hours-per-week',
                                                   'native-country',
                                                   'income-per-year'],
                                                  OptimizerInfo(RangeComparison(0, 100000), (22792, 15),
                                                                RangeComparison(0, 30000000))
                                                  ),
                                   OptionalCodeInfo(CodeReference(12 + line_offset, 11, 12 + line_offset, 62),
                                                    "pd.read_csv(train_file, na_values='?', index_col=0)"),
                                   Comparison(partial))
    expected_graph.add_node(expected_data_source)

    expected_select = DagNode(1,
                              BasicCodeLocation(caller_filename, 14 + line_offset),
                              OperatorContext(OperatorType.SELECTION, FunctionInfo('pandas.core.frame', 'dropna')),
                              DagNodeDetails('dropna',
                                             ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                                              'marital-status',
                                              'occupation', 'relationship', 'race', 'sex', 'capital-gain',
                                              'capital-loss',
                                              'hours-per-week', 'native-country', 'income-per-year'],
                                             OptimizerInfo(RangeComparison(0, 100000), (21145, 15),
                                                           RangeComparison(0, 30000000))),
                              OptionalCodeInfo(CodeReference(14 + line_offset, 7, 14 + line_offset, 24),
                                               'raw_data.dropna()'),
                              Comparison(FunctionType))
    expected_graph.add_edge(expected_data_source, expected_select, arg_index=0)

    pipeline_str = "compose.ColumnTransformer(transformers=[\n" \
                   "    ('categorical', preprocessing.OneHotEncoder(handle_unknown='ignore'), " \
                   "['education', 'workclass']),\n" \
                   "    ('numeric', preprocessing.StandardScaler(), ['age', 'hours-per-week'])\n" \
                   "])"
    expected_pipeline_project_one = DagNode(4,
                                            BasicCodeLocation(caller_filename, 18 + line_offset),
                                            OperatorContext(OperatorType.PROJECTION,
                                                            FunctionInfo('sklearn.compose._column_transformer',
                                                                         'ColumnTransformer')),
                                            DagNodeDetails("to ['education', 'workclass']", ['education', 'workclass'],
                                                           OptimizerInfo(RangeComparison(0, 100000), (21145, 2),
                                                                         RangeComparison(0, 30000000))),
                                            OptionalCodeInfo(CodeReference(18 + line_offset, 25, 21 + line_offset, 2),
                                                             pipeline_str),
                                            Comparison(FunctionType))
    expected_graph.add_edge(expected_select, expected_pipeline_project_one, arg_index=0)
    expected_pipeline_project_two = DagNode(6,
                                            BasicCodeLocation(caller_filename, 18 + line_offset),
                                            OperatorContext(OperatorType.PROJECTION,
                                                            FunctionInfo('sklearn.compose._column_transformer',
                                                                         'ColumnTransformer')),
                                            DagNodeDetails("to ['age', 'hours-per-week']", ['age', 'hours-per-week'],
                                                           OptimizerInfo(RangeComparison(0, 100000), (21145, 2),
                                                                         RangeComparison(0, 30000000))),
                                            OptionalCodeInfo(CodeReference(18 + line_offset, 25, 21 + line_offset, 2),
                                                             pipeline_str),
                                            Comparison(FunctionType))
    expected_graph.add_edge(expected_select, expected_pipeline_project_two, arg_index=0)

    expected_pipeline_transformer_one = DagNode(5,
                                                BasicCodeLocation(caller_filename, 19 + line_offset),
                                                OperatorContext(OperatorType.TRANSFORMER,
                                                                FunctionInfo('sklearn.preprocessing._encoders',
                                                                             'OneHotEncoder')),
                                                DagNodeDetails('One-Hot Encoder: fit_transform', ['array'],
                                                               OptimizerInfo(RangeComparison(0, 100000), (21145, 23),
                                                                             RangeComparison(0, 30000000))),
                                                OptionalCodeInfo(CodeReference(19 + line_offset, 20, 19 + line_offset,
                                                                               72),
                                                                 "preprocessing.OneHotEncoder(handle_unknown='ignore')"),
                                                Comparison(FunctionType))
    expected_pipeline_transformer_two = DagNode(7,
                                                BasicCodeLocation(caller_filename, 20 + line_offset),
                                                OperatorContext(OperatorType.TRANSFORMER,
                                                                FunctionInfo('sklearn.preprocessing._data',
                                                                             'StandardScaler')),
                                                DagNodeDetails('Standard Scaler: fit_transform', ['array'],
                                                               OptimizerInfo(RangeComparison(0, 100000), (21145, 2),
                                                                             RangeComparison(0, 30000000))),
                                                OptionalCodeInfo(CodeReference(20 + line_offset, 16, 20 + line_offset,
                                                                               46),
                                                                 'preprocessing.StandardScaler()'),
                                                Comparison(FunctionType))
    expected_graph.add_edge(expected_pipeline_project_one, expected_pipeline_transformer_one, arg_index=0)
    expected_graph.add_edge(expected_pipeline_project_two, expected_pipeline_transformer_two, arg_index=0)

    expected_pipeline_concatenation = DagNode(8,
                                              BasicCodeLocation(caller_filename, 18 + line_offset),
                                              OperatorContext(OperatorType.CONCATENATION,
                                                              FunctionInfo('sklearn.compose._column_transformer',
                                                                           'ColumnTransformer')),
                                              DagNodeDetails(None, ['array'],
                                                             OptimizerInfo(RangeComparison(0, 100000), (21145, 25),
                                                                           RangeComparison(0, 30000000))),
                                              OptionalCodeInfo(CodeReference(18 + line_offset, 25, 21 + line_offset, 2),
                                                               pipeline_str),
                                              Comparison(FunctionType))
    expected_graph.add_edge(expected_pipeline_transformer_one, expected_pipeline_concatenation, arg_index=0)
    expected_graph.add_edge(expected_pipeline_transformer_two, expected_pipeline_concatenation, arg_index=1)

    expected_train_data = DagNode(9,
                                  BasicCodeLocation(caller_filename, 26 + line_offset),
                                  OperatorContext(OperatorType.TRAIN_DATA,
                                                  FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')),
                                  DagNodeDetails(None, ['array'], OptimizerInfo(RangeComparison(0, 100000), (21145, 25),
                                                                                RangeComparison(0, 30000000))),
                                  OptionalCodeInfo(CodeReference(26 + line_offset, 19, 26 + line_offset, 48),
                                                   'tree.DecisionTreeClassifier()'),
                                  Comparison(FunctionType))
    expected_graph.add_edge(expected_pipeline_concatenation, expected_train_data, arg_index=0)

    expected_project = DagNode(2,
                               BasicCodeLocation(caller_filename, 16 + line_offset),
                               OperatorContext(OperatorType.PROJECTION,
                                               FunctionInfo('pandas.core.frame', '__getitem__')),
                               DagNodeDetails("to ['income-per-year']", ['income-per-year'],
                                              OptimizerInfo(RangeComparison(0, 100000), (21145, 1),
                                                            RangeComparison(0, 30000000))),
                               OptionalCodeInfo(CodeReference(16 + line_offset, 38, 16 + line_offset, 61),
                                                "data['income-per-year']"),
                               Comparison(FunctionType))
    expected_graph.add_edge(expected_select, expected_project, arg_index=0)

    expected_project_modify = DagNode(3,
                                      BasicCodeLocation(caller_filename, 16 + line_offset),
                                      OperatorContext(OperatorType.PROJECTION_MODIFY,
                                                      FunctionInfo('sklearn.preprocessing._label', 'label_binarize')),
                                      DagNodeDetails("label_binarize, classes: ['>50K', '<=50K']", ['array'],
                                                     OptimizerInfo(RangeComparison(0, 100000), (21145, 1),
                                                                   RangeComparison(0, 30000000))),
                                      OptionalCodeInfo(CodeReference(16 + line_offset, 9, 16 + line_offset, 89),
                                                       "preprocessing.label_binarize(data['income-per-year'], "
                                                       "classes=['>50K', '<=50K'])"),
                                      Comparison(FunctionType))
    expected_graph.add_edge(expected_project, expected_project_modify, arg_index=0)

    expected_train_labels = DagNode(10,
                                    BasicCodeLocation(caller_filename, 26 + line_offset),
                                    OperatorContext(OperatorType.TRAIN_LABELS,
                                                    FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')),
                                    DagNodeDetails(None, ['array'],
                                                   OptimizerInfo(RangeComparison(0, 100000), (21145, 1),
                                                                 RangeComparison(0, 30000000))),
                                    OptionalCodeInfo(CodeReference(26 + line_offset, 19, 26 + line_offset, 48),
                                                     'tree.DecisionTreeClassifier()'),
                                    Comparison(FunctionType))
    expected_graph.add_edge(expected_project_modify, expected_train_labels, arg_index=0)

    expected_estimator = DagNode(11,
                                 BasicCodeLocation(caller_filename, 26 + line_offset),
                                 OperatorContext(OperatorType.ESTIMATOR,
                                                 FunctionInfo('sklearn.tree._classes', 'DecisionTreeClassifier')),
                                 DagNodeDetails('Decision Tree', [],
                                                OptimizerInfo(RangeComparison(0, 100000), None,
                                                              RangeComparison(0, 30000000))),
                                 OptionalCodeInfo(CodeReference(26 + line_offset, 19, 26 + line_offset, 48),
                                                  'tree.DecisionTreeClassifier()'),
                                 Comparison(FunctionType),
                                 Comparison(partial))
    expected_graph.add_edge(expected_train_data, expected_estimator, arg_index=0)
    expected_graph.add_edge(expected_train_labels, expected_estimator, arg_index=1)

    if not with_code_references:
        for dag_node in expected_graph.nodes:
            dag_node.optional_code_info = None

    return expected_graph


def get_pandas_read_csv_and_dropna_code():
    """
    Get a simple code snipped that loads the adult_easy data and runs dropna
    """
    code = cleandoc("""
            import os
            import pandas as pd
            from mlwhatif.utils import get_project_root

            train_file = os.path.join(str(get_project_root()), "example_pipelines", "adult_complex", "adult_train.csv")
            raw_data = pd.read_csv(train_file)
            data = raw_data.dropna()
            relevant_data = data[['age', 'workclass', 'education']]
            """)
    return code


def run_and_assert_all_op_outputs_inspected(py_file_path, _, dag_png_path, custom_monkey_patching=None):
    """
    Execute the pipeline with a few checks and inspections.
    Assert that mlwhatif properly lets inspections inspect all DAG nodes
    """
    if custom_monkey_patching is None:
        custom_monkey_patching = []

    inspector_result = PipelineAnalyzer \
        .on_pipeline_from_py_file(py_file_path) \
        .add_custom_monkey_patching_modules(custom_monkey_patching) \
        .execute()

    save_fig_to_path(inspector_result.original_dag, dag_png_path)
    assert os.path.isfile(dag_png_path)

    for dag_node, _ in inspector_result.analysis_to_result_reports.items():
        assert dag_node.operator_info.operator != OperatorType.MISSING_OP

    return inspector_result.original_dag


def black_box_df_op():
    """
    Black box operation returning a dataframe
    """
    pandas_df = DataFrame([0, 1, 2, 3, 4], columns=['A'])
    return pandas_df


def get_test_code_with_function_def_and_for_loop():
    """
    A simple code snippet with a pandas operation in a function def and then pandas calls in a loop
    """
    test_code = cleandoc("""
            import pandas as pd

            def black_box_df_op():
                df = pd.DataFrame([0, 1], columns=['A'])
                return df
            df = black_box_df_op()
            for _ in range(2):
                df = df.dropna()
            """)
    return test_code


def visualize_dags(analysis_result, tmpdir):
    """Visualise the intermediate DAGs"""
    analysis_result.save_original_dag_to_path(os.path.join(str(tmpdir), "orig"))
    analysis_result.save_what_if_dags_to_path(os.path.join(str(tmpdir), "what-if"))
    analysis_result.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir), "what-if-optimised"))


def run_scenario_and_visualize_dags(dataset, scenario, tmpdir):
    pipeline_run_file = os.path.join(str(get_project_root()), "experiments", "end_to_end", "run_pipeline.py")
    analysis = get_analysis_for_scenario_and_dataset(scenario, dataset)
    with patch.object(sys, 'argv', ["mlwhatif", dataset, "fast_loading", "featurization_0", "logistic_regression"]):
        analysis_result_no_opt = PipelineAnalyzer \
            .on_pipeline_from_py_file(pipeline_run_file) \
            .add_custom_monkey_patching_modules([custom_monkeypatching]) \
            .add_what_if_analysis(analysis) \
            .execute()
    analysis_result_no_opt.save_original_dag_to_path(os.path.join(str(tmpdir), "with-opt-orig"))
    analysis_result_no_opt.save_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if"))
    analysis_result_no_opt.save_optimised_what_if_dags_to_path(os.path.join(str(tmpdir), "with-opt-what-if-optimised"))
    analysis_output = analysis_result_no_opt.analysis_to_result_reports[analysis]
    return analysis_output
