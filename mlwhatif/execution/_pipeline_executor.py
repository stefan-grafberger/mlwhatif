"""
Instrument and executes the pipeline
"""
# TODO: At some point, this should be split into two files, one for mere orchestration, one for instrumentation
import ast
import logging
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from copy import copy, deepcopy
from io import StringIO
from typing import List

import gorilla
import nbformat
import networkx
from astmonkey.transformers import ParentChildNodeTransformer
from nbconvert import PythonExporter

from mlwhatif.instrumentation._call_capture_transformer import CallCaptureTransformer
from mlwhatif import monkeypatching
from mlwhatif.instrumentation._operator_types import OperatorType
from mlwhatif._analysis_results import AnalysisResults, RuntimeInfo, DagExtractionInfo
from mlwhatif.analysis._what_if_analysis import WhatIfAnalysis
from mlwhatif.execution._dag_executor import DagExecutor
from mlwhatif.optimization._multi_query_optimizer import MultiQueryOptimizer
from mlwhatif.optimization._query_optimization_rules import QueryOptimizationRule

logging.basicConfig(format='%(asctime)s %(levelname)-5s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')
for _ in ("gensim", "tensorflow", "h5py"):
    logging.getLogger(_).setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """
    Internal class to instrument and execute pipelines
    """
    # pylint: disable=too-many-instance-attributes

    source_code_path = None
    source_code = None
    script_scope = {}
    lineno_next_call_or_subscript = -1
    col_offset_next_call_or_subscript = -1
    end_lineno_next_call_or_subscript = -1
    end_col_offset_next_call_or_subscript = -1
    next_op_id = 0
    next_patch_id = 0
    next_missing_op_id = -1
    track_code_references = True
    op_id_to_dag_node = dict()
    analyses = []
    custom_monkey_patching = []
    # TODO: Do we want to add the analysis to the key next to label to isolate analyses and avoid name clashes?
    original_pipeline_labels_to_extracted_plan_results = dict()
    labels_to_extracted_plan_results = dict()
    analysis_results = AnalysisResults(dict(), networkx.DiGraph(), [], networkx.DiGraph(),
                                       RuntimeInfo(0, 0, 0, 0, None, None, 0, 0, 0, 0, 0, 0, 0),
                                       DagExtractionInfo(networkx.DiGraph(), dict(), 0, 0, 0), None, defaultdict(list))
    monkey_patch_duration = 0
    skip_optimizer = False
    force_optimization_rules = None
    estimate_only = False
    operators_to_runtime_during_analysis = []
    use_dfs_exec_strategy = False
    disable_monkey_patching = False

    def run(self, *,
            notebook_path: str or None = None,
            python_path: str or None = None,
            python_code: str or None = None,
            extraction_info: DagExtractionInfo or None = None,
            analyses: List[WhatIfAnalysis] or None = None,
            reset_state: bool = True,
            track_code_references: bool = True,
            custom_monkey_patching: List[any] = None,
            skip_optimizer=False,
            force_optimization_rules: List[QueryOptimizationRule] or None = None,
            use_dfs_exec_strategy: bool = False,
            estimate_only=False
            ) -> AnalysisResults:
        """
        Instrument and execute the pipeline and evaluate all checks
        """
        # pylint: disable=too-many-arguments,too-many-locals
        self.analysis_results.pipeline_executor = self
        if reset_state:
            # reset_state=False should only be used internally for performance experiments etc!
            # It does not ensure the same inspections are still used as args etc.
            self.reset()

        if custom_monkey_patching is None:
            custom_monkey_patching = []
        if analyses is None:
            analyses = []

        self.track_code_references = track_code_references
        self.custom_monkey_patching = custom_monkey_patching
        self.analyses = analyses
        self.skip_optimizer = skip_optimizer
        self.force_optimization_rules = force_optimization_rules
        self.estimate_only = estimate_only
        self.use_dfs_exec_strategy = use_dfs_exec_strategy

        if extraction_info is None:
            logger.info(f'Running instrumented original pipeline...')
            orig_instrumented_exec_start = time.time()
            sys.stdout.flush()
            stdout_output = StringIO()
            with redirect_stdout(stdout_output):
                self.run_instrumented_pipeline(notebook_path, python_code, python_path)
            # TODO: Do we ever need the captured output from the original pipeline version?
            #  Maybe this gets relevant once we add the DAG as input to mlwhat in case there are multiple executions
            # captured_output = stdout_output.getvalue()
            orig_instrumented_exec_duration = (time.time() - orig_instrumented_exec_start -
                                               singleton.monkey_patch_duration)
            self.analysis_results.runtime_info.original_pipeline_without_importing_and_monkeypatching = \
                orig_instrumented_exec_duration * 1000

            original_estimator_runtime = [node.details.optimizer_info.runtime
                                          for node in self.analysis_results.original_dag.nodes
                                          if node.operator_info.operator == OperatorType.ESTIMATOR]
            self.analysis_results.runtime_info.original_model_training = sum(original_estimator_runtime)
            train_data_nodes = [node for node in self.analysis_results.original_dag.nodes
                                if node.operator_info.operator == OperatorType.TRAIN_DATA]
            if len(train_data_nodes) != 0:
                train_data_node = train_data_nodes[0]
                self.analysis_results.runtime_info.original_pipeline_train_data_shape = \
                    train_data_node.details.optimizer_info.shape
            test_data_nodes = [node for node in self.analysis_results.original_dag.nodes
                               if node.operator_info.operator == OperatorType.TEST_DATA]
            if len(test_data_nodes) != 0:
                test_data_node = test_data_nodes[0]
                self.analysis_results.runtime_info.original_pipeline_test_data_shape = \
                    test_data_node.details.optimizer_info.shape
            # FIXME: Training Data Matrix shape
            logger.info(f'---RUNTIME: Original pipeline execution took {orig_instrumented_exec_duration * 1000} ms '
                        f'(excluding imports and monkey-patching)')
        else:
            logger.info(f'Reusing DAG extraction results results from previously instrumented pipeline...')
            self.analysis_results.original_dag = extraction_info.original_dag.copy()
            self.original_pipeline_labels_to_extracted_plan_results = \
                extraction_info.original_pipeline_labels_to_extracted_plan_results.copy()
            self.analysis_results.runtime_info.original_pipeline_without_importing_and_monkeypatching = None
            self.next_op_id = extraction_info.next_op_id
            self.next_patch_id = extraction_info.next_patch_id
            self.next_missing_op_id = extraction_info.next_missing_op_id

        logger.info(f'Starting execution of {len(self.analyses)} what-if analyses...')
        self.run_what_if_analyses()

        self.analysis_results.dag_extraction_info = DagExtractionInfo(
            self.analysis_results.original_dag.copy(), self.original_pipeline_labels_to_extracted_plan_results.copy(),
            self.next_op_id, self.next_patch_id, self.next_missing_op_id)
        logger.info(f'Done!')
        return self.analysis_results

    def run_what_if_analyses(self):
        """
        Execute the specified what-if analyses
        """
        for analysis in self.analyses:
            logger.info(f'Start plan generation for analysis {type(analysis).__name__}...')
            plan_generation_start = time.time()
            for patches in analysis.generate_plans_to_try(self.analysis_results.original_dag):
                self.analysis_results.what_if_dags.append((patches, networkx.DiGraph()))
                # This part is only necessary for visualising what happens internally, we should add a flag to only
                #  enable this when actually needed
                unoptimised = self.analysis_results.intermediate_stages["unoptimised_variants"]
                orig_dag_to_patch = self.analysis_results.original_dag.copy()
                copy_for_patches = copy(self)
                for patch in copy(patches):
                    patch.apply(orig_dag_to_patch, copy_for_patches)
                unoptimised.append(orig_dag_to_patch)
                self.analysis_results.intermediate_stages["unoptimised_variants"] = unoptimised
            plan_generation_duration = time.time() - plan_generation_start
            logger.info(f'---RUNTIME: Plan generation took {plan_generation_duration * 1000} ms')
            self.analysis_results.runtime_info.what_if_plan_generation = plan_generation_duration * 1000

        # TODO: Add try catch statements so we can see intermediate DAGs even if something goes wrong for debugging
        MultiQueryOptimizer(self, self.force_optimization_rules) \
            .create_optimized_plan(self.analysis_results, self.skip_optimizer)

        if self.estimate_only is False:
            logger.info(f"Executing generated plans")
            execution_start = time.time()
            if self.skip_optimizer is False:
                DagExecutor(self).execute(self.analysis_results.combined_optimized_dag, self.use_dfs_exec_strategy)
            else:
                for _, what_if_dag in self.analysis_results.what_if_dags:
                    DagExecutor(self).execute(what_if_dag, self.use_dfs_exec_strategy)
            execution_duration = time.time() - execution_start
            logger.info(f'---RUNTIME: Execution took {execution_duration * 1000} ms')
            self.analysis_results.runtime_info.what_if_execution = execution_duration * 1000

            analysis_estimator_runtimes = [optimizer_info.runtime
                                           for node, optimizer_info in self.operators_to_runtime_during_analysis
                                           if node.operator_info.operator == OperatorType.ESTIMATOR]
            self.analysis_results.runtime_info.what_if_execution_combined_model_training = sum(
                analysis_estimator_runtimes)
            # Some debugging code to look at actual executon time of different operators in optimized plan
            # ops_with_runtimes = [(operator, optimizer_info.runtime) for operator, optimizer_info
            #                      in self.operators_to_runtime_during_analysis]
            # ops_with_runtimes.sort(key=lambda tuple: tuple[1], reverse=True)
            # ops_with_runtimes = [(op.details.description, runtime) for op, runtime in ops_with_runtimes]
            self.labels_to_extracted_plan_results.update(self.original_pipeline_labels_to_extracted_plan_results)
            for analysis in self.analyses:
                report = analysis.generate_final_report(self.labels_to_extracted_plan_results)
                self.analysis_results.analysis_to_result_reports[analysis] = report

    def run_instrumented_pipeline(self, notebook_path, python_code, python_path):
        """
        Instrument and execute the pipeline
        """
        # pylint: disable=no-self-use, too-many-locals
        self.source_code, self.source_code_path = self.load_source_code(notebook_path, python_path, python_code)
        parsed_ast = ast.parse(self.source_code)
        parsed_modified_ast = self.instrument_pipeline(parsed_ast, self.track_code_references)
        exec(compile(parsed_modified_ast, filename=self.source_code_path, mode="exec"), self.script_scope)

    def get_next_op_id(self):
        """
        Each operator in the DAG gets a consecutive unique id
        """
        current_op_id = self.next_op_id
        self.next_op_id += 1
        return current_op_id

    def get_next_patch_id(self):
        """
        Each operator in the DAG gets a consecutive unique id
        """
        current_patch_id = self.next_patch_id
        self.next_patch_id += 1
        return current_patch_id

    def get_next_missing_op_id(self):
        """
        Each unknown operator in the DAG gets a consecutive unique negative id
        """
        current_missing_op_id = self.next_missing_op_id
        self.next_missing_op_id -= 1
        return current_missing_op_id

    def reset(self):
        """
        Reset all attributes in the singleton object. This can be used when there are multiple repeated calls to mlwhatif
        """
        self.source_code_path = None
        self.source_code = None
        self.script_scope = {}
        self.lineno_next_call_or_subscript = -1
        self.col_offset_next_call_or_subscript = -1
        self.end_lineno_next_call_or_subscript = -1
        self.end_col_offset_next_call_or_subscript = -1
        self.next_op_id = 0
        self.next_patch_id = 0
        self.next_missing_op_id = -1
        self.track_code_references = True
        self.op_id_to_dag_node = dict()
        self.analysis_results = AnalysisResults(dict(), networkx.DiGraph(), [], networkx.DiGraph(),
                                                RuntimeInfo(0, 0, 0, 0, None, None, 0, 0, 0, 0, 0, 0, 0),
                                                DagExtractionInfo(networkx.DiGraph(), dict(), 0, 0, 0), self,
                                                defaultdict(list))
        self.analyses = []
        self.original_pipeline_labels_to_extracted_plan_results = dict()
        self.labels_to_extracted_plan_results = dict()
        self.custom_monkey_patching = []
        self.monkey_patch_duration = 0
        self.skip_optimizer = False
        self.force_optimization_rules = None
        self.estimate_only = False
        self.operators_to_runtime_during_analysis = []
        self.use_dfs_exec_strategy = False
        self.disable_monkey_patching = False

    @staticmethod
    def instrument_pipeline(parsed_ast, track_code_references):
        """
        Instrument the pipeline AST to instrument function calls
        """
        # insert set_code_reference calls
        if track_code_references:
            # Needed to get the parent assign node for subscript assigns.
            #  Without this, "pandas_df['baz'] = baz + 1" would only be "pandas_df['baz']"
            parent_child_transformer = ParentChildNodeTransformer()
            parsed_ast = parent_child_transformer.visit(parsed_ast)
            call_capture_transformer = CallCaptureTransformer()
            parsed_ast = call_capture_transformer.visit(parsed_ast)
            parsed_ast = ast.fix_missing_locations(parsed_ast)

        # from mlinspect2._pipeline_executor import set_code_reference, monkey_patch
        func_import_node = ast.ImportFrom(module='mlwhatif.execution._pipeline_executor',
                                          names=[ast.alias(name='set_code_reference_call', asname=None),
                                                 ast.alias(name='set_code_reference_subscript', asname=None),
                                                 ast.alias(name='monkey_patch', asname=None),
                                                 ast.alias(name='undo_monkey_patch', asname=None)],
                                          level=0)
        parsed_ast.body.insert(0, func_import_node)

        # monkey_patch()
        inspect_import_node = ast.Expr(value=ast.Call(
            func=ast.Name(id='monkey_patch', ctx=ast.Load()), args=[], keywords=[]))
        parsed_ast.body.insert(1, inspect_import_node)
        # undo_monkey_patch()
        inspect_import_node = ast.Expr(value=ast.Call(
            func=ast.Name(id='undo_monkey_patch', ctx=ast.Load()), args=[], keywords=[]))
        parsed_ast.body.append(inspect_import_node)

        parsed_ast = ast.fix_missing_locations(parsed_ast)

        return parsed_ast

    @staticmethod
    def load_source_code(notebook_path, python_path, python_code):
        """
        Load the pipeline source code from the specified source
        """
        sources = [notebook_path, python_path, python_code]
        assert sum(source is not None for source in sources) == 1
        if python_path is not None:
            with open(python_path) as file:
                source_code = file.read()
            source_code_path = python_path
        elif notebook_path is not None:
            with open(notebook_path) as file:
                notebook = nbformat.reads(file.read(), nbformat.NO_CONVERT)
                exporter = PythonExporter()
                source_code, _ = exporter.from_notebook_node(notebook)
            source_code_path = notebook_path
        elif python_code is not None:
            source_code = python_code
            source_code_path = "<string-source>"
        else:
            assert False
        return source_code, source_code_path


# How we instrument the calls

# This instance works as our singleton: we avoid to pass the class instance to the instrumented
# pipeline. This keeps the DAG nodes to be inserted very simple.
singleton = PipelineExecutor()


def set_code_reference_call(lineno, col_offset, end_lineno, end_col_offset, **kwargs):
    """
    Method that gets injected into the pipeline code
    """
    # pylint: disable=too-many-arguments
    singleton.lineno_next_call_or_subscript = lineno
    singleton.col_offset_next_call_or_subscript = col_offset
    singleton.end_lineno_next_call_or_subscript = end_lineno
    singleton.end_col_offset_next_call_or_subscript = end_col_offset
    return kwargs


def set_code_reference_subscript(lineno, col_offset, end_lineno, end_col_offset, arg):
    """
    Method that gets injected into the pipeline code
    """
    # pylint: disable=too-many-arguments
    singleton.lineno_next_call_or_subscript = lineno
    singleton.col_offset_next_call_or_subscript = col_offset
    singleton.end_lineno_next_call_or_subscript = end_lineno
    singleton.end_col_offset_next_call_or_subscript = end_col_offset
    return arg


def monkey_patch():
    """
    Function that does the actual monkey patching
    """
    # The first time this is called, this can take a bit because all of the libraries need to be
    #  loaded by Python, but this cost is present anyway if those libraries are used.
    #  Because of this, we need to be careful how we fair benchmarking.
    logger.info(f"Importing libraries and monkey-patching them... (Imports are slow if not in sys.modules cache yet!)")
    monkey_patch_start = time.time()
    patch_sources = get_monkey_patching_patch_sources()
    patches = gorilla.find_patches(patch_sources)
    for patch in patches:
        gorilla.apply(patch)
    singleton.monkey_patch_duration = time.time() - monkey_patch_start
    logger.info(f'---RUNTIME: Importing and monkey-patching took {singleton.monkey_patch_duration * 1000} ms')
    singleton.analysis_results.runtime_info.original_pipeline_importing_and_monkeypatching = singleton.monkey_patch_duration * 1000


def undo_monkey_patch():
    """
    Function that does the actual monkey patching
    """
    patch_sources = get_monkey_patching_patch_sources()
    patches = gorilla.find_patches(patch_sources)
    for patch in patches:
        gorilla.revert(patch)


def get_monkey_patching_patch_sources():
    """
    Get monkey patches provided by mlwhatif and custom patches provided by the user
    """
    patch_sources = [monkeypatching]
    patch_sources.extend(singleton.custom_monkey_patching)
    return patch_sources
