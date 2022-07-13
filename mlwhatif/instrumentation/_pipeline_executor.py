"""
Instrument and executes the pipeline
"""
import ast
import logging
import time
from typing import List

import gorilla
import nbformat
import networkx
from astmonkey.transformers import ParentChildNodeTransformer
from nbconvert import PythonExporter

from ._call_capture_transformer import CallCaptureTransformer
from ._dag_node import AnalysisResult
from .. import monkeypatching
from .._inspector_result import AnalysisResults
from ..analysis._what_if_analysis import WhatIfAnalysis
from ..execution._dag_executor import DagExecutor
from ..execution._multi_query_optimizer import MultiQueryOptimizer
from ..visualisation import save_fig_to_path

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
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
    next_missing_op_id = -1
    track_code_references = True
    op_id_to_dag_node = dict()
    analyses = []
    custom_monkey_patching = []
    # TODO: Do we want to add the analysis to the key next to label to isolate analyses and avoid name clashes?
    labels_to_extracted_plan_results = dict()
    analysis_results = AnalysisResult(networkx.DiGraph(), dict())
    prefix_original_dag = None
    prefix_analysis_dags = None
    prefix_optimised_analysis_dag = None

    def run(self, *,
            notebook_path: str or None = None,
            python_path: str or None = None,
            python_code: str or None = None,
            analyses: List[WhatIfAnalysis] or None = None,
            reset_state: bool = True,
            track_code_references: bool = True,
            custom_monkey_patching: List[any] = None,
            prefix_original_dag: str or None = None,
            prefix_analysis_dags: str or None = None,
            prefix_optimised_analysis_dag: str or None = None
            ) -> AnalysisResults:
        """
        Instrument and execute the pipeline and evaluate all checks
        """
        # pylint: disable=too-many-arguments
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
        self.prefix_original_dag = prefix_original_dag
        self.prefix_analysis_dags = prefix_analysis_dags
        self.prefix_optimised_analysis_dag = prefix_optimised_analysis_dag

        logger.info(f'Running instrumented original pipeline...')
        orig_instrumented_exec_start = time.time()
        self.run_instrumented_pipeline(notebook_path, python_code, python_path)
        orig_instrumented_exec_duration = time.time() - orig_instrumented_exec_start
        logger.info(f'---RUNTIME: Original pipeline execution took {orig_instrumented_exec_duration * 1000} ms '
                    f'(including monkey-patching)')

        logger.info(f'Starting execution of {len(self.analyses)} what-if analyses...')
        self.run_what_if_analyses()

        logger.info(f'Done!')
        return AnalysisResults(self.analysis_results.dag, self.analysis_results.analysis_to_result_reports)

    def run_what_if_analyses(self):
        """
        Execute the specified what-if analyses
        """
        if self.prefix_original_dag is not None:
            save_fig_to_path(self.analysis_results.dag, f"{self.prefix_original_dag}.png")
        for analysis in self.analyses:
            logger.info(f'Start plan generation for analysis {type(analysis).__name__}...')
            plan_generation_start = time.time()
            what_if_dags = analysis.generate_plans_to_try(self.analysis_results.dag)
            plan_generation_duration = time.time() - plan_generation_start
            logger.info(f'---RUNTIME: Plan generation took {plan_generation_duration * 1000} ms')
            for dag_index, what_if_dag in enumerate(what_if_dags):
                if self.prefix_analysis_dags is not None:
                    save_fig_to_path(what_if_dag,
                                     f"{self.prefix_analysis_dags}-{type(analysis).__name__}-{dag_index}.png")

            # TODO: Potentially, we might want to also combine multiple analyses to one joint execution plan

            logger.info(f"Performing Multi-Query Optimization")
            multi_query_optimization_start = time.time()
            big_execution_dag = MultiQueryOptimizer().create_optimized_plan(what_if_dags)
            multi_query_optimization_duration = time.time() - multi_query_optimization_start
            logger.info(f'---RUNTIME: Multi-Query Optimization took {multi_query_optimization_duration * 1000} ms')
            logger.info(f"Executing generated plan")
            if self.prefix_optimised_analysis_dag is not None:
                save_fig_to_path(big_execution_dag, f"{self.prefix_optimised_analysis_dag}.png")

            execution_start = time.time()
            DagExecutor().execute(big_execution_dag)
            # for what_if_dag in what_if_dags:
            #     DagExecutor().execute(what_if_dag)
            execution_duration = time.time() - execution_start
            logger.info(f'---RUNTIME: Execution took {execution_duration * 1000} ms')

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
        exec(compile(parsed_modified_ast, filename=self.source_code_path, mode="exec"), PipelineExecutor.script_scope)

    def get_next_op_id(self):
        """
        Each operator in the DAG gets a consecutive unique id
        """
        current_op_id = self.next_op_id
        self.next_op_id += 1
        return current_op_id

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
        self.next_missing_op_id = -1
        self.track_code_references = True
        self.op_id_to_dag_node = dict()
        self.analysis_results = AnalysisResult(networkx.DiGraph(), dict())
        self.analyses = []
        self.labels_to_extracted_plan_results = dict()
        self.custom_monkey_patching = []
        self.prefix_original_dag = None
        self.prefix_analysis_dags = None
        self.prefix_optimised_analysis_dag = None

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
        func_import_node = ast.ImportFrom(module='mlwhatif.instrumentation._pipeline_executor',
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
    logger.info(f"Applying Monkey-Patches")
    logger.info(f"(The first time this is called, this can take a bit because all of the libraries need to be "
                f"loaded by Python, but this cost is present anyway if those libraries are used.)")
    monkey_patch_start = time.time()
    patch_sources = get_monkey_patching_patch_sources()
    patches = gorilla.find_patches(patch_sources)
    for patch in patches:
        gorilla.apply(patch)
    monkey_patch_duration = time.time() - monkey_patch_start
    logger.info(f'---RUNTIME: Monkey-Patching took {monkey_patch_duration * 1000} ms')


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
