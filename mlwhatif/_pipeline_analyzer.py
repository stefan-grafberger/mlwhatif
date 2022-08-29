"""
User-facing API for inspecting the pipeline
"""
from typing import Iterable, List

from ._analysis_results import AnalysisResults, DagExtractionInfo
from .analysis._what_if_analysis import WhatIfAnalysis
from mlwhatif.execution._pipeline_executor import singleton, logger
from .optimization._query_optimization_rules import QueryOptimizationRule


class PipelineInspectorBuilder:
    """
    The fluent API builder to build an inspection run
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, notebook_path: str or None = None,
                 python_path: str or None = None,
                 python_code: str or None = None,
                 extraction_info: DagExtractionInfo or None = None
                 ) -> None:
        self._track_code_references = True
        self._monkey_patching_modules = []
        self._notebook_path = notebook_path
        self._python_path = python_path
        self._python_code = python_code
        self._extraction_info = extraction_info
        self._analyses = []
        self._checks = []
        self._prefix_original_dag = None
        self._prefix_analysis_dags = None
        self._prefix_optimised_analysis_dag = None
        self._skip_optimizer = False
        self._force_optimization_rules = None

    def add_what_if_analysis(self, analysis: WhatIfAnalysis):
        """
        Add an analysis. TODO
        """
        self._analyses.append(analysis)
        return self

    def add_what_if_analyses(self, analyses: Iterable[WhatIfAnalysis]):
        """
        Add a list of analyses
        """
        self._analyses.extend(analyses)
        return self

    def set_code_reference_tracking(self, track_code_references: bool):
        """
        Set whether to track code references. The default is tracking them.
        """
        self._track_code_references = track_code_references
        return self

    def add_custom_monkey_patching_modules(self, module_list: List):
        """
        Add additional monkey patching modules.
        """
        self._monkey_patching_modules.extend(module_list)
        return self

    def add_custom_monkey_patching_module(self, module: any):
        """
        Add an additional monkey patching module.
        """
        self._monkey_patching_modules.append(module)
        return self

    def skip_multi_query_optimization(self, skip_optimizer: False):
        """
        A convenience function for benchmarking
        """
        logger.info("The skip_multi_query_optimization function is only intended for benchmarking!")
        self._skip_optimizer = skip_optimizer
        return self

    def overwrite_optimization_rules(self, optimizations: List[QueryOptimizationRule] or None):
        """
        A convenience function for benchmarking
        """
        logger.info("The overwrite_optimization_rules function is only intended for benchmarking!")
        self._force_optimization_rules = optimizations
        return self

    def execute(self) -> AnalysisResults:
        """
        Instrument and execute the pipeline
        """
        return singleton.run(notebook_path=self._notebook_path,
                             python_path=self._python_path,
                             python_code=self._python_code,
                             extraction_info=self._extraction_info,
                             analyses=self._analyses,
                             custom_monkey_patching=self._monkey_patching_modules,
                             skip_optimizer=self._skip_optimizer,
                             force_optimization_rules=self._force_optimization_rules)


class PipelineAnalyzer:
    """
    The entry point to the fluent API to build an inspection run
    """
    @staticmethod
    def on_pipeline_from_py_file(path: str) -> PipelineInspectorBuilder:
        """Inspect a pipeline from a .py file."""
        return PipelineInspectorBuilder(python_path=path)

    @staticmethod
    def on_pipeline_from_ipynb_file(path: str) -> PipelineInspectorBuilder:
        """Inspect a pipeline from a .ipynb file."""
        return PipelineInspectorBuilder(notebook_path=path)

    @staticmethod
    def on_pipeline_from_string(code: str) -> PipelineInspectorBuilder:
        """Inspect a pipeline from a string."""
        return PipelineInspectorBuilder(python_code=code)

    @staticmethod
    def on_previously_extracted_pipeline(extraction_info: DagExtractionInfo) -> PipelineInspectorBuilder:
        """Inspect a pipeline from a string."""
        return PipelineInspectorBuilder(extraction_info=extraction_info)
