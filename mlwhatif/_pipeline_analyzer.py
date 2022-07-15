"""
User-facing API for inspecting the pipeline
"""
from typing import Iterable, List

from ._inspector_result import AnalysisResults
from .analysis._what_if_analysis import WhatIfAnalysis
from .instrumentation._pipeline_executor import singleton, logger


class PipelineInspectorBuilder:
    """
    The fluent API builder to build an inspection run
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, notebook_path: str or None = None,
                 python_path: str or None = None,
                 python_code: str or None = None
                 ) -> None:
        self._track_code_references = True
        self._monkey_patching_modules = []
        self._notebook_path = notebook_path
        self._python_path = python_path
        self._python_code = python_code
        self._analyses = []
        self._checks = []
        self._prefix_original_dag = None
        self._prefix_analysis_dags = None
        self._prefix_optimised_analysis_dag = None
        self._skip_optimizer = False

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

    def save_original_dag_to_path(self, path: str):
        """
        Save the extracted original DAG to a file
        """
        self._prefix_original_dag = path
        return self

    def save_what_if_dags_to_path(self, prefix_analysis_dags: str):
        """
        Save the generated What-If DAGs to a file
        """
        self._prefix_analysis_dags = prefix_analysis_dags
        return self

    def save_optimised_what_if_dags_to_path(self, prefix_optimised_analysis_dag: str):
        """
        Save the extracted original DAG to a file
        """
        self._prefix_optimised_analysis_dag = prefix_optimised_analysis_dag
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
        A convenience function to
        """
        logger.info("The skip_multi_query_optimization function is only intended for benchmarking!")
        self._skip_optimizer = skip_optimizer
        return self

    def execute(self) -> AnalysisResults:
        """
        Instrument and execute the pipeline
        """
        return singleton.run(notebook_path=self._notebook_path,
                             python_path=self._python_path,
                             python_code=self._python_code,
                             analyses=self._analyses,
                             custom_monkey_patching=self._monkey_patching_modules,
                             prefix_original_dag=self._prefix_original_dag,
                             prefix_analysis_dags=self._prefix_analysis_dags,
                             prefix_optimised_analysis_dag=self._prefix_optimised_analysis_dag,
                             skip_optimizer=self._skip_optimizer)


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
