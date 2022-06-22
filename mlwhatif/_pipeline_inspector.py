"""
User-facing API for inspecting the pipeline
"""
from typing import Iterable, List

from ._inspector_result import AnalysisResults
from .analysis._what_if_analysis import WhatIfAnalysis
from .instrumentation._pipeline_executor import singleton


class PipelineInspectorBuilder:
    """
    The fluent API builder to build an inspection run
    """

    def __init__(self, notebook_path: str or None = None,
                 python_path: str or None = None,
                 python_code: str or None = None
                 ) -> None:
        self.track_code_references = True
        self.monkey_patching_modules = []
        self.notebook_path = notebook_path
        self.python_path = python_path
        self.python_code = python_code
        self.analyses = []
        self.checks = []

    def add_what_if_analysis(self, analysis: WhatIfAnalysis):
        """
        Add an analysis. TODO
        """
        self.analyses.append(analysis)
        return self

    def add_what_if_analyses(self, analyses: Iterable[WhatIfAnalysis]):
        """
        Add a list of analyses
        """
        self.analyses.extend(analyses)
        return self

    def set_code_reference_tracking(self, track_code_references: bool):
        """
        Set whether to track code references. The default is tracking them.
        """
        self.track_code_references = track_code_references
        return self

    def add_custom_monkey_patching_modules(self, module_list: List):
        """
        Add additional monkey patching modules.
        """
        self.monkey_patching_modules.extend(module_list)
        return self

    def add_custom_monkey_patching_module(self, module: any):
        """
        Add an additional monkey patching module.
        """
        self.monkey_patching_modules.append(module)
        return self

    def execute(self) -> AnalysisResults:
        """
        Instrument and execute the pipeline
        """
        return singleton.run(notebook_path=self.notebook_path,
                             python_path=self.python_path,
                             python_code=self.python_code,
                             analyses=self.analyses,
                             custom_monkey_patching=self.monkey_patching_modules)


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
