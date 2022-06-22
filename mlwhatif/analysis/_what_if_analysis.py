"""
The Interface for the What-If Analyses
"""
import abc
from typing import Iterable

import networkx


class WhatIfAnalysis(metaclass=abc.ABCMeta):
    """
    The Interface for the What-If Analyses
    """

    @property
    def analysis_id(self):
        """The Interface for the What-If Analyses"""
        return None

    @abc.abstractmethod
    def generate_plans_to_try(self, dag: networkx.DiGraph)\
            -> Iterable[networkx.DiGraph]:
        """Generate the pipeline variants to try out"""
        raise NotImplementedError

    @abc.abstractmethod
    def generate_final_report(self) -> any:
        """Get the final report after trying out the different pipeline variants"""
        raise NotImplementedError

    def __eq__(self, other):
        """What-If Analyses must implement equals"""
        return (isinstance(other, self.__class__) and
                self.analysis_id == other.analysis_id)

    def __hash__(self):
        """What-If Analyses must be hashable"""
        return hash((self.__class__.__name__, self.analysis_id))

    def __repr__(self):
        """What-If Analyses must have a str representation"""
        return "{}({})".format(self.__class__.__name__, self.analysis_id)
