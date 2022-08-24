"""
Utility functions to visualise the extracted DAG
"""
from inspect import cleandoc

import networkx
import numpy
from humanize import naturalsize
from networkx.drawing.nx_agraph import to_agraph

from mlwhatif.instrumentation._dag_node import DagNode


def _save_what_if_dags_as_figs_to_path(prefix_analysis_dags, what_if_dags):
    """Store the what-if DAGs as figs."""
    for dag_index, what_if_dag in enumerate(what_if_dags):
        if prefix_analysis_dags is not None:
            save_fig_to_path(what_if_dag, f"{prefix_analysis_dags}-{dag_index}.png")


def save_fig_to_path(extracted_dag, filename):
    """
    Create a figure of the extracted DAG and save it with some filename
    """

    def get_new_node_label(node: DagNode):
        label = cleandoc(f"""
                {node.node_id}: {node.operator_info.operator.value} (L{node.code_location.lineno})
                {node.details.optimizer_info.shape if node.details.optimizer_info and node.details.optimizer_info.shape 
                else " - "}
                {str(numpy.round(node.details.optimizer_info.runtime, 3)) + "ms"
                if node.details.optimizer_info and isinstance(node.details.optimizer_info.runtime, (int, float)) 
                else ""}
                {naturalsize(node.details.optimizer_info.memory) if node.details.optimizer_info
                and isinstance(node.details.optimizer_info.memory, int) else ""}
                {node.details.description or ""}
                """)
        return label

    # noinspection PyTypeChecker
    extracted_dag = networkx.relabel_nodes(extracted_dag, get_new_node_label)

    agraph = to_agraph(extracted_dag)
    agraph.layout('dot')
    agraph.draw(filename)


def get_dag_as_pretty_string(extracted_dag):
    """
    Create a figure of the extracted DAG and save it with some filename
    """

    def get_new_node_label(node: DagNode):
        description = ""
        if node.details.description:
            description = "({})".format(node.details.description)

        label = "{}{}".format(node.operator_info.operator.value, description)
        return label

    # noinspection PyTypeChecker
    extracted_dag = networkx.relabel_nodes(extracted_dag, get_new_node_label)

    agraph = to_agraph(extracted_dag)
    return agraph.to_string()
