"""
Utility functions to visualise the extracted DAG
"""
from inspect import cleandoc

import networkx
from humanize import naturalsize
from networkx.drawing.nx_agraph import to_agraph

from mlwhatif.instrumentation._dag_node import DagNode


def save_fig_to_path(extracted_dag, filename):
    """
    Create a figure of the extracted DAG and save it with some filename
    """

    def get_new_node_label(node: DagNode):
        label = cleandoc(f"""
                {node.node_id}: {node.operator_info.operator.value} (L{node.code_location.lineno})
                {node.details.optimizer_info.shape if node.details.optimizer_info and node.details.optimizer_info.shape 
                else " - "}
                {str(node.details.optimizer_info.runtime) + "ms" if node.details.optimizer_info else ""}
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
