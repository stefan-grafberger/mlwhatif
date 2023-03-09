"""
Utility functions to visualise the extracted DAG
"""
import random
from collections import Counter
from inspect import cleandoc
from typing import List

import networkx
import numpy
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
    # pylint: disable-all
    # Memory tracking is disabled by default for now
    # from humanize import naturalsize
    # {naturalsize(node.details.optimizer_info.memory)
    #  if node.details.optimizer_info and isinstance(node.details.optimizer_info.memory, int) else ""}

    def get_new_node_label(node: DagNode):
        label = cleandoc(f"""
                {node.node_id}: {node.operator_info.operator.value} (L{node.code_location.lineno})
                {node.details.optimizer_info.shape if node.details.optimizer_info and node.details.optimizer_info.shape 
                else " - "}
                {str(numpy.round(node.details.optimizer_info.runtime, 3)) + "ms"
                if node.details.optimizer_info and isinstance(node.details.optimizer_info.runtime, (int, float)) 
                else ""}
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


def get_colored_simple_dags(dags: List[networkx.DiGraph]) -> List[networkx.DiGraph]:
    result_dags = []

    all_nodes_with_counts = Counter([(node for node in dag.nodes) for dag in dags]).items()
    shared_nodes = {node for (node, count) in all_nodes_with_counts if count >= 2}

    colors = random.choices(['#C06C84', '#355C7D', '#F67280'], k=len(dags))  # TODO: Pick enough suitable colors
    white = '#FFFFFF'
    black = '#000000'

    for dag, variant_color in zip(dags, colors):
        plan = networkx.DiGraph()

        for node in dag.nodes:
            node_id = node.node_id
            operator_name = get_pretty_operator_name(node)

            if node in shared_nodes:
                plan.add_node(node_id, operator_name=operator_name, color=black, font_color=white)
            else:
                plan.add_node(node_id, operator_name=operator_name, color=variant_color, font_color=black)

        for edge in dag.edges:
            plan.add_edge(edge[0].node_id, edge[1].node_id)

        # TODO: This is probably not needed for mlwhatif
        while True:
            nodes_to_remove = [node for node, data in plan.nodes(data=True)
                               if data['operator_name'] == 'π' and len(list(plan.successors(node))) == 0]
            if len(nodes_to_remove) == 0:
                break
            else:
                plan.remove_nodes_from(nodes_to_remove)
        result_dags.append(plan)

    return result_dags


def get_pretty_operator_name(node: DagNode) -> str:
    op_type = node.operator_info.operator
    operator_type = str(op_type).split('.')[1]

    operator_name = operator_type
    if operator_type == 'JOIN':
        operator_name = '⋈'
    elif operator_type == 'PROJECTION' or operator_type == 'PROJECTION_MODIFY':
        operator_name = 'π'
    elif operator_type == 'TRANSFORMER':
        operator_name = 'π'
    elif operator_type == 'SELECTION':
        operator_name = 'σ'
    elif operator_type == 'CONCATENATION':
        operator_name = '+'
    elif operator_type == 'DATA_SOURCE':
        operator_name = f'({node.node_id}) {node.details.description}'
    elif operator_type == 'ESTIMATOR':
        operator_name = 'Model Training'
    elif operator_type == 'SCORE':
        operator_name = 'Model Evaluation'
    elif operator_type == 'TRAIN_DATA':
        operator_name = 'X_train'
    elif operator_type == 'TRAIN_LABELS':
        operator_name = 'y_train'
    elif operator_type == 'TEST_DATA':
        operator_name = 'X_test'
    elif operator_type == 'TEST_LABELS':
        operator_name = 'y_test'

    return operator_name
