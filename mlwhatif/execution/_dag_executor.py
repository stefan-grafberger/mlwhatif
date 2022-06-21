"""
The engine to execute the inspections
"""
import dataclasses

import networkx

from mlwhatif import DagNode, OperatorType


@dataclasses.dataclass(frozen=True)
class DagNodeResult:
    """ Holds the result from a processing_func after a DagNode has been executed """
    node_id: int = dataclasses.field(default=None)
    dag_node: DagNode = dataclasses.field(default=None)
    result_df: any = dataclasses.field(hash=False, default=None)


class DagExecutor:
    """ Executes given DAGs using the processing_funcs started with each DagNode """

    def execute(self, dag: networkx.DiGraph):
        """ Execute a given input DAG """
        # TODO: Currently, this returns the final result from some DagNode without children but in the future,
        #  we want to have a mechanism to store the results from selected DagNodes with a label in some result map
        dag = dag.copy()

        def execute_node(current_node: DagNode):
            if current_node.operator_info.operator == OperatorType.MISSING_OP:
                raise Exception("Missing Ops not supported currently!")
            inputs = self.get_required_values(dag, current_node)
            result_df = current_node.processing_func(*inputs)
            result = self.replace_node_with_result(dag, current_node, result_df)
            return result

        self.traverse_graph_and_process_nodes(dag, execute_node)

        # TODO: Build mechanism instead to select what to extract
        final_result_value = [node for node, out_degree in dag.out_degree() if out_degree == 0][0]
        return final_result_value.result_df

    @staticmethod
    def traverse_graph_and_process_nodes(graph: networkx.DiGraph, func):
        """
        Traverse the DAG node by node from top to bottom
        """
        current_nodes = [node for node in graph.nodes if len(list(graph.predecessors(node))) == 0]
        processed_nodes = set()
        while len(current_nodes) != 0:
            node = current_nodes.pop(0)
            processed_nodes.add(node.node_id)
            result_node = func(node)
            if result_node is not None:
                children = list(graph.successors(result_node))
                # Nodes can have multiple parents, only want to process them once we processed all parents
                for child in children:
                    if child.node_id not in processed_nodes:
                        predecessors = [predecessor.node_id for predecessor in graph.predecessors(child)]
                        if processed_nodes.issuperset(predecessors):
                            current_nodes.append(child)

        return graph

    @staticmethod
    def replace_node_with_result(sub_dag, dag_node: DagNode, result_df):
        """ This replaces a DAG node with the result from its processing_func """
        new_value_node = DagNodeResult(dag_node.node_id, dag_node, result_df)
        sub_dag.add_node(new_value_node)
        for parent_node in sub_dag.predecessors(dag_node):
            edge_data = sub_dag.get_edge_data(parent_node, dag_node)
            sub_dag.add_edge(parent_node, new_value_node, **edge_data)
        for child_node in sub_dag.successors(dag_node):
            edge_data = sub_dag.get_edge_data(dag_node, child_node)
            sub_dag.add_edge(new_value_node, child_node, **edge_data)
        sub_dag.remove_node(dag_node)
        return new_value_node

    @staticmethod
    def get_required_values(sub_dag: networkx.DiGraph, current_node: DagNode):
        """
        This gets all required input values for the processing_func of a dag_node from its DagNode parents.
        Deletes results from parents that are no longer required.
        """
        required_df_values = []
        parent_nodes = list(sub_dag.predecessors(current_node))
        if len(parent_nodes) > 1:
            parent_nodes_with_arg_index = [(parent_node, sub_dag.get_edge_data(parent_node, current_node))
                                           for parent_node in parent_nodes]
            sorted_parent_nodes_with_arg_index = sorted(parent_nodes_with_arg_index, key=lambda x: x[1]['arg_index'])
            parent_nodes = [node_parent[0] for node_parent in sorted_parent_nodes_with_arg_index]

        for parent_node in parent_nodes:
            assert isinstance(parent_node, DagNodeResult)
            df_value = parent_node.result_df
            sub_dag.remove_edge(parent_node, current_node)
            # We want to enable garbage collection of value_node if we no longer need to keep the value around
            if not list(sub_dag.successors(parent_node)):
                sub_dag.remove_node(parent_node)
            required_df_values.append(df_value)
        return required_df_values
