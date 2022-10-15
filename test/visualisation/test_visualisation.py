"""
Tests whether the visualisation of the resulting DAG works
"""
import os

from mlmq.utils import get_project_root
from mlmq.visualisation import save_fig_to_path, get_dag_as_pretty_string
from mlmq.testing._testing_helper_utils import get_expected_dag_adult_easy


def test_save_fig_to_path():
    """
    Tests whether the .py version of the inspector works
    """
    extracted_dag = get_expected_dag_adult_easy("<string-source>")

    filename = os.path.join(str(get_project_root()), "test", "visualisation", "test_save_fig_to_path.png")
    save_fig_to_path(extracted_dag, filename)

    assert os.path.isfile(filename)


def test_get_dag_as_pretty_string():
    """
    Tests whether the .py version of the inspector works
    """
    extracted_dag = get_expected_dag_adult_easy("<string-source>")

    pretty_string = get_dag_as_pretty_string(extracted_dag)

    print(pretty_string)
