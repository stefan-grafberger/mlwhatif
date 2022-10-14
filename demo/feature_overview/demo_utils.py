# pylint: disable-all
import os

from IPython.core.display import Image

from mlwhatif.utils import get_project_root

EXAMPLE_ORIGINAL_PLAN_PATH = os.path.join(str(get_project_root()), "demo", "feature_overview",
                                          "example-orig")
EXAMPLE_OPTIMIZED_PLAN_PATH = os.path.join(str(get_project_root()), "demo", "feature_overview",
                                           "example-optimised")


def display_paper_figure():
    PAPER_IMG = os.path.join(str(get_project_root()), "demo", "feature_overview",
                             "paper_example_image.png")

    Image(filename=f"{PAPER_IMG}")


def save_plans_to_disk(analysis_result):
    analysis_result.save_original_dag_to_path(EXAMPLE_ORIGINAL_PLAN_PATH)
    analysis_result.save_optimised_what_if_dags_to_path(EXAMPLE_OPTIMIZED_PLAN_PATH)
    return f"{EXAMPLE_ORIGINAL_PLAN_PATH}.png", f"{EXAMPLE_OPTIMIZED_PLAN_PATH}.png"
