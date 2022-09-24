"""
Some useful utils for the project
"""
from pathlib import Path

import numpy


def get_project_root() -> Path:
    """Returns the project root folder."""
    return Path(__file__).parent.parent.parent


def decode_image(img_str):
    """Converter for loading images as numpy arrays with pandas."""
    return numpy.array([int(val) for val in img_str.split(':')])
