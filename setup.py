"""Setup PyFlyt."""
import os

from setuptools import setup


def package_files(directory):
    """package_files.

    Args:
        directory: directory of non-python files
    """
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


setup(
    name="PyFlyt",
    package_data={"PyFlyt": package_files("PyFlyt/models/")},
)
