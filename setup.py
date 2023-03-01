"""Setup PyFlyt."""
import os

from setuptools import find_namespace_packages, setup


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


extra_files = package_files("PyFlyt/models/")


def get_version():
    """Gets the PyFlyt version."""
    path = "pyproject.toml"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("version"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


setup(
    name="PyFlyt",
    version=get_version(),
    package_data={"PyFlyt": extra_files},
)
