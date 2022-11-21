import setuptools
from setuptools import setup

setup(
    name="PyFlyt",
    version="0.2.10",
    author="jjshoots",
    author_email="taijunjet@hotmail.com",
    description="Freestyle Quadcopter Flight in Pybullet with Gym and (soon) PettingZoo APIs",
    url="https://github.com/jjshoots/PyFlyt",
    long_description="# [Docs](https://github.com/jjshoots/PyFlyt/blob/master/readme.md)",
    long_description_content_type="text/markdown",
    keywords=[
        "Reinforcement Learning",
        "UAVs",
        "drones",
        "Quadcopter",
        "AI",
        "Gym",
        "PettingZoo",
    ],
    python_requires=">=3.8, <3.11",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        "matplotlib",
        "gymnasium",
        "numpy",
        "pybullet",
        "cflib",
        "cfclient",
        "scipy",
    ],
)
