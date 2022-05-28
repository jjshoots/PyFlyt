from setuptools import setup


setup(
    name="PyFlyt",
    version="0.0.4",
    author="jjshoots",
    author_email="taijunjet@hotmail.com",
    description="Freestyle Quadcopter Flight in Pybullet with Gym and (soon) PettingZoo APIs",
    url="https://github.com/jjshoots/PyFlyt",
    long_description="# [Docs](https://github.com/jjshoots/PyFlyt/blob/master/readme.md)",
    long_description_content_type="text/markdown",
    keywords=["Reinforcement Learning", "UAVs", "drones", "Quadcopter", "AI", "Gym", "PettingZoo"],
    python_requires=">=3.8, <3.11",
    install_requires=[
        "wheel",
        "gym",
        "numpy",
        "pybullet",
        "cflib",
        "cfclient",
        "scipy",
    ],
)
