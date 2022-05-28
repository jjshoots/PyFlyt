from setuptools import setup

with open("readme.md") as fh:
    long_description = ""
    header_count = 0
    for line in fh:
        if line.startswith("##"):
            header_count += 1
        else:
            long_description += line

setup(
    name="PyFlyt",
    version="0.0.1",
    author="jjshoots",
    author_email="taijunjet@hotmail.com",
    description="Freestyle Quadcopter Flight in Pybullet with Gym and (soon) PettingZoo APIs",
    url="https://github.com/jjshoots/PyFlyt",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["Reinforcement Learning", "UAVs", "drones", "Quadcopter", "AI", "Gym", "PettingZoo"],
    python_requires=">=3.8, <3.11",
    install_requires=[
        "gym",
        "numpy",
        "pybullet",
        "cflib",
        "cfclient",
        "scipy",
    ],
)
