from setuptools import setup

def get_version():
    """Gets the pettingzoo version."""
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
    author="Jet",
    author_email="taijunjet@hotmail.com",
    description="Freestyle Quadcopter Flight in Pybullet with Gym and (soon) PettingZoo APIs",
    url="https://github.com/jjshoots/PyFlyt",
    license_files=("LICENSE.txt"),
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
    python_requires=">=3.7",
    packages=["wingman"],
    include_package_data=True,
    install_requires=[
        "wheel",
        "matplotlib",
        "gymnasium",
        "numpy",
        "pybullet",
        "cflib",
        "cfclient",
        "scipy",
    ],
)
