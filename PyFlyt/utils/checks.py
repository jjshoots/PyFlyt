"""Common checks."""
import warnings

import pybullet as p
from gymnasium.utils import colorize


def check_numpy():
    """Checks that numpy is installed."""
    if not p.isNumpyEnabled():
        warnings.warn(
            colorize(
                (
                    "PyBullet is not installed properly with Numpy functionality! This will "
                    "result in a significant performance hit when vision is used.\n\n"
                    "You can fix this by installing Numpy again and rebuilding PyBullet:\n"
                    "\tpip3 uninstall pybullet -y\n"
                    "\tpip3 install numpy\n"
                    "\tpip3 install pybullet --no-cache-dir"
                ),
                color="yellow",
            ),
            category=RuntimeWarning,
        )
