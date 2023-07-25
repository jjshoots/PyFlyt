"""PyFlyt - Multi UAV simulation environment for reinforcement learning research."""
import pybullet

# Throw error if pybullet is not installed with numpy support.
if not pybullet.isNumpyEnabled():
    raise RuntimeWarning(
        "PyBullet is not installed properly with Numpy functionality,\n"
        "This will result in a significant performance hit when using vision.\n'"
        "Please fix this by installing Numpy again, then rebuilding PyBullet:\n"
        "\tpip3 uninstall pybullet -y\n"
        "\tpip3 install numpy\n"
        "\tpip3 install pybullet --no-cache-dir\n"
    )
