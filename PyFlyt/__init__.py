"""PyFlyt - Multi UAV simulation environment for reinforcement learning research."""

# test whether numpy is enabled for simulation


def test_numpy():
    """Tests whether pybullet was installed with Numpy."""
    import pybullet as p
    import pybullet_data

    p.connect(p.DIRECT)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.setGravity(0, 0, -10)

    _, _, rgbaImg, _, _ = p.getCameraImage(
        height=10,
        width=10,
        viewMatrix=(
            0.0,
            0.3420201539993286,
            -0.9396926760673523,
            0.0,
            -1.0,
            0.0,
            -0.0,
            0.0,
            0.0,
            0.9396926760673523,
            0.3420201539993286,
            0.0,
            -0.1599999964237213,
            -4.414617538452148,
            9.205257415771484,
            1.0,
        ),
        projectionMatrix=(
            0.9999999403953552,
            0.0,
            0.0,
            0.0,
            0.0,
            0.9999999403953552,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.0007846355438232,
            -1.0,
            0.0,
            0.0,
            -0.20007847249507904,
            0.0,
        ),
    )

    try:
        rgbaImg.reshape(-1)
    except Exception:
        raise RuntimeError(
            "PyBullet as not installed properly with Numpy functionality,\n"
            "Please fix this by installing Numpy again, then rebuilding PyBullet:\n"
            "\tpip3 uninstall pybullet -y"
            "\tpip3 install numpy"
            "\tpip3 install pybullet --no-cache-dir"
        )

    p.disconnect()
