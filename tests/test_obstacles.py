"""Tests for the static obstacle functionality in the Aviary."""

from __future__ import annotations

import os

import numpy as np
import pytest

from PyFlyt.core import Aviary
from PyFlyt.core.abstractions import Obstacle


def _make_env() -> Aviary:
    """Builds a small single-drone Aviary used across the obstacle tests."""
    start_pos = np.array([[0.0, 0.0, 1.0]])
    start_orn = np.array([[0.0, 0.0, 0.0]])
    return Aviary(
        start_pos=start_pos,
        start_orn=start_orn,
        render=False,
        drone_type="quadx",
    )


@pytest.mark.parametrize("name", ["cube", "cylinder", "sphere"])
def test_builtin_obstacle_loads(name: str):
    """Each built-in obstacle name resolves and spawns a valid PyBullet body."""
    env = _make_env()
    bodies_before = env.getNumBodies()

    obstacle = env.add_obstacle(name, position=np.array([2.0, 0.0, 0.5]))

    assert isinstance(obstacle, Obstacle)
    assert obstacle.Id >= 0
    assert env.getNumBodies() == bodies_before + 1
    assert obstacle in env.obstacles

    env.disconnect()


def test_obstacle_pose_matches_spawn():
    """Obstacle pose in PyBullet matches the requested position and orientation."""
    env = _make_env()
    pos = np.array([1.5, -2.0, 0.75])
    orn = np.array([0.0, 0.0, np.pi / 4])

    obstacle = env.add_obstacle("cube", position=pos, orientation=orn)

    pb_pos, pb_quat = env.getBasePositionAndOrientation(obstacle.Id)
    np.testing.assert_allclose(np.asarray(pb_pos), pos, atol=1e-6)

    expected_quat = env.getQuaternionFromEuler(orn.tolist())
    np.testing.assert_allclose(np.asarray(pb_quat), expected_quat, atol=1e-6)

    env.disconnect()


def test_obstacle_is_static_under_gravity():
    """A spawned obstacle should not move once gravity is applied (fixed base)."""
    env = _make_env()
    pos = np.array([3.0, 0.0, 1.0])
    obstacle = env.add_obstacle("cube", position=pos)

    for _ in range(50):
        env.step()

    pb_pos, _ = env.getBasePositionAndOrientation(obstacle.Id)
    np.testing.assert_allclose(np.asarray(pb_pos), pos, atol=1e-6)

    env.disconnect()


def test_custom_urdf_path():
    """Obstacles can be loaded from an arbitrary URDF path."""
    # reuse an existing PyFlyt-shipped URDF that is known to load cleanly
    import PyFlyt

    pyflyt_dir = os.path.dirname(os.path.realpath(PyFlyt.__file__))
    urdf_path = os.path.join(pyflyt_dir, "models", "obstacles", "cube.urdf")
    assert os.path.isfile(urdf_path)

    env = _make_env()
    obstacle = env.add_obstacle(urdf_path, position=np.array([0.0, 2.0, 0.5]))
    assert obstacle.urdf_path == urdf_path
    env.disconnect()


def test_unknown_obstacle_raises():
    """A bogus URDF name (not built-in, not a file) raises FileNotFoundError."""
    env = _make_env()
    with pytest.raises(FileNotFoundError):
        env.add_obstacle(
            "not_a_real_obstacle_name", position=np.array([0.0, 0.0, 1.0])
        )
    env.disconnect()


def test_invalid_position_shape_raises():
    """Position with the wrong shape raises ValueError."""
    env = _make_env()
    with pytest.raises(ValueError):
        env.add_obstacle("cube", position=np.array([0.0, 0.0]))
    env.disconnect()


def test_obstacle_persists_across_reset():
    """Obstacles registered before a reset() are respawned afterwards."""
    env = _make_env()
    env.add_obstacle("cube", position=np.array([2.0, 0.0, 0.5]))
    env.add_obstacle("cylinder", position=np.array([-2.0, 0.0, 1.0]))

    env.reset()

    assert len(env.obstacles) == 2
    # confirm the respawned bodies exist in pybullet
    for obstacle in env.obstacles:
        pb_pos, _ = env.getBasePositionAndOrientation(obstacle.Id)
        assert pb_pos is not None

    env.disconnect()


def test_collision_with_obstacle_is_tracked():
    """Driving a quadrotor straight down into a cube should register a contact."""
    start_pos = np.array([[0.0, 0.0, 1.0]])
    start_orn = np.array([[0.0, 0.0, 0.0]])
    env = Aviary(
        start_pos=start_pos,
        start_orn=start_orn,
        render=False,
        drone_type="quadx",
    )

    # drop a large flat slab directly under the drone
    obstacle = env.add_obstacle(
        "cube", position=np.array([0.0, 0.0, 0.0]), scale=2.0
    )

    # disarm so the drone falls under gravity onto the slab
    env.set_armed(False)
    for _ in range(240):
        env.step()

    # the contact_array entry between the drone body and the obstacle must be True
    drone_id = env.drones[0].Id
    assert env.contact_array[drone_id, obstacle.Id] or env.contact_array[
        obstacle.Id, drone_id
    ]

    env.disconnect()


def test_builtin_obstacles_list():
    """`Obstacle.builtin_obstacles()` exposes the available built-in names."""
    names = Obstacle.builtin_obstacles()
    assert set(names) == {"cube", "cylinder", "sphere"}
