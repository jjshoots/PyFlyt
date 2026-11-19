"""Tests the posed camera recorder."""

from __future__ import annotations

import os

import numpy as np
import pybullet as p
import pytest

from PyFlyt.core import Aviary
from PyFlyt.core.utils.camera_recorder import (
    CameraRecorder,
    camera_pose_to_view_matrix,
    rotation_matrix_to_quat_xyzw,
    view_matrix_to_camera_pose,
)


def _quat_to_R(q) -> np.ndarray:
    return np.array(p.getMatrixFromQuaternion(q)).reshape(3, 3)


def _make_env(**camera_options) -> Aviary:
    options = dict(use_camera=True, camera_resolution=(240, 320))
    options.update(camera_options)
    return Aviary(
        start_pos=np.array([[0.0, 0.0, 1.0]]),
        start_orn=np.zeros((1, 3)),
        render=False,
        drone_type="quadx",
        drone_options=options,
    )


def test_view_matrix_round_trip():
    """Pose -> view matrix -> pose is the identity."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        T = np.eye(4)
        T[:3, :3] = _quat_to_R(
            p.getQuaternionFromEuler(rng.uniform(-np.pi, np.pi, 3).tolist())
        )
        T[:3, 3] = rng.uniform(-5, 5, 3)
        np.testing.assert_allclose(
            view_matrix_to_camera_pose(camera_pose_to_view_matrix(T)), T, atol=1e-9
        )
        q = rotation_matrix_to_quat_xyzw(T[:3, :3])
        np.testing.assert_allclose(_quat_to_R(q.tolist()), T[:3, :3], atol=1e-9)


@pytest.mark.parametrize(
    "camera_options,rpy_deg",
    [
        (dict(), (0.0, 0.0, 0.0)),
        (dict(camera_angle_degrees=20), (25.0, -15.0, 70.0)),
        (
            dict(camera_position_offset=np.array([0.1, 0.0, 0.05])),
            (-30.0, 10.0, -120.0),
        ),
        (dict(use_gimbal=True, camera_angle_degrees=30), (20.0, 15.0, 40.0)),
    ],
)
def test_logged_pose_matches_rendered_pixels(tmp_path, camera_options, rpy_deg):
    """Projecting a known sphere with the logged pose and intrinsics lands on its rendered pixels."""
    env = _make_env(**camera_options)
    radius = 0.1
    rec = CameraRecorder(env, str(tmp_path), save_depth=True)
    H, W = rec.height, rec.width

    env.resetBasePositionAndOrientation(
        env.drones[0].Id,
        [0.3, -0.2, 1.0],
        p.getQuaternionFromEuler(np.deg2rad(rpy_deg).tolist()),
    )
    env.stepSimulation()  # refresh link states used by the camera
    T_wc = rec.camera_pose()

    # place a sphere 1.5 m ahead of the camera, off the optical axis
    target = T_wc[:3, :3] @ np.array([0.2, -0.1, 1.5]) + T_wc[:3, 3]
    sphere = env.createMultiBody(
        0,
        env.createCollisionShape(p.GEOM_SPHERE, radius=radius),
        env.createVisualShape(p.GEOM_SPHERE, radius=radius),
        target.tolist(),
    )

    _, T_logged = rec.capture()
    rec.close()
    np.testing.assert_allclose(T_logged, T_wc)

    # segmentation rendered with the same matrices
    _, _, _, _, seg = env.getCameraImage(
        width=W,
        height=H,
        viewMatrix=camera_pose_to_view_matrix(T_wc),
        projectionMatrix=rec.camera.proj_mat,
        renderer=p.ER_TINY_RENDERER,
    )
    v, u = np.nonzero(np.reshape(seg, (H, W)) == sphere)
    assert len(u) > 10
    observed = np.array([u.mean(), v.mean()])

    # re-read everything from disk, as a downstream user would
    fx, fy, cx, cy = np.loadtxt(tmp_path / "intrinsics.txt")
    row = np.genfromtxt(tmp_path / "poses.csv", delimiter=",", names=True)
    R = _quat_to_R([row["qx"], row["qy"], row["qz"], row["qw"]])
    t = np.array([row["tx"], row["ty"], row["tz"]])
    x_c = R.T @ (target - t)
    predicted = np.array([fx * x_c[0] / x_c[2] + cx, fy * x_c[1] / x_c[2] + cy])
    np.testing.assert_allclose(observed, predicted, atol=0.3)

    # metric depth at the sphere's centre is its near surface
    depth = np.load(tmp_path / "depth" / "000000.npy")
    d = depth[int(round(predicted[1])), int(round(predicted[0]))]
    assert abs(d - (x_c[2] - radius * x_c[2] / np.linalg.norm(x_c))) < 0.01

    # a gimballed camera keeps a level horizon: its x axis stays horizontal
    if camera_options.get("use_gimbal"):
        assert abs(T_wc[2, 0]) < 1e-6

    env.disconnect()


def test_requires_camera(tmp_path):
    """A drone without a camera is rejected."""
    env = Aviary(
        start_pos=np.array([[0.0, 0.0, 1.0]]),
        start_orn=np.zeros((1, 3)),
        render=False,
        drone_type="quadx",
    )
    with pytest.raises(ValueError):
        CameraRecorder(env, str(tmp_path))
    env.disconnect()


def test_recorded_flight(tmp_path):
    """A short flight produces one image per pose, with increasing timestamps."""
    env = _make_env(camera_fps=24)
    env.set_mode(7)
    env.set_setpoint(0, np.array([1.0, 0.5, 0.5, 1.5]))
    with CameraRecorder(env, str(tmp_path)) as rec:
        for i in range(240):
            env.step()
            if i % 10 == 0:
                rec.capture()
    env.disconnect()

    images = sorted(os.listdir(tmp_path / "images"))
    poses = np.genfromtxt(tmp_path / "poses.csv", delimiter=",", names=True)
    assert len(images) == len(poses) == 24
    assert np.all(np.diff(poses["timestamp"]) > 0)
    q = np.stack([poses[k] for k in ("qx", "qy", "qz", "qw")], 1)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-6)
    assert poses["tx"][-1] - poses["tx"][0] > 0.3

    # images are valid PNGs of the right size
    with open(tmp_path / "images" / images[0], "rb") as f:
        header = f.read(24)
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    assert np.frombuffer(header[16:24], dtype=">u4").tolist() == [320, 240]
