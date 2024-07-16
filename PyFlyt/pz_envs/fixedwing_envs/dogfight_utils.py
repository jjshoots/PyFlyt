import numpy as np
from PyFlyt.pz_envs.fixedwing_envs.ma_fixedwing_base_env import MAFixedwingBaseEnv

# from PyFlyt.core.utils.compile_helpers import jitter

# @jitter
def _compute_agent_states(
    uav_states: np.ndarray,
    lethal_angle: float,
    lethal_distance: float,
    damage_per_hit: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """COMPUTE HITS"""
    # get the rotation matrices and forward vectors
    # attitudes returns the position of the aircraft nose, shift it to the center of the body
    rotation, forward_vecs = MAFixedwingBaseEnv.compute_rotation_forward(uav_states[:, 1])
    uav_states[:, -1, :] = uav_states[:, -1, :] - (forward_vecs * 0.35)

    # separation here is a [self, other, 3] array of pointwise distance vectors
    separation = uav_states[None, :, -1, :] - uav_states[:, None, -1, :]

    # compute the vectors of each drone to each drone
    current_distances = np.linalg.norm(separation, axis=-1)

    # compute engagement angles
    # WARNING: this has NaNs on the diagonal, watch for everything downstream
    with np.errstate(divide="ignore", invalid="ignore"):
        current_angles = np.arccos(
            np.sum(-separation * forward_vecs, axis=-1) / current_distances
        )

    # compute engagement offsets
    current_offsets = np.linalg.norm(np.cross(separation, forward_vecs), axis=-1)

    # whether we're lethal or chasing or have opponent in cone
    in_cone = current_angles < lethal_angle
    in_range = current_distances < lethal_distance
    chasing = np.abs(current_angles) < (np.pi / 2.0)

    # compute whether anyone hit anyone
    current_hits = in_cone & in_range & chasing

    """COMPUTE THE STATE VECTOR"""
    # form the opponent state matrix
    # this is a [n, n, 4, 3] matrix since each agent needs to attend to every other agent
    opponent_attitudes = np.zeros((uav_states.shape[0], *uav_states.shape), dtype=np.float64)

    # opponent angular rates are unchanged because already body frame
    opponent_attitudes[..., 0, :] = uav_states[:, 0, :]

    # opponent angular positions must convert to be relative to ours
    opponent_attitudes[..., 1, :] = (uav_states[None, :, 1] - uav_states[:, None, 1])

    # rotate all velocities to be ground frame, this is [n, 3]
    ground_velocities = (rotation @ uav_states[:, -2, :][..., None])[..., 0]

    # then find all opponent velocities relative to our body frame
    # this is [self, other, 3]
    opponent_velocities = (ground_velocities[None, ..., None, :] @ rotation[:, None, ...])[..., 0, :]

    # opponent velocities should be relative to our current velocity
    opponent_attitudes[:, 2] = (
        opponent_velocities - uav_states[:, 2, :][None, ...]
    )

    # opponent position is relative to ours in our body frame
    opponent_attitudes[:, 3] = separation @ rotation

    # flatten the attitude and opponent attitude, expand dim of health
    flat_attitude = uav_states.reshape(uav_states.shape[0], -1)
    flat_opponent_attitude = opponent_attitudes.reshape(uav_states.shape[0], uav_states.shape[0], -1)

    return (
        in_cone,
        in_range,
        chasing,
        current_hits,
        current_distances,
        current_angles,
        current_offsets,
        opponent_attitudes,
    )
