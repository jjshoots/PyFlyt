"""Registers PyFlyt environments into Gymnasium."""

from gymnasium.envs.registration import register

from PyFlyt.gym_envs.utils.flatten_waypoint_env import FlattenWaypointEnv

# QuadX Envs
register(
    id="PyFlyt/QuadX-Hover-v3",
    entry_point="PyFlyt.gym_envs.quadx_envs.quadx_hover_env:QuadXHoverEnv",
)
register(
    id="PyFlyt/QuadX-Waypoints-v3",
    entry_point="PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env:QuadXWaypointsEnv",
)
register(
    id="PyFlyt/QuadX-Gates-v3",
    entry_point="PyFlyt.gym_envs.quadx_envs.quadx_gates_env:QuadXGatesEnv",
)
register(
    id="PyFlyt/QuadX-Pole-Balance-v3",
    entry_point="PyFlyt.gym_envs.quadx_envs.quadx_pole_balance_env:QuadXPoleBalanceEnv",
)
register(
    id="PyFlyt/QuadX-Pole-Waypoints-v3",
    entry_point="PyFlyt.gym_envs.quadx_envs.quadx_pole_waypoints_env:QuadXPoleWaypointsEnv",
)
register(
    id="PyFlyt/QuadX-Ball-In-Cup-v3",
    entry_point="PyFlyt.gym_envs.quadx_envs.quadx_ball_in_cup_env:QuadXBallInCupEnv",
)

# Fixedwing Envs
register(
    id="PyFlyt/Fixedwing-Waypoints-v3",
    entry_point="PyFlyt.gym_envs.fixedwing_envs.fixedwing_waypoints_env:FixedwingWaypointsEnv",
)

# Rocket Envs
register(
    id="PyFlyt/Rocket-Landing-v3",
    entry_point="PyFlyt.gym_envs.rocket_envs.rocket_landing_env:RocketLandingEnv",
)
