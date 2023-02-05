from gymnasium.envs.registration import register

register(
    id="PyFlyt/QuadX-Hover-v0",
    entry_point="PyFlyt.gym_envs.quadx_hover_env:QuadXHoverEnv",
)

register(
    id="PyFlyt/QuadX-Waypoints-v0",
    entry_point="PyFlyt.gym_envs.quadx_waypoints_env:QuadXWaypointsEnv",
)

register(
    id="PyFlyt/QuadX-Gates-v0",
    entry_point="PyFlyt.gym_envs.quadx_gates_env:QuadXGatesEnv",
)

register(
    id="PyFlyt/Fixedwing-Waypoints-v0",
    entry_point="PyFlyt.gym_envs.fixedwing_waypoints_env:FixedwingWaypointsEnv",
)
