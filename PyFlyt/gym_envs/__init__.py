from gymnasium.envs.registration import register

from .pyflyt_env import PyFlytEnv

register(
    id="PyFlyt/SimpleHoverEnv-v0",
    entry_point="PyFlyt.gym_envs.simple_hover:SimpleHoverEnv",
)

register(
    id="PyFlyt/SimpleWaypointEnv-v0",
    entry_point="PyFlyt.gym_envs.simple_waypoint:SimpleWaypointEnv",
)

register(
    id="PyFlyt/AdvancedGatesEnv-v0",
    entry_point="PyFlyt.gym_envs.advanced_gates:AdvancedGatesEnv",
)
