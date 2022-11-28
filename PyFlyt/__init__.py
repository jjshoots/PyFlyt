from gymnasium.envs.registration import register

register(
    id="PyFlyt/SimpleHoverEnv-v0",
    entry_point="PyFlyt.gym_envs:SimpleHoverEnv",
)

register(
    id="PyFlyt/SimpleWaypointEnv-v0",
    entry_point="PyFlyt.gym_envs:SimpleWaypointEnv",
)

register(
    id="PyFlyt/AdvancedGatesEnv-v0",
    entry_point="PyFlyt.gym_envs:AdvancedGatesEnv",
)
