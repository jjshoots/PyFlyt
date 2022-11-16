from gymnasium.envs.registration import register

register(
    id="PyFlyt/SimpleHoverEnv-v0",
    entry_point="PyFlyt.environments:SimpleHoverEnv",
)

register(
    id="PyFlyt/SimpleWaypointEnv-v0",
    entry_point="PyFlyt.environments:SimpleWaypointEnv",
)

register(
    id="PyFlyt/AdvancedGatesEnv-v0",
    entry_point="PyFlyt.environments:AdvancedGatesEnv",
)
