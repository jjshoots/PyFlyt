from gym.envs.registration import register

register(
    id="PyFlyt/SimpleHover-v0",
    entry_point="PyFlyt.environments:SimpleHoverEnv",
)

register(
    id="PyFlyt/SimpleWaypointEnv-v0",
    entry_point="PyFlyt.environments:SimpleWaypointEnv",
)
