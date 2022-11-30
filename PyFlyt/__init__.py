# gym environment registration
from gymnasium.envs.registration import register

from PyFlyt.core.aviary import Aviary as Aviary
from PyFlyt.core.drone import BaseCtrlClass as BaseCtrlClass
from PyFlyt.core.drone import Drone as Drone

from PyFlyt.core.load_objs import loadOBJ as loadOBJ
from PyFlyt.core.load_objs import obj_visual as obj_visual
from PyFlyt.core.load_objs import obj_collision as obj_collision

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
