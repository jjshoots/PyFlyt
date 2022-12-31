"""Spawn a single fixed wing UAV on x=0, y=0, z=50, with 0 rpy."""
import numpy as np
import time

from PyFlyt.core import Aviary, loadOBJ, obj_collision, obj_visual

# the starting position and orientations
start_pos = np.array([[0.0, 0.0, 10]])
start_orn = np.array([[0.0, 0.0, 0.0]])

# environment setup
env = Aviary(start_pos=start_pos, start_orn=start_orn, use_camera=True, use_gimbal=False, render=True)

# set to position control
env.set_mode(1)

# call this to register all new bodies for collision
env.register_all_new_bodies()

# load the visual and collision entities and load the duck
visualId = obj_visual(env, "duck.obj")
collisionId = obj_collision(env, "duck.obj")
loadOBJ(
    env,
    visualId=visualId,
    collisionId=collisionId,
    baseMass=1.0,
    basePosition=[0.0, 0.0, 10.0],
)

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in range(10000):
    env.step()
    
