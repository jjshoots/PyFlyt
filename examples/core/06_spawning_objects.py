"""Spawn a duck object above the drone."""
import numpy as np

from PyFlyt.core import Aviary, loadOBJ, obj_collision, obj_visual

# the starting position and orientations
start_pos = np.array([[0.0, 0.0, 1.0]])
start_orn = np.array([[0.0, 0.0, 0.0]])

# environment setup
env = Aviary(start_pos=start_pos, start_orn=start_orn, render=True, drone_type="quadx")

# set to position control
env.set_mode(7)

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

# call this to register all new bodies for collision
env.register_all_new_bodies()

# simulate for 1000 steps (1000/120 ~= 8 seconds)
for i in range(1000):
    env.step()
