from gym.envs.registration import register

register(
    id='fltscl-uavchaser-v0',
    entry_point='pybullet_swarming.environments:uav_chaser',
)
