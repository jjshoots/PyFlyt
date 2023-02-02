"""Spawn a single fixed wing UAV on x=0, y=0, z=50, with 0 rpy."""
import argparse
import os
import random
import time
from distutils.util import strtobool
from threading import Event, Thread

import gymnasium
import gymnasium as gym
import numpy as np
from PIL import Image
from pyPS4Controller.controller import Controller

import PyFlyt.gym_envs
from PyFlyt.core import Aviary, loadOBJ, obj_collision, obj_visual


class MyController(Controller):
    def __init__(self, **kwargs):
        Controller.__init__(self, **kwargs)

    def on_R3_down(self, value):
        global cmds

        value = value / 32767

        cmds[0] = value
        return value

    def on_R3_up(self, value):
        global cmds

        value = value / 32767

        cmds[0] = value
        return value

    def on_R3_left(self, value):
        global cmds

        value = value / 32767

        cmds[1] = value
        return value

    def on_R3_right(self, value):
        global cmds

        value = value / 32767

        cmds[1] = value
        return value

    def on_L3_left(self, value):
        global cmds

        value = value / 32767

        cmds[2] = value
        return value

    def on_L3_right(self, value):
        global cmds

        value = value / 32767

        cmds[2] = value
        return value

    def on_R2_press(self, value):
        global cmds

        value = value / 32767

        cmds[3] = value
        return value


def readDS4():
    controller = MyController(interface="/dev/input/js0", connecting_using_ds4drv=False)
    controller.listen()


t = Thread(target=readDS4, args=())
t.start()

cmds = [0, 0, 0, 0]


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="PyFlyt/FWStraightLinePath-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(
            env
        )  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


args = parse_args()
# env setup
# envs = gym.vector.SyncVectorEnv(
#     [make_env(args.env_id, i, args.capture_video, "Test Run", args.gamma) for i in range(args.num_envs)]
# )

envs = gymnasium.make("PyFlyt/FWTargets-v0", render_mode="human")
print("Running")
terminated = False
truncated = False
rewards = 0
stepcount = 0
envs.reset()
timenow = time.time()
imgs_array = []
while not (terminated or truncated):
    imgs_array.append(envs.render()[..., :3].astype(np.uint8))

    next_obs, reward, terminated, truncated, infos = envs.step(cmds)
    rewards += reward
    stepcount += 1
    # print("Reward: {}".format(reward))
    # print("Observation: {}".format(next_obs))
    # print("Speed: {}".format(np.linalg.norm([next_obs[6], next_obs[7], next_obs[8]])))
    # if infos["TargetReached"] or infos["Collide"] or terminated or truncated:
    #     print(next_obs, rewards, terminated, truncated, infos)
    #     print('Episode length: {}'.format(stepcount))
imgs = [Image.fromarray(img) for img in imgs_array]
imgs[0].save(
    "FWTarget.gif", save_all=True, append_images=imgs[1:], duration=100 / 3, loop=0
)
print("Total time: {}".format(time.time() - timenow))
