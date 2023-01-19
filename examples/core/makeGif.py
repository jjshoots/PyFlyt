import numpy as np
import gymnasium
import PyFlyt.gym_envs
from PIL import Image
from mainSAC import SAC
from wingman import Wingman, cpuize, gpuize

import torch.optim as optim
import torch


def setup_net(env, wm):

    has_weights, model_file, optim_file = wm.get_weight_files()

    # set up networks and optimizers
    net = SAC(
        act_size=env.action_space.shape[0],
        obs_size=env.observation_space.shape[0],
    ).to("cuda:0")

    # load the model
    if has_weights:
        net.load_state_dict(torch.load(model_file))
    else:
        raise AssertionError("Something went wrong.")

    return net


if __name__ == "__main__":
    wm = Wingman(config_yaml="./E2SAC/src/settings.yaml")

    env = gymnasium.make("PyFlyt/FWTargets-v0", render_mode="gif")
    net = setup_net(env, wm)
    ep = 0
    total_rwds = []

    while True:
        # Set eval mode
        net.eval()

        # Episode counter
        ep += 1
        print("Running episode: {}".format(ep))

        # Init counters
        terminated = False
        truncated = False
        rewards = 0
        steps = 0
        imgs_array = []

        # Reset env
        state, _ = env.reset()

        while not (terminated or truncated):
            # Capture image
            imgs_array.append(env.render()[..., :3].astype(np.uint8))

            output = net.actor(gpuize(state, "cuda:0").unsqueeze(0))
            action = net.actor.infer(*output)
            action = cpuize(action)[0]
            state, reward, terminated, truncated, infos = env.step(action)

            # Accumulate rewards
            rewards += reward
            steps += 1

            if infos["TargetReached"] or infos["Collide"] or terminated or truncated:
                print(state, rewards, terminated, truncated, infos)
                print('Episode length: {}'.format(steps))

        imgs = [Image.fromarray(img) for img in imgs_array]
        imgs[0].save("FWTarget.gif", save_all=True, append_images=imgs[1:], duration=100/3, loop=0)
        # Show gif
        input("Press Enter to continue:")

        # total_rwds.append(rewards)
    # print("Average rewards after 100 runs: {}".format(np.mean(total_rwds)))




