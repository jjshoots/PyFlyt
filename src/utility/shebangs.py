import os
import argparse

import yaml
import wandb

from ai_lib.helpers import *

def shutdown_handler(*_):
    print("ctrl-c invoked")
    exit(0)


def arg_parser():
    parser = argparse.ArgumentParser(description='IDK yet.')

    parser.add_argument("--wandb", type=bool, nargs='?',
                            const=True, default=False,
                            help="Log to W and B.")

    parser.add_argument("--debug", type=bool, nargs='?',
                            const=True, default=False,
                            help="Activate debug mode.")

    parser.add_argument("--display", type=bool, nargs='?',
                            const=True, default=False,
                            help="Set to display mode.")

    parser.add_argument("--train", type=bool, nargs='?',
                            const=True, default=False,
                            help="Train the network.")

    parser.add_argument("--shutdown", type=bool, nargs='?',
                            const=True, default=False,
                            help="Shutdown after training.")

    parser.add_argument("--name", type=str, nargs='?',
                            const=False, default='',
                            help="Name of the run in wandb.")

    parser.add_argument("--id", type=str, nargs='?',
                            const=False, default='',
                            help="ID of the run in wandb.")

    return parser.parse_args()


def parse_set():
    """
    Fundamentally there's two things we need to deal with:
        - settings.yaml
        - args

    Depending on the output of args, we also may need to do it through wandb
    """

    # parse arguments
    args = arg_parser()

    # parse settings
    with open(os.path.join(os.path.dirname(__file__), '../settings.yaml')) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    # format settings a bit
    settings['num_envs'] = 1 if args.display else settings['num_envs']
    settings['device'] = get_device()
    settings['step_sched_num'] = settings['repeats_per_buffer'] * settings['epochs'] * settings['buffer_size'] / settings['scheduler_steps']
    settings['buffer_size'] = settings['buffer_size_debug'] if args.debug else settings['buffer_size']
    settings['transitions_per_epoch'] = settings['buffer_size_debug'] if args.debug else settings['transitions_per_epoch']
    settings['net_version'] = str(np.random.randint(100000)) if settings['net_version'] == '' else settings['net_version']

    # merge args and settings
    settings = dict({**settings, **vars(args)})

    # set depending on whether wandb is enabled
    set = None
    if args.wandb and args.train:
        wandb.init(
            project=settings['wandb_project'],
            entity='jjshoots',
            config=settings,
            name=args.name + ', v=' + settings['net_version'] if args.name != '' else settings['net_version'],
            id=args.id if args.id != '' else None)

        # set to be consistent with wandb config
        set = wandb.config
    else:
        # otherwise just merge settings with args
        set = argparse.Namespace(**settings)

    return set
