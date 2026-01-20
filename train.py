import gymnasium as gym
import PyFlyt.gym_envs 
from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import (
    CheckpointCallback, 
    EvalCallback, 
    StopTrainingOnNoModelImprovement, 
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
import wandb
from wandb.integration.sb3 import WandbCallback
import argparse
import os
import numpy as np
import torch

# --- IMITATION LEARNING IMPORTS ---
try:
    from imitation.data.types import TrajectoryWithRew
    from imitation.algorithms import bc
    from imitation.algorithms.adversarial import airl
    from imitation.algorithms import sqil
    from imitation.rewards.reward_nets import BasicShapedRewardNet
    from imitation.util.networks import RunningNorm
    from imitation.testing.reward_improvement import is_significant_reward_improvement
except ImportError:
    print("Warning: 'imitation' library not found. IL modes will fail.")

# --- ARGUMENTS ---
parser = argparse.ArgumentParser(description="PyFlyt Advanced Training Script")
parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], default="PPO", help="RL Algorithm")
parser.add_argument("--il-algo", type=str, choices=["BC", "AIRL", "SQIL", "NONE"], default="NONE", help="Imitation Algorithm")
parser.add_argument("--steps", type=int, default=1_000_000, help="Total training timesteps")
parser.add_argument("--expert-path", type=str, default="flight_data/latest_log.npz", help="Path to expert data")
parser.add_argument("--save-path", type=str, default="fixedwing_agent", help="Base filename for saving")
parser.add_argument("--load-path", type=str, default="fixedwing_agent", help="Filename to load for testing")
parser.add_argument("--num-envs", type=int, default=8, help="Parallel environments (Recommend 8-16 for CPU)")
parser.add_argument("--unordered", action="store_true", help="Use unordered waypoints")
parser.add_argument("--p-value", type=float, default=0.05, help="Statistical significance threshold")
parser.add_argument("--test", action="store_true", help="Run test mode")

args = parser.parse_args()

def ensure_dirs(paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def load_expert_trajectories(path):
    """Parses .npz from test_easy.py into imitation Trajectories."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expert data not found at {path}")

    print(f"Loading expert data from: {path}")
    data = np.load(path, allow_pickle=True)
    trajectories = []
    
    ep_indices = set()
    for key in data.keys():
        if key.startswith("ep_") and "obs" in key:
            ep_indices.add(int(key.split("_")[1]))
    
    for i in sorted(ep_indices):
        obs = data[f"ep_{i}_obs"]
        acts = data[f"ep_{i}_act"]
        rews = data[f"ep_{i}_rew"]
        
        last_obs = obs[-1].reshape(1, -1)
        full_obs = np.concatenate([obs, last_obs], axis=0)
        
        traj = TrajectoryWithRew(obs=full_obs, acts=acts, infos=None, terminal=True, rews=rews)
        trajectories.append(traj)
        
    print(f"Loaded {len(trajectories)} expert trajectories.")
    return trajectories

def create_callbacks(args, eval_env):
    """Creates Checkpoint, Early Stopping, and WandB callbacks."""
    
    # 1. Paths
    ckpt_path = f"./checkpoints/{args.algo}_{args.il_algo}/"
    best_model_path = f"./best_models/{args.algo}_{args.il_algo}/"
    log_path = f"./logs/{args.algo}_{args.il_algo}/"
    ensure_dirs([ckpt_path, best_model_path, log_path])
    
    callbacks = []
    
    # 2. WandB Logging
    callbacks.append(WandbCallback(verbose=2))
    
    # 3. Checkpointing (Save every 50k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=ckpt_path,
        name_prefix="rl_model",
        save_replay_buffer=True if args.algo == "SAC" else False,
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)
    
    # 4. Evaluation & Early Stopping
    if eval_env is not None:
        # Stop if no improvement after 10 evaluations (e.g., 10 * 20k = 200k steps stall)
        stop_train_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=10, 
            min_evals=5, 
            verbose=1
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=best_model_path,
            log_path=log_path,
            eval_freq=20000,         # Evaluate every 20k steps
            n_eval_episodes=5,       # Test for 5 episodes
            deterministic=True,
            render=False,
            callback_after_eval=stop_train_callback
        )
        callbacks.append(eval_callback)
    
    # SB3 requires a generic CallbackList
    return CallbackList(callbacks)

def train_model():
    print(f"\n=== STARTING TRAINING ===")
    print(f"Algorithm:  {args.algo} | Mode: {args.il_algo}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device:     {device.upper()}")
    print(f"Num Envs:   {args.num_envs}")

    # Initialize WandB
    wandb.init(
        project="pyflyt-training",
        name=f"{args.algo}_{args.il_algo}_FIXEDWING",
        sync_tensorboard=True,
        monitor_gym=False,
    )

    # 1. Training Environment (Vectorized)
    train_env = make_vec_env(
        "PyFlyt/Fixedwing-Waypoints-v4", 
        n_envs=args.num_envs, 
        wrapper_class=FlattenWaypointEnv,   
        wrapper_kwargs={"context_length": 2},
        env_kwargs={"unordered": args.unordered}
    )
    
    # 2. Evaluation Environment (Single, for Callbacks)
    # Must match training env specs but run singly for accurate testing
    eval_env = gym.make("PyFlyt/Fixedwing-Waypoints-v4", render_mode=None, unordered=args.unordered)
    eval_env = Monitor(eval_env) # Wraps in Monitor for Stats
    eval_env = FlattenWaypointEnv(eval_env, context_length=2)
    
    # 3. Setup Callbacks
    training_callbacks = create_callbacks(args, eval_env)

    # --- TRAINING BRANCHES ---
    model = None
    
    if args.il_algo != "NONE":
        expert_data = load_expert_trajectories(args.expert_path)
        rng = np.random.default_rng(0)
        
        if args.il_algo == "BC":
            print("--- BEHAVIOR CLONING (Callbacks limited) ---")
            bc_trainer = bc.BC(
                observation_space=train_env.observation_space,
                action_space=train_env.action_space,
                demonstrations=expert_data,
                rng=rng,
                device=device
            )
            # BC uses epochs, not steps. Callbacks support is limited in BC trainer currently.
            # We save manually.
            bc_trainer.train(n_epochs=args.steps // 1000 if args.steps > 100 else 50)
            model = bc_trainer.policy
            torch.save(model, args.save_path + "_bc.pth")
            
        elif args.il_algo == "AIRL":
            print("--- AIRL ---")
            reward_net = BasicShapedRewardNet(
                observation_space=train_env.observation_space,
                action_space=train_env.action_space,
                normalize_input_layer=RunningNorm,
            )
            gen_algo = PPO("MlpPolicy", train_env, verbose=1, device=device, tensorboard_log=f"./logs/{args.il_algo}")
            
            airl_trainer = airl.AIRL(
                demonstrations=expert_data,
                demo_batch_size=2048,
                gen_algo=gen_algo,
                reward_net=reward_net,
                venv=train_env,
                rng=rng
            )
            # AIRL supports callbacks passed to the generator training
            airl_trainer.train(total_timesteps=args.steps, callback=training_callbacks)
            model = airl_trainer.gen_algo
            model.save(args.save_path)

        elif args.il_algo == "SQIL":
            print("--- SQIL ---")
            sqil_trainer = sqil.SQIL(
                venv=train_env,
                demonstrations=expert_data,
                policy="MlpPolicy",
                rl_algo_class=SAC if args.algo == "SAC" else PPO,
                rl_kwargs={"verbose": 1, "device": device, "tensorboard_log": f"./logs/{args.il_algo}"},
            )
            sqil_trainer.train(total_timesteps=args.steps, callback=training_callbacks)
            model = sqil_trainer.rl_algo
            model.save(args.save_path)

    else:
        # --- STANDARD RL ---
        print("--- STANDARD RL ---")
        ModelClass = SAC if args.algo == "SAC" else PPO
        model_kwargs = {"verbose": 1, "tensorboard_log": "./ppo_tensorboard/", "device": device}
        if args.algo == "SAC": model_kwargs["buffer_size"] = 1_000_000
        
        model = ModelClass("MlpPolicy", train_env, **model_kwargs)
        
        # Attach our robust callback list here
        model.learn(total_timesteps=args.steps, callback=training_callbacks)
        model.save(args.save_path)
    
    print(f"Training Finished. Final Model saved to {args.save_path}.zip")
    train_env.close()
    eval_env.close()
    wandb.finish()

def test_model():
    print(f"\n=== TESTING MODEL ===")
    print(f"Loading: {args.load_path}")

    # Load Model (Handle BC vs RL)
    if args.il_algo == "BC":
        policy = torch.load(args.load_path + "_bc.pth")
        model = None
    else:
        ModelClass = SAC if args.algo == "SAC" else PPO
        # Try loading 'best_model' if exists, else load specific path
        best_path = f"./best_models/{args.algo}_{args.il_algo}/best_model.zip"
        if os.path.exists(best_path):
            print(f"Found BEST MODEL at {best_path}, utilizing that instead of final checkpoint.")
            model = ModelClass.load(best_path)
        else:
            model = ModelClass.load(args.load_path)

    env = gym.make("PyFlyt/Fixedwing-Waypoints-v4", render_mode="human", unordered=args.unordered)
    env = FlattenWaypointEnv(env, context_length=2)

    for episode in range(5):
        obs, _ = env.reset()
        done = False
        total_rew = 0
        print(f"--- Episode {episode+1} ---")
        while not done:
            if args.il_algo == "BC":
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, device="cpu").unsqueeze(0)
                    action = policy(obs_t).sample().numpy()[0]
            else:
                action, _ = model.predict(obs, deterministic=True)
                
            obs, reward, terminated, truncated, _ = env.step(action)
            total_rew += reward
            done = terminated or truncated
        print(f"Score: {total_rew:.2f}")

    env.close()

if __name__ == "__main__":
    if args.test:
        test_model()
    else:
        train_model()