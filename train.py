import gymnasium as gym
import PyFlyt.gym_envs 
from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import argparse
import torch
import os

# --- ARGUMENTS ---
parser = argparse.ArgumentParser(description="Optimized Distributed Training Script")
parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], default="PPO", help="RL Algorithm to use")
parser.add_argument("--steps", type=int, default=1_000_000, help="Total training timesteps")
parser.add_argument("--save-path", type=str, default="fixedwing_agent", help="Path to save the .zip model")
parser.add_argument("--num-envs", type=int, default=8, help="Number of CPU cores for parallel training")
parser.add_argument("--unordered", action="store_true", help="Use unordered waypoints (Harder)")
parser.add_argument("--waypoint-dist", type=float, default=4.0, help="Distance to collect waypoint (default: 4.0m)")
parser.add_argument("--zone", type=float, default=100.0, help="Zone Radius (0 = Auto)")
parser.add_argument("--load-path", type=str, default=None, help="Path to existing .zip model to fine-tune")

args = parser.parse_args()

def train():
    print(f"--- STARTING TRAINING ({args.algo}) ---")
    print(f"CPUs: {args.num_envs} | Steps: {args.steps}")
    print(f"Mode: {'Unordered' if args.unordered else 'Ordered'}")

    # 1. Environment Setup (Standard Physics)
    # Reverted to default settings as requested (100m zone, 4m target)
    env_kwargs = {
        "unordered": args.unordered,
        "flight_dome_size": args.zone, 
        "goal_reach_distance": args.waypoint_dist,
        "max_duration_seconds": 30.0
    }

    # Distributed Environment (SubprocVecEnv = True Parallelism)
    train_env = make_vec_env(
        "PyFlyt/Fixedwing-Waypoints-v4", 
        n_envs=args.num_envs, 
        wrapper_class=FlattenWaypointEnv,   
        wrapper_kwargs={"context_length": 2},
        env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv 
    )

    # 2. Model Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device.upper()}")

    # Determine which algorithm class to use
    ModelClass = PPO if args.algo == "PPO" else SAC

    if args.load_path:
        print(f"--- FINE-TUNING FROM: {args.load_path} ---")
        if not os.path.exists(args.load_path) and not os.path.exists(args.load_path + ".zip"):
            raise FileNotFoundError(f"Model file not found: {args.load_path}")

        # LOAD EXISTING MODEL
        # We attach it to the NEW environment (train_env) which has the updated Zone/Dist settings
        model = ModelClass.load(args.load_path, env=train_env, device=device)
        
        # Optional: Lower learning rate for fine-tuning (prevent destroying old knowledge)
        model.learning_rate = 1e-4
    else:
        print("--- INITIALIZING NEW MODEL FROM SCRATCH ---")

        if args.algo == "PPO":
            # --- PPO OPTIMIZED CONFIG ---
            # Architecture: [256, 256] (Better for Flight Dynamics than default 64x64)
            policy_kwargs = dict(
                activation_fn=torch.nn.Tanh,
                net_arch=dict(pi=[256, 256], vf=[256, 256])
            )
            
            model = PPO(
                "MlpPolicy", 
                train_env, 
                verbose=1, 
                device=device,
                tensorboard_log="./training_logs/",
                policy_kwargs=policy_kwargs,
                
                # Speed & Stability Hyperparameters
                n_steps=2048,           # Long rollout per core
                batch_size=4096,        # Large GPU batch size
                n_epochs=10,            # Reuse data 10 times
                learning_rate=3e-4,
                ent_coef=0.01,          # Prevent early stagnation
                gamma=0.99
            )
            
        else:
            # --- SAC OPTIMIZED CONFIG ---
            # Architecture: [400, 300] (The "Gold Standard" for Continuous Control)
            policy_kwargs = dict(
                activation_fn=torch.nn.Tanh,
                net_arch=dict(pi=[400, 300], qf=[400, 300])
            )

            model = SAC(
                "MlpPolicy", 
                train_env, 
                verbose=1, 
                device=device,
                tensorboard_log="./training_logs/",
                policy_kwargs=policy_kwargs,
                
                # --- HYPERPARAMETERS TO TUNE ---
                learning_rate=3e-4,      # Try 1e-4 if unstable, 7e-4 if too slow
                buffer_size=1_000_000,   # Keep this high (1M)
                batch_size=2048,         # Keep this large for GPU efficiency
                
                # The "Update Ratio" (Currently 1:1)
                train_freq=(100, "step"), 
                gradient_steps=100,       # Try 200 to learn faster (2x updates)
                
                # Look-ahead & Stability
                gamma=0.995,              # <--- ADD THIS: Better for long flights
                tau=0.01,                 # <--- ADD THIS: Faster target updates
                
                # Entropy (Exploration)
                ent_coef="auto",          # Try 'auto_0.1' if it jitters too much
            )

    # 3. Train
    model.learn(total_timesteps=args.steps)
    
    # 4. Save
    model.save(args.save_path)
    print(f"DONE. Model saved to {args.save_path}.zip")
    train_env.close()

if __name__ == "__main__":
    train()