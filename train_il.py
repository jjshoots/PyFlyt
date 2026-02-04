import gymnasium as gym
import PyFlyt.gym_envs 
from PyFlyt.gym_envs import FlattenWaypointEnv
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import argparse
import torch
import torch.nn as nn
import os
import glob
from imitation.data import types, rollout
# Added SQIL to imports
from imitation.algorithms import bc, sqil
from imitation.algorithms.adversarial import airl
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm

# --- ARGUMENTS ---
parser = argparse.ArgumentParser(description="Imitation Learning Training Script")
parser.add_argument("--device", type=str, default="auto", help="Compute device: 'auto', 'cpu', 'cuda', '0', '1'")
# Added SQIL to choices
parser.add_argument("--algo", type=str, choices=["BC", "AIRL", "SQIL"], default="BC", help="Imitation Algorithm")
parser.add_argument("--base-algo", type=str, choices=["PPO", "SAC"], default="PPO", help="Base RL Agent")
parser.add_argument("--steps", type=int, default=500_000, help="Timesteps (AIRL/SQIL) or Epochs (BC)")
parser.add_argument("--save-path", type=str, default="fixedwing_il_agent", help="Path to save model")
parser.add_argument("--load-path", type=str, default=None, help="Path to existing model to fine-tune (Optional)")
parser.add_argument("--num-envs", type=int, default=4, help="CPU cores for parallel training")
parser.add_argument("--unordered", action="store_true", help="Use unordered waypoints")
parser.add_argument("--waypoint-dist", type=float, default=4.0, help="Goal reach distance")
parser.add_argument("--zone", type=float, default=100.0, help="Flight Zone Radius")
parser.add_argument("--data-dir", type=str, default="flight_data", help="Root folder of experiment data")
parser.add_argument("--session", type=int, default=1, help="Session to pull data from (1 or 2)")
parser.add_argument("--tasks", nargs="+", default=["task1", "task_arrow", "task_ghost"], help="List of task folders to use")

args = parser.parse_args()

# ==============================================================================
# 1. SAC WRAPPER (For BC Only)
# ==============================================================================
class SACBCWrapper(nn.Module):
    def __init__(self, sac_policy):
        super().__init__()
        self.policy = sac_policy
        self.actor = sac_policy.actor
        
    @property
    def device(self): 
        return next(self.actor.parameters()).device
        
    @property
    def observation_space(self): return self.policy.observation_space
    @property
    def action_space(self): return self.policy.action_space

    def forward(self, obs, deterministic=False):
        return self.actor(obs, deterministic=deterministic)

    def evaluate_actions(self, obs, actions):
        target_device = self.device 
        if obs.device != target_device:
            obs = obs.to(target_device)
        if actions.device != target_device:
            actions = actions.to(target_device)

        dist_params = self.actor.get_action_dist_params(obs)
        if isinstance(dist_params, tuple) and len(dist_params) > 2:
            dist_params = dist_params[:2]
        
        dist = self.actor.action_dist.proba_distribution(*dist_params)
        
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy() 
        
        return None, log_prob, entropy

# ==============================================================================
# 2. DATA LOADER
# ==============================================================================
def load_expert_trajectories(data_root, session_num, allowed_tasks, algo):
    print(f"\n--- LOADING EXPERT DATA (Session {session_num}) ---")
    trajectories = []
    total_transitions = 0
    files_loaded = 0

    subject_dirs = [d for d in glob.glob(os.path.join(data_root, "*")) if os.path.isdir(d)]
    
    for subj in subject_dirs:
        session_path = os.path.join(subj, f"session{session_num}")
        if not os.path.exists(session_path): continue
            
        for task in allowed_tasks:
            task_path = os.path.join(session_path, task)
            if not os.path.exists(task_path): continue
            
            npz_files = glob.glob(os.path.join(task_path, "*.npz"))
            if not npz_files: continue
            npz_files.sort(key=os.path.getmtime, reverse=True)
            target_file = npz_files[0] 
            
            try:
                print(f"Loading: {target_file}")
                data = np.load(target_file, allow_pickle=True)
                keys = list(data.keys())
                ep_indices = sorted(list(set([k.split('_')[1] for k in keys if "obs" in k and "real" not in k])))
                
                valid_eps = 0
                for i in ep_indices:
                    obs = data[f"ep_{i}_obs"]
                    acts = data[f"ep_{i}_human_act"]
                    infos = data[f"ep_{i}_info"]
                    completed = infos[-1]["env_completed"]
                    crashed = infos[-1]["collision"]
                    incomplete = not(completed or crashed)
                    if (algo == "BC" and completed) or algo in ["AIRL", "SQIL"]:

                        min_len = min(len(obs), len(acts))
                        obs = obs[:min_len]
                        acts = acts[:min_len]
                        if min_len < 10: continue 

                        if len(obs) == len(acts):
                            obs = np.concatenate([obs, obs[-1][None]], axis=0)
                        
                        new_traj = types.Trajectory(
                            obs=np.array(obs),
                            acts=np.array(acts),
                            infos=np.array(infos),
                            terminal= (completed or crashed)
                        )
                        trajectories.append(new_traj)
                        total_transitions += len(acts)
                        valid_eps += 1
                
                if valid_eps > 0: files_loaded += 1

            except Exception as e:
                print(f"Failed to load {target_file}: {e}")

    print(f"Loaded {files_loaded} files. Total Transitions: {total_transitions}")
    if len(trajectories) == 0:
        raise ValueError("No valid data found! Check path/session/tasks.")
    return trajectories

# ==============================================================================
# 3. TRAINING LOOP
# ==============================================================================
def train():
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        d_str = args.device if "cuda" in args.device or "cpu" in args.device else f"cuda:{args.device}"
        device = torch.device(d_str)

    print(f"--- STARTING {args.algo} with {args.base_algo} on {device} ---")
    
    rng = np.random.default_rng(0) 

    # 1. Load Data
    expert_trajectories = load_expert_trajectories(args.data_dir, args.session, args.tasks, args.algo)

    # 2. Setup Env
    env_kwargs = {
        "unordered": args.unordered,
        "flight_dome_size": args.zone, 
        "goal_reach_distance": args.waypoint_dist,
        "max_duration_seconds": 30.0
    }
    venv = make_vec_env(
        "PyFlyt/Fixedwing-Waypoints-v4", 
        n_envs=args.num_envs, 
        seed=0, 
        wrapper_class=FlattenWaypointEnv,   
        wrapper_kwargs={"context_length": 2},
        env_kwargs=env_kwargs, 
        vec_env_cls=SubprocVecEnv
    )

    # 3. Algorithm Selection
    
    # --- SQIL BLOCK ---
    if args.algo == "SQIL":
        if args.base_algo != "SAC":
            raise ValueError("SQIL requires an Off-Policy algorithm (SAC). PPO is On-Policy and incompatible.")

        print("--- INITIALIZING SQIL (Constructing new SAC agent internally) ---")
        
        # SQIL constructs the agent itself, so we pass the CLASS, not an instance
        trainer = sqil.SQIL(
            venv=venv,
            demonstrations=expert_trajectories,
            policy="MlpPolicy",
            rl_algo_class=SAC, 
            rl_kwargs=dict(
                policy_kwargs=dict(net_arch=[400, 300]),
                device=device,
                seed=0
            )
        )
        trainer.train(total_timesteps=args.steps)
        # SQIL stores the trained agent in trainer.rl_algo
        trainer.rl_algo.save(args.save_path)

    # --- BC & AIRL BLOCK ---
    else:
        # Initialize standard agent first
        ModelClass = SAC if args.base_algo == "SAC" else PPO
        
        if args.base_algo == "SAC":
            policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=dict(pi=[400, 300], qf=[400, 300]))
        else:
            policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=dict(pi=[256, 256], vf=[256, 256]))

        if args.load_path:
            print(f"--- LOADING WEIGHTS: {args.load_path} ---")
            learner = ModelClass.load(args.load_path, env=venv, device=device)
        else:
            print(f"--- INITIALIZING NEW {args.base_algo} ---")
            learner = ModelClass("MlpPolicy", venv, policy_kwargs=policy_kwargs, device=device)

        if args.algo == "BC":
            transitions = rollout.flatten_trajectories(expert_trajectories)
            
            policy_to_train = learner.policy
            if args.base_algo == "SAC":
                print("Wrapping SAC Policy for BC compatibility...")
                policy_to_train = SACBCWrapper(learner.policy).to(device)

            trainer = bc.BC(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                demonstrations=transitions,
                policy=policy_to_train, 
                device=device,
                rng=rng 
            )
            trainer.train(n_epochs=args.steps)
            learner.save(args.save_path)

        elif args.algo == "AIRL":
            reward_net = BasicShapedRewardNet(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                normalize_input_layer=RunningNorm,
            ).to(device)
            
            # Safe batch size to prevent crashes on small datasets
            demo_batch_size = min(128, len(rollout.flatten_trajectories(expert_trajectories)))
            
            trainer = airl.AIRL(
                demonstrations=expert_trajectories,
                demo_batch_size=demo_batch_size,
                gen_algo=learner,
                reward_net=reward_net,
                venv=venv,
                allow_variable_horizon=True
            )
            trainer.train(total_timesteps=args.steps)
            trainer.gen_algo.save(args.save_path)

    print(f"DONE. Model saved to {args.save_path}.zip")
    venv.close()

if __name__ == "__main__":
    train()