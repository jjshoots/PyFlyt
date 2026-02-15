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
from imitation.algorithms import bc, sqil
from imitation.algorithms.adversarial import airl
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm

# --- ARGUMENTS ---
parser = argparse.ArgumentParser(description="Imitation Learning Training Script (Batch Mode)")
parser.add_argument("--device", type=str, default="auto", help="Compute device")
parser.add_argument("--algo", type=str, choices=["BC", "AIRL", "SQIL"], default="BC", help="Imitation Algorithm")
parser.add_argument("--base-algo", type=str, choices=["PPO", "SAC"], default="PPO", help="Base RL Agent")
parser.add_argument("--steps", type=int, default=500_000, help="Timesteps (AIRL/SQIL) or Epochs (BC)")
parser.add_argument("--save-path", type=str, default="models", help="Folder to save models") 
parser.add_argument("--load-path", type=str, default=None, help="Path to existing model to fine-tune (Optional)") # <--- RESTORED
parser.add_argument("--num-envs", type=int, default=4, help="CPU cores")
parser.add_argument("--unordered", action="store_true", help="Use unordered waypoints")
parser.add_argument("--waypoint-dist", type=float, default=4.0, help="Goal reach distance")
parser.add_argument("--zone", type=float, default=100.0, help="Flight Zone Radius")
parser.add_argument("--data-dir", type=str, default="flight_data", help="Root folder of experiment data")
parser.add_argument("--session", type=int, default=1, help="Session to pull data from")

# NEW ARGUMENTS
parser.add_argument("--auto-experiment", action="store_true", help="Run all 14 experimental conditions automatically")
parser.add_argument("--tasks", nargs="+", default=["task1", "task_arrow", "task_ghost"], help="Manual task list")
parser.add_argument("--success-only", action="store_true", help="Manual quality filter")

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
    def device(self): return next(self.actor.parameters()).device
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
        return None, dist.log_prob(actions), dist.entropy() 

# ==============================================================================
# 2. DATA LOADER
# ==============================================================================
def load_expert_trajectories(data_root, session_num, allowed_tasks, success_only):
    print(f"   -> Loading: Session {session_num} | Tasks: {allowed_tasks} | Success Only: {success_only}")
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
                data = np.load(target_file, allow_pickle=True)
                keys = list(data.keys())
                ep_indices = sorted(list(set([k.split('_')[1] for k in keys if "obs" in k and "real" not in k])))
                
                for i in ep_indices:
                    obs = data[f"ep_{i}_obs"]
                    acts = data[f"ep_{i}_human_act"]
                    infos = data[f"ep_{i}_info"]
                    
                    completed = infos[-1]["env_complete"]
                    crashed = infos[-1]["collision"]
                    
                    if success_only and not completed: continue

                    min_len = min(len(obs), len(acts))
                    if min_len < 10: continue 

                    obs = obs[:min_len]
                    acts = acts[:min_len]
                    
                    if len(obs) == len(acts):
                        obs = np.concatenate([obs, obs[-1][None]], axis=0)
                    
                    new_traj = types.Trajectory(
                        obs=np.array(obs),
                        acts=np.array(acts),
                        infos=np.array(infos),
                        terminal=(completed or crashed)
                    )
                    trajectories.append(new_traj)
                    total_transitions += len(acts)
                
                files_loaded += 1

            except Exception as e:
                print(f"      [!] Failed to load {target_file}: {e}")

    print(f"      -> Loaded {len(trajectories)} trajectories ({total_transitions} steps) from {files_loaded} files.")
    return trajectories

# ==============================================================================
# 3. TRAINING ROUTINE
# ==============================================================================
def run_training(task_list, success_flag, save_name, device):
    print(f"\n=== STARTING EXPERIMENT: {save_name} ===")
    
    # 1. Load Data
    try:
        expert_trajectories = load_expert_trajectories(args.data_dir, args.session, task_list, success_flag)
        if len(expert_trajectories) == 0:
            print("      [SKIP] No data found for this condition.")
            return
    except Exception as e:
        print(f"      [ERROR] Data load failed: {e}")
        return

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

    # 3. Initialize Agent
    rng = np.random.default_rng(0) 
    
    # --- SQIL ---
    if args.algo == "SQIL":
        if args.base_algo != "SAC":
            print("      [ERROR] SQIL requires SAC.")
            venv.close()
            return
            
        # For SQIL, loading is handled differently as it wraps the Algo
        # We initialize new, but if load_path exists, we could manually load weights.
        # However, SQIL usually trains from scratch or wraps a policy. 
        # Standard SQIL implementation in `imitation` creates its own buffer.
        trainer = sqil.SQIL(
            venv=venv,
            demonstrations=expert_trajectories,
            policy="MlpPolicy",
            rl_algo_class=SAC, 
            rl_kwargs=dict(policy_kwargs=dict(net_arch=[400, 300]), device=device, seed=0)
        )
        # Note: Fine-tuning SQIL from a loaded model is complex in this lib, 
        # so we default to training fresh. If you need fine-tuning, use BC/AIRL.
        trainer.train(total_timesteps=args.steps)
        trainer.rl_algo.save(save_name)

    # --- BC / AIRL ---
    else:
        ModelClass = SAC if args.base_algo == "SAC" else PPO
        policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=dict(pi=[400, 300], qf=[400, 300]) if args.base_algo=="SAC" else dict(pi=[256, 256], vf=[256, 256]))
        
        # --- RESTORED LOAD LOGIC ---
        if args.load_path:
            print(f"      [INFO] Loading Pretrained Weights: {args.load_path}")
            learner = ModelClass.load(args.load_path, env=venv, device=device)
        else:
            learner = ModelClass("MlpPolicy", venv, policy_kwargs=policy_kwargs, device=device)
        # ---------------------------

        if args.algo == "BC":
            transitions = rollout.flatten_trajectories(expert_trajectories)
            policy_to_train = learner.policy
            if args.base_algo == "SAC":
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
            learner.save(save_name)

        elif args.algo == "AIRL":
            reward_net = BasicShapedRewardNet(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                normalize_input_layer=RunningNorm,
            ).to(device)
            
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
            trainer.gen_algo.save(save_name)

    print(f"   -> Saved: {save_name}.zip")
    venv.close()

# ==============================================================================
# 4. MAIN LOOP
# ==============================================================================
def main():
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Define Experiment Conditions
    if args.auto_experiment:
        task_mappings = {
            "Alone": ["task1"],
            "Arrow": ["task_arrow"],
            "Ghost": ["task_ghost"],
            "AloneArrow": ["task1", "task_arrow"],
            "AloneGhost": ["task1", "task_ghost"],
            "ArrowGhost": ["task_arrow", "task_ghost"],
            "All": ["task1", "task_arrow", "task_ghost"]
        }
        quality_settings = {
            "AllData": False,
            "SuccessOnly": True
        }
    else:
        # Manual Mode (Single run)
        task_mappings = {"Manual": args.tasks}
        quality_settings = {"Manual": args.success_only}

    # RUN LOOP
    for qual_name, success_flag in quality_settings.items():
        for task_name, task_list in task_mappings.items():
            
            # Construct Filename
            fname = f"{args.algo}_{task_name}_{qual_name}"
            full_path = os.path.join(args.save_path, fname)
            
            run_training(task_list, success_flag, full_path, device)

if __name__ == "__main__":
    main()