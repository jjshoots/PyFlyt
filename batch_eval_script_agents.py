import pandas as pd
import numpy as np
import subprocess
import glob
import os
import time
import sys

# --- IMPORTS FROM METRICS.PY ---
try:
    from metrics import *
except ImportError:
    print("[Error] metrics.py not found. Please ensure it is in the same directory.")
    sys.exit(1)

# --- CONFIGURATION ---
AGENTS_TO_TEST = ["fw-ppo-v4", "fw-ppo-v4-/AIRL_ALL_ALLData", "fw-ppo-v4-/AIRL_ALL_SuccessOnly", "fw-ppo-v4-/AIRL_Alone_ALLData", "fw-ppo-v4-/AIRL_Alone_SuccessOnly", "fw-ppo-v4-/AIRL_AloneArrow_ALLData", "fw-ppo-v4-/AIRL_AloneArrow_SuccessOnly", "fw-ppo-v4-/AIRL_AloneGhost_ALLData", "fw-ppo-v4-/AIRL_AloneGhost_SuccessOnly", "fw-ppo-v4-/AIRL_Arrow_ALLData", "fw-ppo-v4-/AIRL_Arrow_SuccessOnly", "fw-ppo-v4-/AIRL_ArrowGhost_ALLData", "fw-ppo-v4-/AIRL_ArrowGhost_SuccessOnly", "fw-ppo-v4-/AIRL_Ghost_ALLData", "fw-ppo-v4-/AIRL_Ghost_SuccessOnly", ] 
RENDER_MODE = "human"
FPS = 60.0


# --- PROCESSING LOOP ---
def process_agent(agent_name):
    print(f"\n>>> Processing Agent: {agent_name}")
    
    # 1. Run test_easy.py (5 Minute Endurance)
    print("    Running 5-minute flight test...")
    start_time_sys = time.time()
    
    cmd = [
        "python", "test_easy.py",
        "--pilot", "agent",
        "--algo", "PPO",
        "--model-path", agent_name,
        "--render-mode", RENDER_MODE,
        "--time-per-task", "300.0",
        "--target-throttle", "0.7",
        "--waypoint-dist", "4.0",
        "--break-time", "0.0",
        "--zone", "150.0",
        "--eval",
    ]
    
    subprocess.run(cmd, capture_output=True) 
    
    # 2. Find the Output File
    list_of_files = glob.glob('flight_data/*.npz') 
    if not list_of_files:
        print("    [Error] No .npz file generated.")
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    
    if os.path.getctime(latest_file) < start_time_sys:
        print("    [Error] Latest .npz file is old. Script didn't save new data.")
        return None
        
    print(f"    Parsing {latest_file}...")
    data = np.load(latest_file, allow_pickle=True)
    keys = list(data.keys())
    
    # 3. Iterate Episodes
    episode_metrics = []
    
    ep_indices = [int(k.split('_')[1]) for k in keys if '_obs' in k and k.startswith('ep_')]
    ep_indices = sorted(list(set(ep_indices)))
    
    print(f"    Found {len(ep_indices)} episodes.")
    
    for i in ep_indices:
        try:
            # Extract Data
            obs = data[f"ep_{i}_obs"]
            act = data[f"ep_{i}_act"] # Agent actions
            rew = data[f"ep_{i}_rew"]
            info = data[f"ep_{i}_info"]
            targets = data[f"ep_{i}_global_targets"]
            
            # Physics Extraction
            pos = obs[:, 10:13] # Index 10-12 is Pos XYZ
            vel = obs[:, 7:10]  # Index 7-9 is Vel XYZ
            ang_vel = obs[:, 0:3]     # P, Q, R
            quat = obs[:, 3:7]        # X, Y, Z, W (Fixed Mapping)
            
            # Calculate Basic Stats
            duration = data[f"ep_{i}_real_duration"] if f"ep_{i}_real_duration" in data else len(rew)/FPS
            completed = calculate_success(info)
            crashed = calculate_crashes(info)
            flight_path = calculate_flight_distance(pos)
            
            # Metric Dictionary
            m = {
                # Performance
                "Waypoints": calculate_waypoints_captured(rew),
                "Crashed": crashed,
                "Success": completed,
                "Failures": 0 if completed else 1,
                "Time_Total": duration,
                "Time_Success": duration if completed else np.nan,
                "Near_Misses": calculate_near_misses(pos, targets, rew), 
                "Near_Crashes": calculate_near_crashes(pos), 
                "Flight_Path_Length": flight_path,
                "Flight_Path_Length_Success": flight_path if completed else np.nan,
                "Inversion": calculate_inverted_time(quat),
                "Action_Vol_Bang": calculate_volatility(act),
                
                # Advanced
                "CTE_Avg": calculate_cte(pos, targets, rew),
                "Control_Entropy": calculate_control_entropy(act), # Imported from metrics.py
                "Max_G_Force": calculate_max_g_force(vel, ang_vel),
                "Energy_Variance": calculate_energy_variance(pos, vel), # Imported from metrics.py
                "PIO_Count_Pitch": calculate_pio(act[:, 1]), # Imported from metrics.py
                "PIO_Count_Roll": calculate_pio(act[:, 0]),  # Imported from metrics.py
            }
            episode_metrics.append(m)
            
        except KeyError as e:
            print(f"    [Warning] Skipping Ep {i}, missing key: {e}")
            continue

    if not episode_metrics: return None

    # 4. Aggregation
    df_ep = pd.DataFrame(episode_metrics)
    
    # Aggregation Rules (Trust metrics removed)
    agg_rules = {
        "Waypoints": "sum",
        "Crashed": "sum",
        "Success": "sum",
        "Failures": "sum",
        "Time_Total": "mean",
        "Time_Success": "mean",
        "Near_Misses": "sum",
        "Near_Crashes": "sum",
        "Flight_Path_Length": "mean",
        "Flight_Path_Length_Success": "mean",
        "Inversion": "sum",
        "Action_Vol_Bang": "mean",
        "CTE_Avg": "mean",
        "Control_Entropy": "mean",
        "Max_G_Force": "mean",
        "Energy_Variance": "mean",
        "PIO_Count_Pitch": "sum",
        "PIO_Count_Roll": "sum"
    }
    
    # Aggregate only existing columns
    valid_rules = {k: v for k, v in agg_rules.items() if k in df_ep.columns}
    summary = df_ep.agg(valid_rules)
    
    # Add Identity
    summary["Agent_Name"] = agent_name
    summary["Episodes_Completed"] = len(df_ep)
    
    return summary

# --- MAIN RUNNER ---
def run_benchmark():
    all_summaries = []
    
    for agent in AGENTS_TO_TEST:
        res = process_agent(agent)
        if res is not None:
            all_summaries.append(res)
            
    if not all_summaries:
        print("No data collected.")
        return
        
    final_df = pd.DataFrame(all_summaries)
    
    # Reorder columns
    cols = ["Agent_Name", "Episodes_Completed"] + [c for c in final_df.columns if c not in ["Agent_Name", "Episodes_Completed"]]
    final_df = final_df[cols]
    
    output_file = "agent_benchmark_results.csv"
    final_df.to_csv(output_file, index=False)
    print(f"\n>>> Benchmark Complete. Saved to {output_file}")

if __name__ == "__main__":
    run_benchmark()