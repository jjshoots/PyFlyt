import numpy as np
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- FULL 29-COLUMN MAPPING ---
LABELS = [
    "AngVel_X", "AngVel_Y", "AngVel_Z",    # 0-2
    "Quat_W", "Quat_X", "Quat_Y", "Quat_Z", # 3-6
    "LinVel_X", "LinVel_Y", "LinVel_Z",    # 7-9
    "Pos_X", "Pos_Y", "Pos_Z",             # 10-12 (GLOBAL POSITION)
    "Act_Roll", "Act_Pitch", "Act_Yaw", "Act_Thr", # 13-16
    "Aux_Thr", "Aux_MinH", "Aux_Dome", "Aux_Time"  # 17-20
]
TARGET_LABELS = [
    "WP1_Rel_X", "WP1_Rel_Y", "WP1_Rel_Z", "WP1_Rel_Yaw", # 21-24
    "WP2_Rel_X", "WP2_Rel_Y", "WP2_Rel_Z", "WP2_Rel_Yaw"  # 25-28
]

def inspect_and_convert(file_path, output_dir="csv_logs", plot=False):
    try:
        data = np.load(file_path, allow_pickle=True)
        print(f"\n--- Processing: {os.path.basename(file_path)} ---")
    except Exception as e:
        print(f"CRITICAL: Error loading file: {e}")
        return

    keys = list(data.keys())
    ep_indices = sorted(list(set([k.split('_')[1] for k in keys if "obs" in k and "real" not in k])))
    
    if not ep_indices:
        print("No flight episodes found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for i in ep_indices:
        obs = data[f"ep_{i}_obs"]
        rew = data[f"ep_{i}_rew"] if f"ep_{i}_rew" in data else np.zeros(len(obs))
        dones = data[f"ep_{i}_done"] if f"ep_{i}_done" in data else np.zeros(len(obs))
        infos = data[f"ep_{i}_info"] if f"ep_{i}_info" in data else [None] * len(obs)
        
        # --- NEW: Load Global Targets ---
        # This will be None for old files, and a (N, 3) array for new files
        global_targets = None
        if f"ep_{i}_global_targets" in data:
            global_targets = data[f"ep_{i}_global_targets"]

        # Load Actions
        if f"ep_{i}_human_act" in data and np.abs(data[f"ep_{i}_human_act"]).sum() > 1e-6:
            acts = data[f"ep_{i}_human_act"]
        elif f"ep_{i}_ai_act" in data:
            acts = data[f"ep_{i}_ai_act"]
        else:
            acts = np.zeros((len(obs), 4))

        # Sync Lengths
        min_len = min(len(obs), len(acts), len(rew), len(dones))
        obs = obs[:min_len]
        acts = acts[:min_len]
        rew = rew[:min_len]
        dones = dones[:min_len]
        infos = infos[:min_len]

        # Status Logic
        total_wps = np.sum(rew > 90.0)
        status = "In_Flight"
        if dones[-1]:
            last_info = infos[-1]
            if isinstance(last_info, dict):
                status = last_info.get("termination_reason", "Terminated")
                if last_info.get("TimeLimit.truncated"): status = "Timeout"
            else:
                if rew[-1] <= -90.0: status = "CRASH"
                elif total_wps >= 4: status = "SUCCESS"
                else: status = "Timeout"

        # CSV Export (Time Series Data)
        final_cols = LABELS.copy()
        if obs.shape[1] == 29:
            final_cols += TARGET_LABELS
        else:
            final_cols += [f"Feat_{j}" for j in range(obs.shape[1] - len(LABELS))]

        df = pd.concat([
            pd.DataFrame({"Step": range(min_len), "Reward": rew, "Done": dones, "Status": status}),
            pd.DataFrame(acts, columns=["Input_Ail", "Input_Ele", "Input_Rud", "Input_Thr"]),
            pd.DataFrame(obs, columns=final_cols)
        ], axis=1)

        csv_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_ep{i}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Ep {i} | Status: {status} | Waypoints Hit: {total_wps}")

        if plot:
            visualize_flight(obs, i, global_targets)

def visualize_flight(obs, ep_num, global_targets=None):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot Drone Path (Indices 10, 11, 12)
    x, y, z = obs[:, 10], obs[:, 11], obs[:, 12]
    
    ax.plot(x, y, z, label='Drone Path', linewidth=2, color='blue')
    ax.scatter(x[0], y[0], z[0], c='green', s=50, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], c='red', marker='x', s=50, label='End')
    
    # 2. Plot Waypoints
    if global_targets is not None and len(global_targets) > 0:
        # --- NEW METHOD: EXACT GLOBAL COORDINATES ---
        # global_targets is shape (Num_Targets, 3) -> [x, y, z]
        wx = global_targets[:, 0]
        wy = global_targets[:, 1]
        wz = global_targets[:, 2]
        
        # Plot all targets clearly
        ax.scatter(wx, wy, wz, c='orange', marker='o', s=100, edgecolors='black', label='Waypoints (Ground Truth)')
        
        # Add labels (1, 2, 3...)
        for jdx, (tx, ty, tz) in enumerate(zip(wx, wy, wz)):
            ax.text(tx, ty, tz, f"{jdx+1}", color='black', fontsize=12, fontweight='bold')
            
    elif obs.shape[1] >= 24:
        # --- OLD METHOD: RELATIVE RECONSTRUCTION (Fallback) ---
        print("Warning: Global Targets not found. Approximating from relative vectors (Less Accurate).")
        # Note: This is rough because of Body Frame rotation issues we discovered.
        wp_x = x + obs[:, 21]
        wp_y = y + obs[:, 22]
        wp_z = z + obs[:, 23]
        ax.scatter(wp_x[::20], wp_y[::20], wp_z[::20], c='orange', alpha=0.3, s=15, label='Target Trace (Approx)')

    # 3. Independent Axis Scaling
    def get_lims(arr):
        span = arr.max() - arr.min()
        if span < 5.0: span = 5.0
        margin = span * 0.1
        return arr.min() - margin, arr.max() + margin

    ax.set_xlim(*get_lims(x))
    ax.set_ylim(*get_lims(y))
    ax.set_zlim(*get_lims(z))

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title(f"Episode {ep_num} Flight Path")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to .npz file")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    inspect_and_convert(args.file, plot=args.plot)