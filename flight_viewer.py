import numpy as np
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PHYSICS_LABELS = [
    "AngVel_Roll", "AngVel_Pitch", "AngVel_Yaw",
    "AngPos_Roll", "AngPos_Pitch", "AngPos_Yaw",
    "LinVel_X",    "LinVel_Y",     "LinVel_Z",
    "Pos_X",       "Pos_Y",        "Pos_Z"
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
        rew = data[f"ep_{i}_rew"]
        
        # Actions Priority: Human -> AI -> Actions
        if f"ep_{i}_human_act" in data and np.abs(data[f"ep_{i}_human_act"]).sum() > 1e-6:
            acts = data[f"ep_{i}_human_act"]
        elif f"ep_{i}_ai_act" in data:
            acts = data[f"ep_{i}_ai_act"]
        else:
            acts = data[f"ep_{i}_act"] if f"ep_{i}_act" in data else np.zeros((len(obs), 4))

        # USE SAVED DATA: Dones and Infos
        dones = data[f"ep_{i}_done"] if f"ep_{i}_done" in data else np.zeros(len(obs))
        infos = data[f"ep_{i}_info"] if f"ep_{i}_info" in data else [None] * len(obs)

        min_len = min(len(obs), len(acts), len(rew))
        obs, acts, rew, dones, infos = obs[:min_len], acts[:min_len], rew[:min_len], dones[:min_len], infos[:min_len]

        # Waypoint & Termination Logic using Data
        total_wps = np.sum(rew > 90.0)
        
        # Determine Status from saved info if available
        status = "In Flight"
        if dones[-1]:
            last_info = infos[-1]
            if isinstance(last_info, dict) and "termination_reason" in last_info:
                status = last_info["termination_reason"]
            elif rew[-1] <= -90.0:
                status = "CRASH"
            elif total_wps >= 4:
                status = "SUCCESS"
            else:
                status = "TERMINATED"

        # CSV Export
        target_labels = [
            "NN_Dist_WP1", "NN_Vec_WP1_X", "NN_Vec_WP1_Y", "NN_Vec_WP1_Z",
            "NN_Dist_WP2", "NN_Vec_WP2_X", "NN_Vec_WP2_Y", "NN_Vec_WP2_Z",
            "NN_Aux_8", "NN_Aux_9", "NN_Aux_10",
            "WP1_Rel_X", "WP1_Rel_Y", "WP1_Rel_Z",
            "WP2_Rel_X", "WP2_Rel_Y", "WP2_Rel_Z"
        ]
        
        cols = PHYSICS_LABELS + (target_labels if obs.shape[1]-12 == 17 else [f"T_{j}" for j in range(obs.shape[1]-12)])
        df = pd.concat([
            pd.DataFrame({"Step": range(min_len), "Reward": rew, "Done": dones, "Status": status}),
            pd.DataFrame(acts, columns=["Ail", "Ele", "Rud", "Thr"]),
            pd.DataFrame(obs, columns=cols)
        ], axis=1)

        csv_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(file_path))[0]}_ep{i}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Ep {i} | Status: {status} | Waypoints: {total_wps}")

        if plot: visualize_flight(obs, i)

def visualize_flight(obs, ep_num):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = obs[:, 9], obs[:, 10], obs[:, 11]
    ax.plot(x, y, z, label='Path', linewidth=2)
    if obs.shape[1] >= 26: # Plot waypoints using WP1_Rel
        ax.scatter((x + obs[:, 23])[::20], (y + obs[:, 24])[::20], (z + obs[:, 25])[::20], c='orange', s=10, alpha=0.3)
    ax.set_box_aspect((1, 1, 1))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    inspect_and_convert(args.file, plot=args.plot)