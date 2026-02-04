import gymnasium as gym
import PyFlyt.gym_envs
import numpy as np
import argparse
import time
import os
import pybullet as p

# --- CONFIGURATION ---
IDX_POS_QUAT = slice(10, 13)
IDX_ORN_QUAT = slice(3, 7)
IDX_POS_EUL = slice(9, 12)
IDX_ORN_EUL = slice(3, 6)

# --- PYBULLET KEY CODES ---
KEY_ESC = 27
KEY_SPACE = 32
KEY_LEFT = 65295  # Left Arrow
KEY_RIGHT = 65296 # Right Arrow
KEY_R = 114       # 'r' key

def replay_log(file_path, fps=30):
    if not os.path.exists(file_path):
        print(f"Error: File not found {file_path}")
        return

    print(f"Loading {os.path.basename(file_path)}...")
    data = np.load(file_path, allow_pickle=True)
    
    keys = list(data.keys())
    ep_indices = sorted(list(set([k.split('_')[1] for k in keys if "obs" in k and "real" not in k])))
    
    if not ep_indices:
        print("No episodes found.")
        return

    # Visual-only environment
    env = gym.make("PyFlyt/Fixedwing-Waypoints-v4", render_mode="human")
    env.reset()
    uav_id = env.unwrapped.env.drones[0].Id
    
    print("\n" + "="*40)
    print("      REPLAY CONTROLS (CLICK WINDOW)")
    print("="*40)
    print("  [Space]       : Pause / Resume")
    print("  [Right Arrow] : Fast Forward (Hold)")
    print("  [Left Arrow]  : Restart Episode")
    print("  [Esc]         : Quit")
    print("="*40 + "\n")
    
    for ep_i in ep_indices:
        obs = data[f"ep_{ep_i}_obs"]
        steps = len(obs)
        print(f"Playing Episode {ep_i} ({steps} frames)...")
        
        # Detect Orientation Mode
        idx_pos, idx_orn = IDX_POS_QUAT, IDX_ORN_QUAT
        is_quat = True
        # Simple heuristic: Quaternions have norm ~1.0
        if np.abs(np.linalg.norm(obs[0, 3:7]) - 1.0) > 0.1:
            idx_pos, idx_orn = IDX_POS_EUL, IDX_ORN_EUL
            is_quat = False

        i = 0
        paused = False
        
        while i < steps:
            start_time = time.time()
            
            # --- INPUT HANDLING ---
            keys_pressed = p.getKeyboardEvents()
            
            # QUIT
            if KEY_ESC in keys_pressed and (keys_pressed[KEY_ESC] & p.KEY_WAS_TRIGGERED):
                env.close()
                return

            # PAUSE
            if KEY_SPACE in keys_pressed and (keys_pressed[KEY_SPACE] & p.KEY_WAS_TRIGGERED):
                paused = not paused
                print(">> PAUSED" if paused else ">> RESUMED")

            # RESTART (Left Arrow or 'r')
            if (KEY_LEFT in keys_pressed and (keys_pressed[KEY_LEFT] & p.KEY_WAS_TRIGGERED)) or \
               (KEY_R in keys_pressed and (keys_pressed[KEY_R] & p.KEY_WAS_TRIGGERED)):
                i = 0
                print("<< RESTARTING EPISODE")
                # --- FIX: Use native PyBullet command instead of missing method ---
                p.removeAllUserDebugItems() 
                time.sleep(0.2)
                continue

            # SPEED (Right Arrow)
            playback_speed = 1.0
            if KEY_RIGHT in keys_pressed and (keys_pressed[KEY_RIGHT] & p.KEY_IS_DOWN):
                playback_speed = 4.0 # 4x Speed

            if paused:
                time.sleep(0.1)
                continue

            # --- GHOST UPDATE ---
            current_obs = obs[i]
            pos = current_obs[idx_pos]
            orn = current_obs[idx_orn]
            
            if not is_quat:
                orn = p.getQuaternionFromEuler(orn)

            p.resetBasePositionAndOrientation(uav_id, pos, orn)
            
            # Camera
            p.resetDebugVisualizerCamera(
                cameraDistance=4.0,
                cameraYaw=-90,
                cameraPitch=-20,
                cameraTargetPosition=pos
            )
            
            # Draw Trail (Red Line)
            if i > 0 and i % 5 == 0:
                prev_pos = obs[i-5][idx_pos]
                p.addUserDebugLine(prev_pos, pos, [1, 0, 0], 2.0, 5.0)

            i += 1
            
            # Time Sync
            elapsed = time.time() - start_time
            sleep_time = (1.0 / (fps * playback_speed)) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("Replay Finished.")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Path to .npz file")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS")
    args = parser.parse_args()
    
    replay_log(args.file, args.fps)