import gymnasium as gym
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv
import pygame
import numpy as np
import math
import argparse
import os
import sys
import json
import datetime
from flight_analytics import FlightAnalytics

# --- RL IMPORTS ---
try:
    from stable_baselines3 import PPO, SAC
except ImportError:
    print("Error: Stable Baselines3 not found.")
    print("Please run: pip install stable-baselines3 shimmy")
    sys.exit(1)

# --- ARGUMENTS ---
parser = argparse.ArgumentParser(description="Universal Drone System: Train, Fly, Record")

# RL & Mode Args
parser.add_argument("--train", action="store_true", help="Training Mode (No Graphics)")
parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], default="PPO", help="RL Algorithm")
parser.add_argument("--steps", type=int, default=100000, help="Training timesteps")
parser.add_argument("--pilot", type=str, choices=["human", "agent"], default="human", help="Who flies?")
parser.add_argument("--model_path", type=str, default="fixedwing_agent", help="Agent file path (no ext)")
parser.add_argument("--render-mode", type=str, choices=["human", "none"], default="human", help="Render Mode")

# Assist Args
parser.add_argument("--assist-shadow", action="store_true", help="Show AI 'Shadow Inputs' on HUD")
parser.add_argument("--assist-ghost", action="store_true", help="Show AI 'Ghost Plane' future prediction")
parser.add_argument("--assist-arrow", action="store_true", help="Show HUD 'Navigation Arrow' to target")

# Visual & Sim Args
parser.add_argument("--zone", type=float, default=0.0, help="Zone Radius (0 = Auto)")
parser.add_argument("--disable-hud", action="store_true", help="Disable ALL HUD")
parser.add_argument("--no-horizon", action="store_true", help="Disable Artificial Horizon")
parser.add_argument("--show-data", action="store_true", help="Show text telemetry")
parser.add_argument("--unordered", action="store_true", help="Use unordered waypoints")
args = parser.parse_args()

# --- CONFIG ---
AXIS_ROLL, AXIS_PITCH, AXIS_YAW, AXIS_THROTTLE = 0, 1, 2, 3
BTN_PAUSE, BTN_PIP, BTN_RADAR, BTN_RESET = 1, 2, 3, 7 
INVERT_PITCH, INVERT_THROTTLE = True, True

EXPO_VALUE = 0.0 
MAX_ROLL_RATE = 1.0
MAX_PITCH_RATE = 1.0
MAX_YAW_RATE = 1.0

# --- DATA RECORDING & SESSION SETUP ---
output_dir = "flight_data"
os.makedirs(output_dir, exist_ok=True)

# List to store multiple episodes (Flight 1, Flight 2, etc.)
session_data = []
# Buffer for the current specific flight
current_episode = {"observations": [], "actions": [], "rewards": [], "terminals": []}

# 5-Minute Timer Setup
TIME_LIMIT_SECONDS = 300 
start_ticks = pygame.time.get_ticks()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- 1. TRAINING MODE ---
if args.train:
    print(f"\n=== TRAINING MODE ===")
    try:
        env = gym.make("PyFlyt/Fixedwing-Waypoints-v4", render_mode=None, unordered=args.unordered)
    except:
        env = gym.make("PyFlyt/Fixedwing-Waypoints-v0", render_mode=None, unordered=args.unordered)
    
    env = FlattenWaypointEnv(env, context_length=2)
    ModelClass = SAC if args.algo == "SAC" else PPO
    model = ModelClass("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.steps)
    model.save(args.model_path)
    print(f"\nTraining Complete! Saved to {args.model_path}.zip")
    env.close()
    sys.exit()

# --- 2. FLIGHT MODE ---
render_mode = args.render_mode if args.render_mode != "none" else None
try:
    env = gym.make("PyFlyt/Fixedwing-Waypoints-v4", render_mode=render_mode, unordered=args.unordered)
except:
    env = gym.make("PyFlyt/Fixedwing-Waypoints-v0", render_mode=render_mode, unordered=args.unordered)

env = FlattenWaypointEnv(env, context_length=2)

# Load Agent
agent_model = None
if args.pilot == "agent" or args.assist_shadow or args.assist_ghost:
    path = f"{args.model_path}.zip"
    if os.path.exists(path):
        print(f"Loading Agent from {path}...")
        ModelClass = SAC if args.algo == "SAC" else PPO
        agent_model = ModelClass.load(path)
    else:
        print(f"Warning: Agent {path} not found. Shadow/Ghost assist disabled.")

# Detect Zone
ZONE_RADIUS = args.zone
if ZONE_RADIUS == 0.0:
    try: ZONE_RADIUS = env.unwrapped.env.flight_dome_size
    except: ZONE_RADIUS = 100.0

# Physics Client
try:
    if hasattr(env.unwrapped, 'ctx'): p = env.unwrapped.ctx.pybullet_client
    elif hasattr(env.unwrapped, 'env'): p = env.unwrapped.env.aviary.ctx.pybullet_client
    else: import pybullet as p
except: import pybullet as p

# Pygame Setup
pygame.init()
pygame.joystick.init()
MAIN_RENDER_W, MAIN_RENDER_H = 960, 540
WINDOW_W, WINDOW_H = 1920, 1080 
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption(f"Pilot: {args.pilot.upper()} | Algo: {args.algo}")

# PIP Settings
PIP_SIZE = (240, 180)
PIP_POS = (WINDOW_W - 250, 10)

font = pygame.font.SysFont("monospace", 20, bold=True)
warning_font = pygame.font.SysFont("monospace", 40, bold=True)
capture_font = pygame.font.SysFont("monospace", 60, bold=True)
small_font = pygame.font.SysFont("monospace", 16)
tiny_font = pygame.font.SysFont("monospace", 12, bold=True)

joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

# --- GHOST PLANE LOGIC ---
ghost_left_id = None
ghost_right_id = None
ghost_tail_id = None
smoothed_action = np.zeros(4) # Reset smoothing filter

def draw_hud_arrow(screen, drone_pos, drone_orn, targets, unordered):
    """
    Draws a SINGLE 3D-style arrow pointing to the active target.
    - Tip points EXACTLY at the target vector.
    """
    if not targets or len(targets) == 0: return

    # 1. Select the Active Target
    if unordered:
        # Find closest
        min_dist = float('inf')
        active_target = None
        for t in targets:
            d = np.linalg.norm(t) 
            if d < min_dist:
                min_dist = d
                active_target = t
    else:
        # First in list
        active_target = targets[0]
        min_dist = np.linalg.norm(active_target)

    if active_target is None: return

    # 2. Project to Screen
    rot_mat = np.array(p.getMatrixFromQuaternion(drone_orn)).reshape(3, 3)
    inv_rot = rot_mat.T
    
    # Transform target into Drone Body Frame
    local_vec = inv_rot.dot(np.array(active_target))
    
    # 3. Calculate Screen Position
    cx, cy = WINDOW_W // 2, WINDOW_H // 2
    scale = 800.0
    
    norm = np.linalg.norm(local_vec)
    if norm < 0.1: return
    direction = local_vec / norm
    
    # PyGame Coords: X=Right, Y=Down
    # Body Coords: Y=Left (Standard Aero), Z=Up
    dx = -direction[1] 
    dy = -direction[2]
    
    # 4. Calculate Angle (Standard 2D Rotation)
    # This aligns 0 radians with the X-axis (Right)
    angle = math.atan2(dy, dx)

    # 5. Clamp to HUD Box
    arrow_x = cx + (dx * scale)
    arrow_y = cy + (dy * scale)
    
    hud_radius = 350
    screen_dist = math.sqrt((arrow_x - cx)**2 + (arrow_y - cy)**2)
    
    if screen_dist > hud_radius:
        ratio = hud_radius / screen_dist
        arrow_x = cx + (arrow_x - cx) * ratio
        arrow_y = cy + (arrow_y - cy) * ratio

    # 6. Rotate Polygon (Defined Pointing RIGHT)
    size = 20
    # Shape: Tip at (size, 0), Base at (-size, +/- size*0.6)
    points = [
        (size, 0),            # The Pointy End
        (-size, -size * 0.6), # Back Top
        (-size, size * 0.6)   # Back Bottom
    ]
    
    rot_points = []
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    
    for px, py in points:
        # Standard 2D Rotation Matrix
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a
        rot_points.append((arrow_x + rx, arrow_y + ry))

    # 7. Draw
    color = (255, 0, 255) # Magenta
    pygame.draw.polygon(screen, color, rot_points)
    pygame.draw.polygon(screen, (255, 255, 255), rot_points, 2)
    
    # Text
    lbl = font.render(f"{min_dist:.0f}m", True, (255, 255, 255))
    screen.blit(lbl, (arrow_x - 20, arrow_y + 30))

# --- GHOST PLANE LOGIC ---
ghost_left_id = None
ghost_right_id = None
ghost_tail_id = None
smoothed_action = np.zeros(4) # Memory for smoothing

def update_ghost_plane(p, drone_id, obs, agent):
    """
    Overlays a smooth, multi-colored 'Ghost Plane' to show Control Intent.
    - SMOOTHED: Filters raw AI jitter to mimic real physics inertia.
    - COLORED: Red=Left, Green=Right, Blue=Tail.
    - RELATIVE: Applies rotation to the drone's current body frame.
    """
    global ghost_left_id, ghost_right_id, ghost_tail_id, smoothed_action
    
    # 1. Create Bodies (One-time setup with Navigation Colors)
    if ghost_left_id is None:
        # Left Wing (Red)
        l_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 1.0, 0.05], rgbaColor=[1, 0, 0, 0.6])
        ghost_left_id = p.createMultiBody(baseVisualShapeIndex=l_shape)
        
        # Right Wing (Green)
        r_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 1.0, 0.05], rgbaColor=[0, 1, 0, 0.6])
        ghost_right_id = p.createMultiBody(baseVisualShapeIndex=r_shape)
        
        # Tail (Blue)
        t_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.05, 0.3], rgbaColor=[0.2, 0.2, 1, 0.8])
        ghost_tail_id = p.createMultiBody(baseVisualShapeIndex=t_shape)
    
    try:
        # 2. Get Current State
        h_pos, h_orn = p.getBasePositionAndOrientation(drone_id)
        
        # 3. Get AI Prediction & Apply Smoothing
        raw_action, _ = agent.predict(obs, deterministic=True)
        
        # Smooth Factor: 0.1 = Very Smooth/Slow, 0.9 = Raw/Jittery
        # 0.15 gives a nice "heavy plane" feel
        alpha = 0.15 
        smoothed_action = (smoothed_action * (1 - alpha)) + (raw_action * alpha)
        
        # 4. Calculate Relative Rotation (Body Frame)
        PREDICTION_SCALE = 1.0 
        
        # Map Actions: Roll (Right+), Pitch (Up+), Yaw (Right+)
        # Note: We invert Pitch because PyBullet +Pitch is Nose Down
        d_roll  = smoothed_action[0] * PREDICTION_SCALE
        d_pitch = -smoothed_action[1] * PREDICTION_SCALE 
        d_yaw   = smoothed_action[2] * PREDICTION_SCALE
        
        # Create Delta Quaternion from smoothed input
        delta_orn = p.getQuaternionFromEuler([d_roll, d_pitch, d_yaw])
        
        # 5. Apply Delta to Current Orientation (Relative Rotation)
        _, ghost_orn = p.multiplyTransforms([0,0,0], h_orn, [0,0,0], delta_orn)
        
        # 6. Update Parts (With Offsets for Left/Right wings)
        # Center Position
        center_pos = h_pos 

        # Helper to place parts relative to the new Ghost Orientation
        def place_part(body_id, local_offset):
            # Rotate the offset by the Ghost's new orientation
            p_pos, p_orn = p.multiplyTransforms(center_pos, ghost_orn, local_offset, [0,0,0,1])
            p.resetBasePositionAndOrientation(body_id, p_pos, p_orn)

        # PyFlyt Frame: X=Forward, Y=Right, Z=Up (Standard Check)
        # If wings are inverted, swap the Y signs below.
        place_part(ghost_left_id,  [0, -0.5, 0])  # Left Wing (Offset -Y)
        place_part(ghost_right_id, [0, 0.5, 0])   # Right Wing (Offset +Y)
        place_part(ghost_tail_id,  [-0.5, 0, 0.1]) # Tail (Back)
        
    except Exception: 
        pass
   
# --- VISUAL FUNCTIONS ---
def get_drone_state(env):
    try:
        drone = env.unwrapped.env.drones[0]
        pos, orn = p.getBasePositionAndOrientation(drone.Id)
        euler = p.getEulerFromQuaternion(orn)
        return drone.Id, pos, orn, euler
    except: return None, None, None, None

def render_camera(drone_id, pos, orn, mode="chase", w=320, h=240):
    if drone_id is None: return None
    try:
        rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        cam_offset = [0.5, 0, 0.1] if mode == "fpv" else [-3.5, 0, 1.0] 
        target_offset = [5.0, 0, 0.0] if mode == "fpv" else [0, 0, 0]
        fov = 80 if mode == "fpv" else 60
        cam_pos = np.array(pos) + rot_mat.dot(cam_offset)
        cam_target = np.array(pos) + rot_mat.dot(target_offset)
        view_matrix = p.computeViewMatrix(cam_pos, cam_target, rot_mat.dot([0, 0, 1]))
        proj_matrix = p.computeProjectionMatrixFOV(fov, float(w)/h, 0.1, 1000.0)
        _, _, rgb, _, _ = p.getCameraImage(width=w, height=h, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return pygame.surfarray.make_surface(np.transpose(np.array(rgb, dtype=np.uint8).reshape(h, w, 4)[:, :, :3], (1, 0, 2)))
    except: return None


def draw_shadow_controls(screen, human_act, ai_act):
    BOX_SIZE = 150
    X_OFF = (WINDOW_W // 2) - (BOX_SIZE // 2)
    Y_OFF = WINDOW_H - BOX_SIZE - 20
    s = pygame.Surface((BOX_SIZE, BOX_SIZE)); s.set_alpha(150); s.fill((30, 30, 30))
    screen.blit(s, (X_OFF, Y_OFF))
    pygame.draw.rect(screen, (150, 150, 150), (X_OFF, Y_OFF, BOX_SIZE, BOX_SIZE), 2)
    cx, cy = X_OFF + BOX_SIZE//2, Y_OFF + BOX_SIZE//2
    pygame.draw.line(screen, (100, 100, 100), (cx, Y_OFF), (cx, Y_OFF+BOX_SIZE), 1)
    pygame.draw.line(screen, (100, 100, 100), (X_OFF, cy), (X_OFF+BOX_SIZE, cy), 1)
    ai_x = cx + int(ai_act[0] * (BOX_SIZE/2))
    ai_y = cy + int(ai_act[1] * (BOX_SIZE/2)) 
    pygame.draw.circle(screen, (0, 255, 0), (ai_x, ai_y), 8, 2)
    hu_x = cx + int(human_act[0] * (BOX_SIZE/2))
    hu_y = cy + int(human_act[1] * (BOX_SIZE/2))
    pygame.draw.circle(screen, (255, 0, 0), (hu_x, hu_y), 6)
    l = small_font.render("SHADOW CONTROL", True, (200, 200, 200))
    screen.blit(l, (X_OFF + 10, Y_OFF - 20))

def draw_radar(screen, drone_pos, drone_yaw, targets, zone_radius):
    RADAR_SIZE, CENTER = 200, (WINDOW_W - 120, WINDOW_H - 120)
    PX_PER_METER = (RADAR_SIZE / 2) / (zone_radius * 1.5)
    s = pygame.Surface((RADAR_SIZE, RADAR_SIZE), pygame.SRCALPHA)
    pygame.draw.circle(s, (0, 40, 0, 200), (RADAR_SIZE//2, RADAR_SIZE//2), RADAR_SIZE // 2)
    screen.blit(s, (CENTER[0] - RADAR_SIZE//2, CENTER[1] - RADAR_SIZE//2))
    pygame.draw.circle(screen, (150, 150, 150), CENTER, RADAR_SIZE // 2, 2)
    
    # Drone Arrow
    pygame.draw.polygon(screen, (255, 255, 255), [(CENTER[0], CENTER[1]-10), (CENTER[0]-7, CENTER[1]+8), (CENTER[0]+7, CENTER[1]+8)])
    
    if targets:
        radar_points = []
        for t in targets:
            tx, ty = t[0] * PX_PER_METER, -t[1] * PX_PER_METER
            rx = tx * math.cos(-drone_yaw) - ty * math.sin(-drone_yaw)
            ry = tx * math.sin(-drone_yaw) + ty * math.cos(-drone_yaw)
            dist = math.sqrt(rx**2 + ry**2)
            if dist > RADAR_SIZE/2 - 5:
                ratio = (RADAR_SIZE/2 - 5) / dist
                rx *= ratio; ry *= ratio
            radar_points.append((CENTER[0]+int(rx), CENTER[1]+int(ry)))
            
        if len(radar_points) > 1:
            pygame.draw.lines(screen, (100, 100, 100), False, radar_points, 1)
            
        for i, p in enumerate(radar_points):
            color = (0, 255, 255) if i == 0 else (255, 255, 0)
            pygame.draw.circle(screen, color, p, 5)
            n = tiny_font.render(str(i+1), True, (255, 255, 255))
            screen.blit(n, (p[0]+6, p[1]-6))

def save_data(session_data, incomplete_episode, args):
    if len(incomplete_episode["observations"]) > 0:
        session_data.append(incomplete_episode)

    if len(session_data) == 0: return

    # Save RAW Data
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("flight_data", f"log_{ts}.npz")
    
    save_dict = {}
    for i, ep in enumerate(session_data):
        save_dict[f"ep_{i}_obs"] = ep["observations"]
        save_dict[f"ep_{i}_act"] = ep["actions"]
        save_dict[f"ep_{i}_rew"] = ep["rewards"]
    
    np.savez(filename, **save_dict)
    print(f"\nSession Log Saved to {filename}")

    # SESSION REPORT
    combined_buffer = {
        "observations": [], 
        "actions": [], 
        "rewards": [], 
        "terminals": [] 
    }
    
    total_crashes = 0
    total_waypoints = 0
    total_time = 0.0

    print("\n" + "="*50)
    print(f"       SESSION REPORT (Average Consistency)")
    print("="*50)

    for i, ep in enumerate(session_data):
        rews = np.array(ep["rewards"])
        crashes = np.sum(rews <= -90.0)
        captures = np.sum(rews >= 90.0)
        duration = len(rews) / 30.0
        
        total_crashes += crashes
        total_waypoints += captures
        total_time += duration
        
        combined_buffer["observations"].extend(ep["observations"])
        combined_buffer["actions"].extend(ep["actions"])
        combined_buffer["rewards"].extend(ep["rewards"])
        
        result = "CRASH" if crashes > 0 else "OK"
        print(f"Run {i+1}: {duration:.1f}s | {captures} Targets | {result}")

    print("-" * 50)
    print(f"GLOBAL SESSION TOTALS:")
    print(f"  Total Time:      {total_time:.1f} s")
    print(f"  Total Waypoints: {total_waypoints}")
    print(f"  Total Crashes:   {total_crashes}")
    
    analytics = FlightAnalytics(combined_buffer)
    analytics.calculate_all()
    
# --- MAIN LOOP ---
clock = pygame.time.Clock()
print("Resetting environment...")
obs, _ = env.reset()
print("Env Ready.")

# Count Targets
total_targets = 0
if hasattr(env.unwrapped, "waypoints") and hasattr(env.unwrapped.waypoints, "targets"):
    total_targets = len(env.unwrapped.waypoints.targets)

action = np.array([0.0, 0.0, 0.0, 0.0])
current_targets = []
last_target_count = 0
paused = True 
running = True

try:
    while running:
        clock.tick(30)
        # --- 1. TIME LIMIT CHECK ---
        elapsed_sec = (pygame.time.get_ticks() - start_ticks) / 1000.0
        if elapsed_sec > TIME_LIMIT_SECONDS:
            print(f">>> TIME LIMIT REACHED ({elapsed_sec:.1f}s). ENDING SESSION. <<<")
            running = False
        drone_id, pos, orn, euler = get_drone_state(env)
        pygame.event.pump() 
        
        # INPUTS
        human_action = np.array([0.0, 0.0, 0.0, 0.0])
        ai_action = np.array([0.0, 0.0, 0.0, 0.0])

        # 1. Get Human Input (With Safety Clip)
        if joystick:
            r_roll = joystick.get_axis(AXIS_ROLL)
            r_pitch = joystick.get_axis(AXIS_PITCH)
            r_yaw = joystick.get_axis(AXIS_YAW)
            r_thr = joystick.get_axis(AXIS_THROTTLE)
            
            def expo(v, e): return (v**3 * e) + (v * (1-e))
            
            raw_human = np.array([
                expo(r_roll, EXPO_VALUE) * MAX_ROLL_RATE,
                expo(-r_pitch if INVERT_PITCH else r_pitch, EXPO_VALUE) * MAX_PITCH_RATE,
                expo(r_yaw, EXPO_VALUE) * MAX_YAW_RATE,
                np.clip((-r_thr + 1.0)/2.0 if INVERT_THROTTLE else r_thr, 0, 1)
            ])
            # CLIP HUMAN ACTION
            human_action = np.clip(raw_human, -1.0, 1.0)
            
        # 2. Get AI Input (With Safety Clip)
        if agent_model:
            raw_ai, _ = agent_model.predict(obs, deterministic=True)
            # CLIP AI ACTION
            ai_action = np.clip(raw_ai, -1.0, 1.0)
        
        
        # --- 3. SELECT CONTROL SOURCE ---
        if args.pilot == "agent":
            # 1. Full AI Pilot
            final_action = ai_action
            
        else:
            # 2. Human Pilot
            final_action = human_action.copy()
            # AUTO-THROTTLE
            # if agent_model is not None:
            #     final_action[3] = ai_action[3]

        # --- 2. STEP & RECORD ---
        if not paused:
            if args.assist_ghost and agent_model and drone_id is not None:
                update_ghost_plane(p, drone_id, obs, agent_model)

            current_episode["observations"].append(obs)
            current_episode["actions"].append(final_action.copy()) 
            obs, reward, terminated, truncated, _ = env.step(final_action)
            current_episode["rewards"].append(reward)
            current_episode["terminals"].append(terminated or truncated)

            if hasattr(env.unwrapped, "waypoints"):
                current_targets = [t - pos for t in env.unwrapped.waypoints.targets]
            else: current_targets = []

            # --- 3. HANDLE CRASH / RESET ---
            if terminated or truncated:
                session_data.append(current_episode)
                print(f"Flight Completed. Waypoints: {np.sum(np.array(current_episode['rewards']) >= 90.0)}")
                
                # Reset Buffer
                current_episode = {"observations": [], "actions": [], "rewards": [], "terminals": []}

                # RESET GHOST IDS (FIX FOR DISAPPEARING GHOST)
                ghost_left_id = None
                ghost_right_id = None
                ghost_tail_id = None
                smoothed_action = np.zeros(4) # Reset smoothing filter

                obs, _ = env.reset()
                if hasattr(env.unwrapped, "waypoints"):
                    total_targets = len(env.unwrapped.waypoints.targets)
                paused = True

        # RENDER
        screen.fill((0, 0, 0))
        main_surf = render_camera(drone_id, pos, orn, mode="chase", w=MAIN_RENDER_W, h=MAIN_RENDER_H)
        if main_surf:
            screen.blit(pygame.transform.scale(main_surf, (WINDOW_W, WINDOW_H)), (0, 0))

        if not args.disable_hud:
            if args.assist_arrow:
                draw_hud_arrow(screen, pos, orn, current_targets, args.unordered)
            
            if args.assist_shadow and agent_model:
                draw_shadow_controls(screen, human_action, ai_action)
                
            draw_radar(screen, pos, euler[2], current_targets, ZONE_RADIUS)
            
            # --- TELEMETRY ---
            if args.show_data and pos:
                BOX_X, BOX_Y = 20, 80
                s = pygame.Surface((300, 160)) 
                s.set_alpha(150); s.fill((0, 0, 0))
                screen.blit(s, (BOX_X, BOX_Y))
                pygame.draw.rect(screen, (255, 255, 255), (BOX_X, BOX_Y, 300, 160), 2)
                
                num_remaining = len(current_targets)
                num_captured = total_targets - num_remaining
                dist = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
                
                lines = [
                    f"PILOT: {args.pilot.upper()}",
                    f"ALGO:  {args.algo}",
                    f"ALT:   {pos[2]:.1f} m",
                    f"DIST:  {dist:.1f} / {ZONE_RADIUS:.0f}m",
                    f"THR:   {final_action[3]*100:.0f}%",
                    f"GOALS: {num_captured} / {total_targets}" 
                ]
                for i, line in enumerate(lines):
                    color = (255, 50, 50) if (i == 3 and dist > ZONE_RADIUS * 0.9) else (0, 255, 0)
                    txt = small_font.render(line, True, color)
                    screen.blit(txt, (BOX_X + 20, BOX_Y + 15 + (i * 24)))

        if paused:
            screen.blit(font.render("PAUSED", True, (255, 255, 0)), (WINDOW_W//2 - 50, WINDOW_H//2))
        
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE: paused = not paused
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == BTN_PAUSE: paused = not paused
                if event.button == BTN_RESET: 
                    # Manual Reset also needs ghost reset
                    ghost_left_id = None
                    ghost_right_id = None
                    ghost_tail_id = None
                    smoothed_action = np.zeros(4) # Reset smoothing filter
                    obs, _ = env.reset()
                    paused = True

except KeyboardInterrupt:
    print("Interrupted.")
finally:
    env.close()
    pygame.quit()
    save_data(session_data, current_episode, args)