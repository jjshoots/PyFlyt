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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import csv

# --- RL IMPORTS ---
try:
    from stable_baselines3 import PPO, SAC
except ImportError:
    print("Error: Stable Baselines3 not found.")
    print("Please run: pip install stable-baselines3 shimmy")
    sys.exit(1)

# --- ARGUMENTS ---
parser = argparse.ArgumentParser(description="Universal Drone System: Fly, Record")

# RL & Mode Args
parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], default="PPO", help="RL Algorithm")
parser.add_argument("--pilot", type=str, choices=["human", "agent"], default="human", help="Who flies?")
parser.add_argument("--model-path", type=str, default="fixedwing_agent", help="Agent file path (no ext)")
parser.add_argument("--render-mode", type=str, choices=["human", "none"], default="human", help="Render Mode")

# Assist Args
parser.add_argument("--assist-shadow", action="store_true", help="Show AI 'Shadow Inputs' on HUD")
parser.add_argument("--assist-ghost", action="store_true", help="Show AI 'Ghost Plane' future prediction")
parser.add_argument("--assist-arrow", action="store_true", help="Show HUD 'Navigation Arrow' to target")

# Visual & Sim Args
parser.add_argument("--zone", type=float, default=100.0, help="Zone Radius (0 = Auto)")
parser.add_argument("--disable-hud", action="store_true", help="Disable ALL HUD")
parser.add_argument("--no-horizon", action="store_true", help="Disable Artificial Horizon")
parser.add_argument("--show-data", action="store_true", help="Show text telemetry")
parser.add_argument("--unordered", action="store_true", help="Use unordered waypoints")

# experiment settings
parser.add_argument("--time-per-task", type=float, default=60.0, help="Time per task in seconds, default=300s")
parser.add_argument("--target-throttle", type=float, default=0.5, help="Target throttle for human pilots, default 0.5 (50%)")
parser.add_argument("--waypoint-dist", type=float, default=4.0, help="Distance to collect waypoint (default: 4.0m)")
parser.add_argument("--experiment", action="store_true", help="Run Experiment Mode")
parser.add_argument("--subject-id", type=str, default="test", help="Subject ID (e.g. abc_123)")
parser.add_argument("--session", type=int, choices=[1, 2], default=1, help="Session Number (1 or 2)")
parser.add_argument("--break-time", type=float, default=30.0, help="time in seconds between tasks")
parser.add_argument("--monitor", type=int, default=0, help="External monitor to use")


args = parser.parse_args()

def update_manifest(args, ordered_phases):
    """
    Logs the session details to 'experiment_manifest.csv'.
    Columns: Date, Time, SubjectID, Session, Task1, Path1, Task2, Path2, Task3, Path3
    """
    manifest_file = "experiment_manifest.csv"
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # Base folder structure (must match save_data logic)
    base_data_dir = os.path.join("flight_data", args.subject_id, f"session{args.session}")
    
    # Prepare Row Data
    row = [date_str, time_str, args.subject_id, args.session]
    
    # Loop through the relevant phases (Task 1, Task 2, Task 3)
    # We expect ordered_phases to contain the actual tasks to be flown
    for phase in ordered_phases:
        task_name = phase["tag"] # e.g., "task1", "task_arrow"
        task_path = os.path.join(base_data_dir, task_name)
        row.append(task_name)
        row.append(task_path)
        
    # Check if header is needed
    file_exists = os.path.isfile(manifest_file)
    
    with open(manifest_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Date", "Time", "Subject_ID", "Session", 
                "Task_1", "Folder_1", 
                "Task_2", "Folder_2", 
                "Task_3", "Folder_3"
            ])
        writer.writerow(row)
    
    print(f"Manifest updated for Subject {args.subject_id}")

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
current_episode = {"observations": [], "actions": [], "rewards": [], "terminals": [], "human_actions": [],
    "ai_actions": [], "boundary_hits": []}

# Experiment setup
TARGET_THROTTLE = args.target_throttle
# --- EXPERIMENT CONFIGURATION ---
experiment_phases = []

if args.experiment:
    config_path = "config.json"
    if os.path.exists(config_path):
        print(f"Loading Experiment Config from {config_path}...")
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Overwrite CLI args with JSON values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                print(f"Warning: Config key '{key}' not found in arguments.")
    else:
        print("Error: config.json not found! Using defaults.")

    # 0. Pretest (Warmup - usually not logged in manifest as a "Task")
    # 0. Pretest (Warmup - 1 Minute, Unordered)
    phase_pretest = {
        "tag": "pretest", 
        "name": "Pretest (Free Flight)", 
        "duration": 60.0,       # <--- FIXED: 1 Minute
        "arrow": False, 
        "ghost": False, 
        "show_hud": False,
        "unordered": True       # <--- NEW: Random collection
    }
    
    # 1. Task 1 (Fixed: Solo, Ordered)
    phase_task1 = {
        "tag": "task1", 
        "name": "Task 1 (No Assist)", 
        "duration": args.time_per_task, 
        "arrow": False, 
        "ghost": False, 
        "show_hud": True,
        "unordered": False      # <--- NEW: Ordered path
    }

    # 2. Randomized Conditions (Ordered)
    condition_arrow = {
        "tag": "task_arrow", 
        "name": "Task: Arrow Assist", 
        "duration": args.time_per_task, 
        "arrow": True, 
        "ghost": False, 
        "show_hud": True,
        "unordered": False      # <--- NEW: Ordered path
    }
    condition_ghost = {
        "tag": "task_ghost", 
        "name": "Task: Ghost Assist", 
        "duration": args.time_per_task, 
        "arrow": False, 
        "ghost": True, 
        "show_hud": True,
        "unordered": False      # <--- NEW: Ordered path
    }

    # 3. Determine Randomization
    try:
        subj_num = int(''.join(filter(str.isdigit, args.subject_id)))
    except ValueError:
        subj_num = 0 

    # --- REMOVED INCREMENT LOGIC AS REQUESTED ---
    # if args.session == 2: subj_num += 1 

    variable_tasks = []
    if subj_num % 2 == 0:
        print(f"Subject {args.subject_id}: Even -> Arrow First")
        variable_tasks = [condition_arrow, condition_ghost]
    else:
        print(f"Subject {args.subject_id}: Odd -> Ghost First")
        variable_tasks = [condition_ghost, condition_arrow]

    # 4. Build Final List
    # We run Pretest -> Task 1 -> Variable 1 -> Variable 2
    experiment_phases.append(phase_pretest)
    experiment_phases.append(phase_task1)
    experiment_phases.extend(variable_tasks)
    
    # 5. UPDATE MANIFEST
    # We only want to log the "Real" tasks: Task 1 + the 2 Variable ones
    tasks_to_log = [phase_task1] + variable_tasks
    update_manifest(args, tasks_to_log)

else:
    # Default Mode
    experiment_phases.append({
        "tag": "flight", "name": "Free Flight", "duration": args.time_per_task, 
        "arrow": args.assist_arrow, "ghost": args.assist_ghost, "show_hud": True
    })

# Experiment State
phase_idx = 0
current_phase = experiment_phases[phase_idx]
# --- SYNC INITIAL PHASE SETTINGS ---
args.unordered = current_phase.get("unordered", False)
TARGET_FLIGHT_TIME = current_phase["duration"]
accumulated_time = 0.0
total_crashes_in_task = 0
total_waypoints_in_task = 0

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# --- 2. FLIGHT MODE ---
render_mode = args.render_mode if args.render_mode != "none" else None
try:
    env = gym.make("PyFlyt/Fixedwing-Waypoints-v4", render_mode=render_mode, max_duration_seconds=3600.0, goal_reach_distance=args.waypoint_dist, flight_dome_size=args.zone)
except:
    env = gym.make("PyFlyt/Fixedwing-Waypoints-v0", render_mode=render_mode, max_duration_seconds=3600.0, goal_reach_distance=args.waypoint_dist, flight_dome_size=args.zone)

env = FlattenWaypointEnv(env, context_length=2)

if hasattr(env.unwrapped, "waypoints"):
    env.unwrapped.waypoints.unordered = args.unordered


# Load Agent
agent_model = None
if args.pilot == "agent" or args.assist_shadow or args.assist_ghost or args.experiment:
    path = f"{args.model_path}.zip"
    if os.path.exists(path):
        print(f"Loading Agent from {path}...")
        ModelClass = SAC if args.algo == "SAC" else PPO
        agent_model = ModelClass.load(path)
    else:
        print(f"Warning: Agent {path} not found. Shadow/Ghost assist disabled.")

# Detect Zone
ZONE_RADIUS = args.zone
# if ZONE_RADIUS == 0.0:
#     try: ZONE_RADIUS = env.unwrapped.env.flight_dome_size
#     except: ZONE_RADIUS = 100.0

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
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), display=args.monitor)
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


def get_screen_coords(pos_3d, view_matrix, proj_matrix, width, height):
    """
    Projects a 3D world point to 2D screen coordinates.
    """
    # 1. Convert PyBullet flat tuples to 4x4 Numpy Matrices
    # (Fortran ordering is required for OpenGL matrices)
    view = np.array(view_matrix).reshape(4, 4, order='F')
    proj = np.array(proj_matrix).reshape(4, 4, order='F')
    
    # 2. Project: World -> Camera -> Clip Space
    pos_4d = np.array([pos_3d[0], pos_3d[1], pos_3d[2], 1.0])
    clip_pos = proj @ (view @ pos_4d)
    
    # 3. Check if point is behind the camera (W <= 0)
    if clip_pos[3] <= 0: 
        return None
        
    # 4. Normalize (NDC Space: -1 to 1)
    ndc_x = clip_pos[0] / clip_pos[3]
    ndc_y = clip_pos[1] / clip_pos[3]
    
    # 5. Map to Screen Pixels
    # PyGame Y is down, so we use (1 - ndc_y)
    screen_x = (ndc_x + 1) * 0.5 * width
    screen_y = (1 - ndc_y) * 0.5 * height
    
    return int(screen_x), int(screen_y)

def draw_artificial_horizon(screen, roll, pitch):
    """Draws a pitch ladder and horizon line."""
    CX, CY = WINDOW_W // 2, WINDOW_H // 2
    LENGTH = 400
    pitch_scale = 50.0 / 0.17 # 10 deg = 50px
    y_offset = pitch * pitch_scale
    
    # Reference (Center Marker)
    pygame.draw.line(screen, (255, 255, 0), (CX - 40, CY), (CX - 10, CY), 3)
    pygame.draw.line(screen, (255, 255, 0), (CX + 10, CY), (CX + 40, CY), 3)
    pygame.draw.line(screen, (255, 255, 0), (CX, CY), (CX, CY + 10), 3)

    # Rotating Horizon
    h_surf = pygame.Surface((LENGTH + 100, LENGTH + 100), pygame.SRCALPHA)
    hcx, hcy = (LENGTH + 100)//2, (LENGTH + 100)//2
    pygame.draw.line(h_surf, (0, 255, 0), (0, hcy), (LENGTH+100, hcy), 2)
    
    rotated_surf = pygame.transform.rotate(h_surf, math.degrees(roll))
    rect = rotated_surf.get_rect(center=(CX, CY + y_offset))
    screen.blit(rotated_surf, rect)


def draw_attitude_indicator(screen, roll, pitch):
    """
    Draws a standard aviation Attitude Indicator (AI) Gauge.
    - Blue Sky / Brown Ground
    - Pitch Ladder
    - Bank Indicator
    - Fixed 'Mini Plane' Reference
    """
    # 1. Configuration
    GAUGE_SIZE = 200
    RADIUS = GAUGE_SIZE // 2
    CENTER_X = WINDOW_W -120
    CENTER_Y = WINDOW_H - 120 # Positioned at bottom center
    
    # Colors
    SKY_COLOR = (50, 150, 255)   # Light Blue
    GND_COLOR = (140, 70, 20)    # Brown
    LINE_COLOR = (255, 255, 255) # White Pitch Lines
    
    # Scale: How many pixels does the horizon move for 1 radian of pitch?
    # 90 degrees (1.57 rad) should fill the radius (100px).
    PITCH_SCALE = 100.0 / 1.57 
    
    # 2. Create the "Card" (The internal sliding background)
    # Make it large enough to handle rotation and extreme pitch
    CARD_SIZE = GAUGE_SIZE * 3 
    card_surf = pygame.Surface((CARD_SIZE, CARD_SIZE))
    card_surf.fill(GND_COLOR)
    
    # Draw Sky (Top Half)
    pygame.draw.rect(card_surf, SKY_COLOR, (0, 0, CARD_SIZE, CARD_SIZE // 2))
    
    # Draw Horizon Line
    h_y = CARD_SIZE // 2
    pygame.draw.line(card_surf, LINE_COLOR, (0, h_y), (CARD_SIZE, h_y), 3)
    
    # Draw Pitch Ladder (every 10 degrees = ~0.17 rad)
    # We draw lines above and below the horizon
    for i in range(1, 9): # 10 to 80 degrees
        offset = i * 0.174 * PITCH_SCALE
        
        # Positive Pitch (Sky)
        pygame.draw.line(card_surf, LINE_COLOR, (CARD_SIZE//2 - 20, h_y - offset), (CARD_SIZE//2 + 20, h_y - offset), 2)
        
        # Negative Pitch (Ground)
        pygame.draw.line(card_surf, LINE_COLOR, (CARD_SIZE//2 - 20, h_y + offset), (CARD_SIZE//2 + 20, h_y + offset), 2)

    # 3. Apply Transformations
    # A. Shift for Pitch (Move texture UP for positive pitch)
    # Note: In Pygame, Y increases downwards. Positive pitch = Sky moves down? 
    # Real AI: Pitch Up -> Horizon Bar goes DOWN relative to center.
    pitch_offset = pitch * PITCH_SCALE
    
    # B. Rotate for Roll
    # Pygame rotates CCW. Banking Right (Pos Roll) -> Horizon tilts Left (Pos Rotation)
    rotated_card = pygame.transform.rotate(card_surf, math.degrees(roll))
    
    # 4. Clip/Mask to Circle
    # We create a final gauge surface that is square
    gauge_surf = pygame.Surface((GAUGE_SIZE, GAUGE_SIZE), pygame.SRCALPHA)
    
    # Calculate blit position to keep the horizon centered + pitched
    card_rect = rotated_card.get_rect(center=(RADIUS, RADIUS + pitch_offset))
    gauge_surf.blit(rotated_card, card_rect)
    
    # Create the Circular Mask
    # We draw a circle on a separate surface and use BLEND_RGBA_MIN to keep only the intersection
    mask = pygame.Surface((GAUGE_SIZE, GAUGE_SIZE), pygame.SRCALPHA)
    mask.fill((0,0,0,0)) # Transparent
    pygame.draw.circle(mask, (255, 255, 255, 255), (RADIUS, RADIUS), RADIUS)
    
    # Apply Mask (This keeps the circle content and discards corners)
    gauge_surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    
    # 5. Draw Bezel (Ring)
    pygame.draw.circle(gauge_surf, (50, 50, 50), (RADIUS, RADIUS), RADIUS, 5)
    pygame.draw.circle(gauge_surf, (200, 200, 200), (RADIUS, RADIUS), RADIUS, 2)

    # 6. Draw Fixed Reference (The Plane)
    # Yellow "W" or "Bird" shape fixed in the center
    ref_color = (255, 215, 0) # Gold
    cy = RADIUS
    cx = RADIUS
    # Left Wing
    pygame.draw.line(gauge_surf, ref_color, (cx - 40, cy), (cx - 10, cy), 4)
    pygame.draw.line(gauge_surf, (0,0,0),   (cx - 40, cy+2), (cx - 10, cy+2), 2) # Shadow
    # Right Wing
    pygame.draw.line(gauge_surf, ref_color, (cx + 10, cy), (cx + 40, cy), 4)
    pygame.draw.line(gauge_surf, (0,0,0),   (cx + 10, cy+2), (cx + 40, cy+2), 2) # Shadow
    # Center Dot
    pygame.draw.circle(gauge_surf, ref_color, (cx, cy), 4)

    # 7. Blit to Main Screen
    screen.blit(gauge_surf, (CENTER_X - RADIUS, CENTER_Y - RADIUS))

def draw_altimeter(screen, altitude):
    """Draws a tape-style altimeter on the right."""
    RIGHT_X = WINDOW_W - 100
    CENTER_Y = WINDOW_H // 2
    WIDTH, HEIGHT = 60, 300
    
    # Background
    s = pygame.Surface((WIDTH, HEIGHT)); s.set_alpha(100); s.fill((50, 50, 50))
    screen.blit(s, (RIGHT_X, CENTER_Y - HEIGHT//2))
    pygame.draw.rect(screen, (255, 255, 255), (RIGHT_X, CENTER_Y - HEIGHT//2, WIDTH, HEIGHT), 2)
    
    # Ticks
    scale = 50.0 / 10.0 # 10m = 50px
    start_alt = int(altitude) - 30
    end_alt = int(altitude) + 30
    
    for alt in range(start_alt, end_alt):
        if alt % 5 == 0:
            diff = altitude - alt
            y_pos = CENTER_Y + (diff * scale)
            if (CENTER_Y - HEIGHT//2) < y_pos < (CENTER_Y + HEIGHT//2):
                len_tick = 15 if alt % 10 == 0 else 8
                pygame.draw.line(screen, (255, 255, 255), (RIGHT_X, y_pos), (RIGHT_X + len_tick, y_pos), 2)
                if alt % 10 == 0:
                    screen.blit(tiny_font.render(str(alt), True, (255, 255, 255)), (RIGHT_X + 20, y_pos - 6))

    # Current Alt Box
    pygame.draw.rect(screen, (0, 0, 0), (RIGHT_X - 10, CENTER_Y - 15, WIDTH + 20, 30))
    pygame.draw.rect(screen, (255, 255, 255), (RIGHT_X - 10, CENTER_Y - 15, WIDTH + 20, 30), 2)
    screen.blit(font.render(f"{altitude:.0f}", True, (255, 255, 0)), (RIGHT_X + 5, CENTER_Y - 12))

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
arrow_body_id = None
compass_arrow_id = None


def update_3d_arrow(p, drone_id, targets, active):
    """
    3D Gyro-Compass: A high-fidelity arrow.
    Visuals: 
    - Gold Sphere Hub
    - Red Cylinder Shaft
    - Red Pyramid Tip (Custom Mesh)
    - White Tail Shaft
    """
    global compass_arrow_id
    
    # 1. Cleanup if disabled
    if not active:
        if compass_arrow_id is not None:
            p.removeBody(compass_arrow_id)
            compass_arrow_id = None
        return

    # 2. Create Body (The Gyro Compass)
    if compass_arrow_id is None:
        # A. Central Hub (Gold Sphere)
        hub = p.createVisualShape(p.GEOM_SPHERE, radius=0.06, rgbaColor=[1.0, 0.8, 0.0, 1.0])
        
        # B. North Shaft (Red Cylinder)
        # Rotated 90 deg on Y to point along X-axis
        shaft_orn = p.getQuaternionFromEuler([0, 1.5708, 0])
        north_shaft = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02, length=0.3, 
                                    rgbaColor=[1.0, 0.0, 0.0, 1.0], 
                                    visualFramePosition=[0.15, 0, 0],
                                    visualFrameOrientation=shaft_orn)
        
        # C. Arrow Head (Custom Pyramid Mesh)
        # Since GEOM_CONE doesn't exist, we draw a 4-sided pyramid pointing +X
        tip_len = 0.2
        w = 0.06  # Base width radius
        vertices = [
            [tip_len, 0, 0],    # 0: Tip
            [0, w, w],          # 1: Base Top Right
            [0, -w, w],         # 2: Base Top Left
            [0, -w, -w],        # 3: Base Bot Left
            [0, w, -w],         # 4: Base Bot Right
            [0, 0, 0]           # 5: Base Center (for cap)
        ]
        indices = [
            0,1,2,  0,2,3,  0,3,4,  0,4,1,  # 4 Sides
            1,2,5,  2,3,5,  3,4,5,  4,1,5   # Base Cap
        ]
        
        # Tip starts at end of shaft (0.3)
        arrow_tip = p.createVisualShape(p.GEOM_MESH, 
                                    vertices=vertices, 
                                    indices=indices,
                                    rgbaColor=[1.0, 0.0, 0.0, 1.0],
                                    visualFramePosition=[0.3, 0, 0])

        # D. Tail (White Cylinder)
        south_shaft = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02, length=0.2, 
                                    rgbaColor=[0.9, 0.9, 0.9, 1.0], 
                                    visualFramePosition=[-0.1, 0, 0],
                                    visualFrameOrientation=shaft_orn)

        # Create Compound Body
        compass_arrow_id = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=hub,
            baseCollisionShapeIndex=-1,
            
            linkMasses=[0.0, 0.0, 0.0],
            linkCollisionShapeIndices=[-1, -1, -1],
            linkVisualShapeIndices=[north_shaft, arrow_tip, south_shaft],
            linkPositions=[[0,0,0], [0,0,0], [0,0,0]], 
            linkOrientations=[[0,0,0,1], [0,0,0,1], [0,0,0,1]],
            linkInertialFramePositions=[[0,0,0], [0,0,0], [0,0,0]],
            linkInertialFrameOrientations=[[0,0,0,1], [0,0,0,1], [0,0,0,1]],
            linkParentIndices=[0, 0, 0],
            linkJointTypes=[p.JOINT_FIXED, p.JOINT_FIXED, p.JOINT_FIXED],
            linkJointAxis=[[0,0,0], [0,0,0], [0,0,0]]
        )
        p.setCollisionFilterGroupMask(compass_arrow_id, -1, 0, 0)

    # 3. Update Orientation & Position
    if targets and len(targets) > 0:
        pos, orn = p.getBasePositionAndOrientation(drone_id)
        
        rel_target = targets[0] 
        dist = np.linalg.norm(rel_target)
        if dist < 0.1: return

        # Heading Calculation
        dx, dy, dz = rel_target / dist
        yaw = math.atan2(dy, dx)
        pitch = -math.atan2(dz, math.sqrt(dx*dx + dy*dy))
        
        # Position: 0.8m above drone
        arrow_pos = np.array(pos) + [0, 0, 0.8]
        arrow_orn = p.getQuaternionFromEuler([0, pitch, yaw])
        
        p.resetBasePositionAndOrientation(compass_arrow_id, arrow_pos, arrow_orn)


def update_ghost_plane(p, drone_id, ai_action):
    """
    Overlays a sleek 'Ghost Drone' projected AHEAD of the user.
    - VISUAL: High-Vis White Body with Safety Orange Tail.
    - LOGIC: Projects 3.0m forward. 
    - FIXED: Tail positioned at -0.8m to sit FLUSH against the fuselage back.
             (Fuselage ends at -0.6, Tail extends -0.6 to -1.0)
    """
    global ghost_left_id, ghost_right_id, ghost_tail_id, smoothed_action
    
    # --- CONFIGURATION ---
    MAIN_COLOR = [1.0, 1.0, 1.0, 0.85]   # Bright White
    TIP_COLOR  = [1.0, 0.2, 0.0, 0.9]    # Safety Orange
    GHOST_PITCH_TRIM = -0.1              # Bias Ghost 5-6 deg UP (Negative=Up)
    
    # 1. Create Bodies (One-time setup)
    if ghost_left_id is None:
        # Wing
        wing_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 1.0, 0.03], rgbaColor=MAIN_COLOR)
        # Fuselage (Length 1.2m, ends at +/- 0.6)
        fuse_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.1, 0.1], rgbaColor=MAIN_COLOR)
        # Tail (Length 0.4m)
        tail_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.02, 0.25], rgbaColor=TIP_COLOR)
        
        # Creation Order: Wing -> Fuse -> Tail (Draws Tail last, though Z-buffer dominates)
        ghost_left_id  = p.createMultiBody(baseVisualShapeIndex=wing_shape)
        ghost_right_id = p.createMultiBody(baseVisualShapeIndex=fuse_shape)
        ghost_tail_id  = p.createMultiBody(baseVisualShapeIndex=tail_shape)
        
        for body in [ghost_left_id, ghost_right_id, ghost_tail_id]:
            p.setCollisionFilterGroupMask(body, -1, 0, 0)
    
    try:
        # 2. Get User Drone State
        h_pos, h_orn = p.getBasePositionAndOrientation(drone_id)
        
        # 3. Smooth Input
        alpha = 0.1 
        smoothed_action = (smoothed_action * (1 - alpha)) + (ai_action * alpha)
        
        # 4. Calculate "Rabbit" Position (3.0m Forward)

        rot_mat = np.array(p.getMatrixFromQuaternion(h_orn)).reshape(3, 3)
        forward_offset = 3.5
        forward_offset = rot_mat.dot([forward_offset, 0, 0]) 
        ghost_pos = np.array(h_pos) + forward_offset
        
        # 5. Calculate "Rabbit" Orientation (RELATIVE TO COCKPIT)
        VISUAL_SCALE_ROLL = 0.78  
        VISUAL_SCALE_PITCH = 0.52 
        
        ai_local_euler = [
            smoothed_action[0] * VISUAL_SCALE_ROLL, 
            (smoothed_action[1] * VISUAL_SCALE_PITCH) + GHOST_PITCH_TRIM, 
            0.0 
        ]
        
        ai_local_orn = p.getQuaternionFromEuler(ai_local_euler)
        
        # Apply Rotation
        _, ghost_orn = p.multiplyTransforms([0,0,0], h_orn, [0,0,0], ai_local_orn)
        
        # 6. Apply to Bodies
        p.resetBasePositionAndOrientation(ghost_left_id, ghost_pos, ghost_orn)
        p.resetBasePositionAndOrientation(ghost_right_id, ghost_pos, ghost_orn)
        
        # Update Tail
        ghost_rot_mat = np.array(p.getMatrixFromQuaternion(ghost_orn)).reshape(3,3)
        
        # --- OFFSET FIX: -0.8 ---
        # Fuselage Ends: -0.6
        # Tail Starts:   -0.6 (Center -0.8 + HalfExtent 0.2 = -0.6)
        # This makes it flush without clipping.
        tail_offset = ghost_rot_mat.dot([-0.7, 0, 0.15])
        
        p.resetBasePositionAndOrientation(ghost_tail_id, ghost_pos + tail_offset, ghost_orn)
        
    except Exception: 
        pass

# --- VISUAL FUNCTIONS ---


import pybullet_data  # <--- Make sure to add this import at the top


def ensure_safety_floor(p):
    """
    1. Loads the standard 'plane.urdf' (The OG Checkerboard).
    2. Spawns it at Z = 0.01m to COVER the internal PyFlyt floor (hiding the seam).
    3. Disables collision so PyFlyt doesn't crash.
    """
    # Access PyBullet standard assets
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # --- LOAD THE OG PLANE ---
    # Position: [0, 0, 0.01] -> Lifts it 1cm up to hide the default 100m circle.
    # globalScaling: Optional. 1.0 is default. 10.0 makes giant checks.
    floor_id = p.loadURDF("plane.urdf", [0, 0, 0.02], useFixedBase=True, globalScaling=1.5)

    # --- DISABLE COLLISION ---
    # Prevents "IndexError" in PyFlyt.
    p.setCollisionFilterGroupMask(floor_id, -1, 0, 0)
    
    # No custom texture code needed. It will use the default grey/white tile.Matte (No reflection)

def force_pygame_focus():
    """
    1. Forces PyBullet debug window to Monitor 0 (0,0).
    2. Forces PyGame window to Monitor 1 (Focus).
    """
    try:
        # --- A. MOVE PYBULLET TO MONITOR 0 ---
        # "Bullet" is the substring usually found in the PyBullet window title.
        # -e 0,0,0,-1,-1  => Gravity, X=0, Y=0, W=Keep, H=Keep
        os.system("wmctrl -r 'Bullet' -e 0,0,0,-1,-1")
        
        # --- B. FOCUS PYGAME ON MONITOR 1 ---
        # (PyGame handles its own position via the 'display' arg, 
        # we just ensure it is on top of the stack)
        caption = f"Pilot: {args.pilot.upper()} | Algo: {args.algo}"
        os.system(f"wmctrl -a '{caption}'")
    except Exception:
        pass

def get_drone_state(env):
    """Safely extracts the drone state from Standard OR Vectorized environments."""
    try:
        # 1. Unwrap if it's a Vectorized Environment (Guided Agent)
        if hasattr(env, 'envs'):
            # Grab the actual environment from inside the list
            target_env = env.envs[0]
        else:
            target_env = env
            
        # 2. Access the internal PyFlyt structure
        # We need to dig down to .env.drones[0]
        if hasattr(target_env, 'unwrapped'):
            core = target_env.unwrapped
        else:
            core = target_env
            
        # 3. Find the drone
        if hasattr(core, 'env') and hasattr(core.env, 'drones'):
             drone = core.env.drones[0]
        elif hasattr(core, 'drones'):
             drone = core.drones[0]
        else:
            return None, None, None, None

        # 4. Get PyBullet ID
        # Note: If accessing 'p' directly here is hard, use drone.Id
        pos, orn = p.getBasePositionAndOrientation(drone.Id)
        euler = p.getEulerFromQuaternion(orn)
        return drone.Id, pos, orn, euler
        
    except Exception as e:
        # print(f"Drone State Error: {e}") # Uncomment for debug
        return None, None, None, None

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

def save_data(session_data, incomplete_episode, args, phase_tag):
    """Saves data to a structured experiment folder."""
    if len(incomplete_episode["observations"]) > 0:
        session_data.append(incomplete_episode)
    if len(session_data) == 0: return

    # Construct Path: flight_data / subject_id / sessionX / taskY / log.npz
    if args.experiment:
        base_path = os.path.join("flight_data", args.subject_id, f"session{args.session}", phase_tag)
    else:
        base_path = "flight_data"
        
    os.makedirs(base_path, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(base_path, f"log_{ts}.npz")
    
    save_dict = {}
    for i, ep in enumerate(session_data):
        save_dict[f"ep_{i}_obs"] = ep["observations"]
        save_dict[f"ep_{i}_act"] = ep["actions"]
        save_dict[f"ep_{i}_human_act"] = ep["human_actions"]
        save_dict[f"ep_{i}_ai_act"] = ep["ai_actions"]
        save_dict[f"ep_{i}_rew"] = ep["rewards"]
        save_dict[f"ep_{i}_wall"] = ep.get("boundary_hits", []) # Handle legacy
    
    np.savez(filename, **save_dict)
    print(f"\nSaved Task Data to: {filename}")

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
ensure_safety_floor(p)

force_pygame_focus()
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
FPS = 60.0
last_capture_time = -100.0
BREAK_TIME = args.break_time

try:
    while running:
        dt_ms = clock.tick(FPS) 
        dt_sec = dt_ms / 1000.0   # Convert to seconds (e.g., 0.033s if running at 30fps)
        # 1. TIME CHECK & PHASE TRANSITION
        if accumulated_time > TARGET_FLIGHT_TIME:
            print(f">>> {current_phase['name']} COMPLETE. SAVING... <<<")
            
            # A. Save Current Task Data
            save_data(session_data, current_episode, args, current_phase["tag"])
            
            # B. Clear Data Buffers
            session_data = []
            current_episode = {
                "observations": [], "actions": [], 
                "human_actions": [], "ai_actions": [], 
                "rewards": [], "terminals": [], "boundary_hits": []
            }
            
            # C. Show "Break" Screen
            waiting_for_next = True
            break_start_time = pygame.time.get_ticks()
            while waiting_for_next:
                screen.fill((0,0,0))
                # BREAK_TIME = 30.0

                # Enforce 5 second wait
                wait_time = (pygame.time.get_ticks() - break_start_time) / 1000.0
                can_continue = wait_time > BREAK_TIME
                msg1 = font.render(f"{current_phase['name']} COMPLETED", True, (0, 255, 0))

                if can_continue:
                    msg2 = font.render("Press TRIGGER or SPACE to Start Next Task", True, (255, 255, 255))
                else:
                    msg2 = font.render(f"Please rest... {BREAK_TIME - wait_time:.0f}", True, (150, 150, 150))

                screen.blit(msg1, (WINDOW_W//2 - 200, WINDOW_H//2 - 40))
                screen.blit(msg2, (WINDOW_W//2 - 300, WINDOW_H//2 + 10))
                pygame.display.flip()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: 
                        running = False; waiting_for_next = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE: 
                        waiting_for_next = False
                    # if event.type == pygame.JOYBUTTONDOWN and event.button == 0: 
                    #     waiting_for_next = False
                    if can_continue and (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
                         waiting_for_next = False
            
            # D. Advance Phase
            phase_idx += 1
            if phase_idx >= len(experiment_phases):
                print("ALL TASKS COMPLETE.")
                running = False
            else:
                # E. Setup Next Phase
                current_phase = experiment_phases[phase_idx]
                TARGET_FLIGHT_TIME = current_phase["duration"]
                
                # --- APPLY NEW SETTINGS (CRITICAL CHANGE) ---
                # 1. Update CLI arg so HUD/AI know what mode we are in
                args.unordered = current_phase.get("unordered", False)

                # 2. Force Environment to switch modes
                if hasattr(env.unwrapped, "waypoints"):
                    env.unwrapped.waypoints.unordered = args.unordered
                    # Reset capture index so Ordered mode starts at Target 0
                    env.unwrapped.waypoints.captured_index = -1 
                # ---------------------------------------------

                accumulated_time = 0.0
                last_capture_time = -100.0
                total_crashes_in_task = 0 
                total_waypoints_in_task = 0
                
                # Reset Env & Ghost
                obs, _ = env.reset()
                ensure_safety_floor(p)
                force_pygame_focus()
                ghost_left_id = None # Reset Ghost Bodies
                ghost_right_id = None
                ghost_tail_id = None
                compass_arrow_id = None
                paused = True # Auto-start next task? Or keep True to wait.

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
            fixed_throttle_cmd = (TARGET_THROTTLE * 2.0) - 1.0
            raw_human = np.array([
                expo(r_roll, EXPO_VALUE) * MAX_ROLL_RATE,
                expo(-r_pitch if INVERT_PITCH else r_pitch, EXPO_VALUE) * MAX_PITCH_RATE,
                expo(r_yaw, EXPO_VALUE) * MAX_YAW_RATE,
                fixed_throttle_cmd
            ])
            # CLIP HUMAN ACTION
            human_action = np.clip(raw_human, -1.0, 1.0)
            
        # 2. Get AI Input
        if agent_model:
            raw_obs_data = obs
            
            # B. Smart Adapter: Translate Unordered Env -> Ordered AI
            if args.unordered and hasattr(env.unwrapped, "waypoints"):
                # 1. Physics State
                plane_state = raw_obs_data[:23]
                
                # 2. Get Targets
                targets = env.unwrapped.waypoints.targets
                
                # We need exactly 2 targets for context_length=2
                # Since it's "Unordered", we sort them by distance to find the "Next 2"
                if len(targets) > 0:
                    _, my_pos, _, _ = get_drone_state(env)
                    
                    # Create list of (distance, vector) tuples
                    target_list = []
                    for t in targets:
                        t_vec = t - my_pos
                        dist = np.linalg.norm(t_vec)
                        target_list.append((dist, t_vec))
                    
                    # Sort by distance (Closest first)
                    target_list.sort(key=lambda x: x[0])
                    
                    # Get 1st Target (Immediate Goal)
                    vec1 = target_list[0][1]
                    
                    # Get 2nd Target (Lookahead), or use Zeros if only 1 left
                    vec2 = target_list[1][1] if len(target_list) > 1 else np.zeros(3)
                    
                    # Concatenate: State + Target1 + Target2
                    final_obs = np.concatenate([plane_state, vec1, vec2])
                else:
                    # No targets left: Send zeros
                    final_obs = np.concatenate([plane_state, np.zeros(6)])
            else:
                final_obs = raw_obs_data
            # C. Predict
            raw_ai, _ = agent_model.predict(final_obs.reshape(1, -1), deterministic=True)
            # D. SHAPE FIX & SPEED CLAMP
            ai_action = np.array(raw_ai).reshape(-1)

            # --- CRITICAL FIX: FORCE AI TO RESPECT HUMAN SPEED ---
            # We overwrite the AI's throttle intent with the fixed human target.
            # This ensures the "Ghost" projects a path that is possible at 50% speed.
            fixed_throttle_cmd = (TARGET_THROTTLE * 2.0) - 1.0
            ai_action[3] = fixed_throttle_cmd

            ai_action = np.clip(ai_action, -1.0, 1.0)
        
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
            accumulated_time += dt_sec
            if current_phase["ghost"] and agent_model and drone_id is not None:
                update_ghost_plane(p, drone_id, ai_action)
            
            if current_phase["arrow"] and drone_id is not None:
                update_3d_arrow(p, drone_id, current_targets, current_phase["arrow"])

            current_episode["observations"].append(obs)
            current_episode["actions"].append(final_action.copy()) 
            # --- STEP 4 FIX: Handle Environment Type ---
            obs, reward, terminated, truncated, info = env.step(final_action)
            # ... env.step(final_action) is above ...

            # --- GROUND CRASH LOGIC ---
            # If the plane hits the visual floor, KILL the run.
            if drone_id is not None:
                d_pos, _ = p.getBasePositionAndOrientation(drone_id)
                
                # Check altitude (Z). If we touch the "Visual Floor" (approx 0.0 to 0.2)...
                if d_pos[2] < 0.2:
                    print(f"CRASH: Ground Impact (Alt: {d_pos[2]:.2f})")
                    
                    # 1. Force Terminal State
                    terminated = True
                    
                    # 2. Apply Crash Penalty
                    reward = -100.0
                    
                    # 3. Mark as Crash for Analytics
                    # (Overrides any 'Boundary Hit' info from the walls)
                    info["boundary_hit"] = True 
                    current_episode["boundary_hits"][-1] = 1.0 # Ensure log catches it
            current_episode["rewards"].append(reward)
            current_episode["terminals"].append(terminated or truncated)
            current_episode["human_actions"].append(human_action.copy())
            current_episode["ai_actions"].append(ai_action.copy())
            hit = 1.0 if info.get("boundary_hit", False) else 0.0
            current_episode["boundary_hits"].append(hit)

            if reward >= 90.0: # Waypoint captured
                last_capture_time = accumulated_time

            if hasattr(env.unwrapped, "waypoints"):
                current_targets = [t - pos for t in env.unwrapped.waypoints.targets]
            else: current_targets = []

            # --- 3. HANDLE CRASH / RESET ---
            if terminated or truncated:
                # Calculate rewards for this episode
                rews = np.array(current_episode['rewards'])
                crashes = np.sum(rews <= -90.0)
                captures = np.sum(rews >= 90.0)

                last_capture_time = -100.0
                
                # Update Experiment Totals
                total_crashes_in_task += crashes
                total_waypoints_in_task += captures

                session_data.append(current_episode)
                print(f"Flight Completed. Waypoints: {np.sum(np.array(current_episode['rewards']) >= 90.0)}")
                
                # Reset Buffer
                current_episode = {"observations": [], "actions": [], "rewards": [], "terminals": [], "human_actions": [],
                    "ai_actions": [], "boundary_hits": []}

                # RESET GHOST IDS (FIX FOR DISAPPEARING GHOST)
                ghost_left_id = None
                ghost_right_id = None
                ghost_tail_id = None
                compass_arrow_id = None
                smoothed_action = np.zeros(4) # Reset smoothing filter

                
                obs, _ = env.reset()
                ensure_safety_floor(p)
                force_pygame_focus()
                if hasattr(env.unwrapped, "waypoints"):
                    total_targets = len(env.unwrapped.waypoints.targets)
                paused = True

        # RENDER
        screen.fill((0, 0, 0))
        main_surf = render_camera(drone_id, pos, orn, mode="chase", w=MAIN_RENDER_W, h=MAIN_RENDER_H)
        if main_surf:
            screen.blit(pygame.transform.scale(main_surf, (WINDOW_W, WINDOW_H)), (0, 0))

        if not args.disable_hud:
            if (accumulated_time - last_capture_time) < 1.0: # Show for 1 second
                msg_text = "WAYPOINT CAPTURED!"
                
                # 1. Draw Black Shadow (Offset by 3 pixels)
                shadow = capture_font.render(msg_text, True, (0, 0, 0))
                screen.blit(shadow, (WINDOW_W//2 - shadow.get_width()//2 + 3, 200 + 3))
                
                # 2. Draw Bright Yellow Text
                msg = capture_font.render(msg_text, True, (255, 215, 0)) # Gold/Yellow
                screen.blit(msg, (WINDOW_W//2 - msg.get_width()//2, 200))

            # --- EXPERIMENT HUD ---
            if args.experiment:
                # 1. Allowed Visuals
                if pos is not None:
                    draw_attitude_indicator(screen, euler[0], euler[1])
                    draw_altimeter(screen, pos[2])

                # 2. Text Info (Bottom Left)
                time_left = max(0.0, TARGET_FLIGHT_TIME - accumulated_time)
                lines = [
                    f"TASK:     {current_phase['name']}",
                    f"TIME:     {time_left:.0f} s",
                    f"GOALS:    {total_waypoints_in_task}",
                    f"CRASHES:  {total_crashes_in_task}"
                ]
                
                # Draw Text Box
                BOX_X, BOX_Y = 20, WINDOW_H - 150
                for i, line in enumerate(lines):
                    txt = font.render(line, True, (255, 255, 255))
                    screen.blit(txt, (BOX_X, BOX_Y + (i * 30)))
                
                # 3. Assistants (Only if allowed in this phase)
                if current_phase["show_hud"] and drone_id is not None:
                    
                    # --- A. RECONSTRUCT CAMERA MATRICES ---
                    # We need these to find where the 3D objects are on the 2D screen.
                    # These values match render_camera(mode="chase") exactly.
                    rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
                    
                    # Chase Camera Settings
                    cam_offset = [-3.5, 0, 1.0] 
                    cam_target_offset = [0, 0, 0]
                    fov = 60
                    
                    cam_pos = np.array(pos) + rot_mat.dot(cam_offset)
                    cam_target = np.array(pos) + rot_mat.dot(cam_target_offset)
                    cam_up = rot_mat.dot([0, 0, 1])
                    
                    view_mat = p.computeViewMatrix(cam_pos, cam_target, cam_up)
                    proj_mat = p.computeProjectionMatrixFOV(fov, float(MAIN_RENDER_W)/MAIN_RENDER_H, 0.1, 1000.0)

                    # --- B. DRAW ARROW DISTANCE ---
                    if current_phase["arrow"] and compass_arrow_id is not None:
                        # 1. Get 3D Position of the arrow
                        arrow_pos_3d, _ = p.getBasePositionAndOrientation(compass_arrow_id)
                        
                        # 2. Project to 2D Screen
                        screen_pos = get_screen_coords(arrow_pos_3d, view_mat, proj_mat, WINDOW_W, WINDOW_H)
                        
                        # 3. Draw Text (Same logic as 2D arrow)
                        if screen_pos:
                            dist = np.linalg.norm(current_targets[0]) if len(current_targets) > 0 else 0
                            
                            # Draw Yellow Text
                            lbl = font.render(f"{dist:.0f}m", True, (255, 255, 0)) 
                            
                            # Center text 60 pixels ABOVE the arrow
                            draw_x = screen_pos[0] - lbl.get_width() // 2
                            draw_y = screen_pos[1] - 60
                            screen.blit(lbl, (draw_x, draw_y))

                    # --- C. DRAW GHOST DISTANCE ---
                    if current_phase["ghost"] and agent_model and ghost_left_id is not None:
                        # 1. Get 3D Position of ghost
                        ghost_pos_3d, _ = p.getBasePositionAndOrientation(ghost_left_id)
                        
                        # 2. Project to 2D Screen
                        screen_pos = get_screen_coords(ghost_pos_3d, view_mat, proj_mat, WINDOW_W, WINDOW_H)
                        
                        # 3. Draw Text
                        if screen_pos:
                            dist = np.linalg.norm(current_targets[0]) if len(current_targets) > 0 else 0
                            
                            # Draw White Text
                            lbl = font.render(f"{dist:.0f}m", True, (255, 255, 255))
                            
                            # Center text 60 pixels ABOVE the ghost plane
                            draw_x = screen_pos[0] - lbl.get_width() // 2
                            draw_y = screen_pos[1] - 60
                            screen.blit(lbl, (draw_x, draw_y))

            # --- STANDARD HUD (Non-Experiment) ---
            else:    
                if args.assist_arrow:
                    # draw_hud_arrow(screen, pos, orn, current_targets, args.unordered)
                    pass
                
                if args.assist_shadow and agent_model:
                    draw_shadow_controls(screen, human_action, ai_action)
                    
                # draw_radar(screen, pos, euler[2], current_targets, ZONE_RADIUS)
                if pos is not None:
                    # draw_artificial_horizon(screen, euler[0], euler[1]) 
                    draw_attitude_indicator(screen, euler[0], euler[1])
                    draw_altimeter(screen, pos[2])
                
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


                # 3. DRAW ASSIST TEXT (The missing part!)
                if drone_id is not None:
                    # --- A. RECONSTRUCT CAMERA MATRICES ---
                    # (Matches render_camera logic)
                    rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
                    cam_offset = [-3.5, 0, 1.0] 
                    cam_pos = np.array(pos) + rot_mat.dot(cam_offset)
                    cam_target = np.array(pos) + rot_mat.dot([0, 0, 0])
                    cam_up = rot_mat.dot([0, 0, 1])
                    
                    view_mat = p.computeViewMatrix(cam_pos, cam_target, cam_up)
                    proj_mat = p.computeProjectionMatrixFOV(60, float(MAIN_RENDER_W)/MAIN_RENDER_H, 0.1, 1000.0)

                    # --- B. ARROW TEXT ---
                    if args.assist_arrow and compass_arrow_id is not None:
                        arrow_pos_3d, _ = p.getBasePositionAndOrientation(compass_arrow_id)
                        screen_pos = get_screen_coords(arrow_pos_3d, view_mat, proj_mat, WINDOW_W, WINDOW_H)
                        
                        if screen_pos:
                            dist = np.linalg.norm(current_targets[0]) if len(current_targets) > 0 else 0
                            lbl = font.render(f"{dist:.0f}m", True, (255, 255, 0)) # Yellow
                            screen.blit(lbl, (screen_pos[0] - lbl.get_width()//2, screen_pos[1] - 60))

                    # --- C. GHOST TEXT ---
                    if args.assist_ghost and agent_model and ghost_left_id is not None:
                        ghost_pos_3d, _ = p.getBasePositionAndOrientation(ghost_left_id)
                        screen_pos = get_screen_coords(ghost_pos_3d, view_mat, proj_mat, WINDOW_W, WINDOW_H)
                        
                        if screen_pos:
                            dist = np.linalg.norm(current_targets[0]) if len(current_targets) > 0 else 0
                            lbl = font.render(f"{dist:.0f}m", True, (255, 255, 255)) # White
                            screen.blit(lbl, (screen_pos[0] - lbl.get_width()//2, screen_pos[1] - 60))

                
        if paused:
            screen.blit(font.render("PAUSED", True, (255, 255, 0)), (WINDOW_W//2 - 50, WINDOW_H//2))
        
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE: paused = not paused
            # if event.type == pygame.JOYBUTTONDOWN:
            #     # if event.button == BTN_PAUSE: paused = not paused
            #     if event.button == BTN_RESET: 
            #         # Manual Reset also needs ghost reset
            #         ghost_left_id = None
            #         ghost_right_id = None
            #         ghost_tail_id = None
            #         smoothed_action = np.zeros(4) # Reset smoothing filter
            #         obs, _ = env.reset()
            #         ensure_safety_floor(p)
            #         paused = True

except KeyboardInterrupt:
    print("Interrupted.")
finally:
    env.close()
    pygame.quit()
    save_data(session_data, current_episode, args, current_phase["tag"])