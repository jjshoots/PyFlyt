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
parser.add_argument("--vec-norm", action="store_true", help="Enable Vector Normalization")

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
    # Phase 0: Pretest (No assists, No waypoints shown)
    experiment_phases.append({
        "tag": "pretest", 
        "name": "Pretest (Free Flight)", 
        "duration": args.time_per_task, 
        "arrow": False, 
        "ghost": False,
        "show_hud": False # Hides waypoints/arrows
    })
    # Phase 1: Solo
    experiment_phases.append({
        "tag": "task1", 
        "name": "Task 1 (No Assist)", 
        "duration": args.time_per_task, 
        "arrow": False, 
        "ghost": False,
        "show_hud": True
    })
    # Phase 2: Arrow
    experiment_phases.append({
        "tag": "task2", 
        "name": "Task 2 (Arrow Assist)", 
        "duration": args.time_per_task, 
        "arrow": True, 
        "ghost": False,
        "show_hud": True
    })
    # Phase 3: Ghost
    experiment_phases.append({
        "tag": "task3", 
        "name": "Task 3 (Ghost Assist)", 
        "duration": args.time_per_task, 
        "arrow": False, 
        "ghost": True,
        "show_hud": True
    })
else:
    # Default Mode (Just runs once based on command line args)
    experiment_phases.append({
        "tag": "flight",
        "name": "Free Flight", 
        "duration": args.time_per_task, 
        "arrow": args.assist_arrow, 
        "ghost": args.assist_ghost, # Note: args.assist_ghost was renamed/removed in previous steps? Use assist-ghost arg.
        "show_hud": True
    })

# Experiment State
phase_idx = 0
current_phase = experiment_phases[phase_idx]
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
    env = gym.make("PyFlyt/Fixedwing-Waypoints-v4", render_mode=render_mode, unordered=args.unordered, max_duration_seconds=3600.0, goal_reach_distance=args.waypoint_dist)
except:
    env = gym.make("PyFlyt/Fixedwing-Waypoints-v0", render_mode=render_mode, unordered=args.unordered, max_duration_seconds=3600.0, goal_reach_distance=args.waypoint_dist)

env = FlattenWaypointEnv(env, context_length=2)

if args.vec_norm:
    stats_path = args.model_path + "_vecnorm.pkl"
    # 1. SB3 requires a VecEnv to use VecNormalize
    # We wrap our single env in a DummyVecEnv
    env = DummyVecEnv([lambda: env])
    
    # 2. Load the statistics (Mean/Variance)
    env = VecNormalize.load(stats_path, env)
    
    # 3. CRITICAL: Turn off training updates! 
    # We only want to USE the stats, not change them.
    env.training = False
    env.norm_reward = False
    
    # Flag to handle the shape difference later
    is_vectorized = True
else:
    is_vectorized = False

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


def update_ghost_plane(p, drone_id, ai_action):
    """
    Overlays a sleek 'Ghost Drone' anchored to your position.
    - VISUAL: High-Vis White Body with Safety Orange tips.
    - LOGIC: Banks 45 degrees to show AI intent.
    """
    global ghost_left_id, ghost_right_id, ghost_tail_id, smoothed_action
    
    # --- CONFIGURATION: HIGH VISIBILITY SCHEME ---
    # Bright White Body (85% Opacity - stands out against dark ground)
    MAIN_COLOR = [1.0, 1.0, 1.0, 0.85] 
    # Safety Orange Wingtips (High contrast against blue sky)
    TIP_COLOR  = [1.0, 0.2, 0.0, 0.9]
    
    # 1. Create Bodies (One-time setup)
    if ghost_left_id is None:
        # A. Main Wing (White)
        wing_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 1.0, 0.03], rgbaColor=MAIN_COLOR)
        
        # B. Wing Tips (Orange - helps see rotation instantly)
        tip_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.21, 0.15, 0.04], rgbaColor=TIP_COLOR)
        
        # C. Fuselage (White)
        fuse_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.1, 0.1], rgbaColor=MAIN_COLOR)
        
        # D. Tail (Orange - for directional clarity)
        tail_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.02, 0.25], rgbaColor=TIP_COLOR)
        
        # Create MultiBodies
        ghost_left_id  = p.createMultiBody(baseVisualShapeIndex=wing_shape)
        ghost_right_id = p.createMultiBody(baseVisualShapeIndex=fuse_shape) # Fuselage
        ghost_tail_id  = p.createMultiBody(baseVisualShapeIndex=tail_shape) # Tail
        
        # Note: We aren't creating a separate body for tips in this simplified version to save performance,
        # but the orange tail serves the same orientation purpose. 
        # If you really want orange wingtips, we would need 2 more bodies. 
        # For now, let's make the TAIL orange, which is very effective for orientation.

        # Disable collisions
        p.setCollisionFilterGroupMask(ghost_left_id, -1, 0, 0)
        p.setCollisionFilterGroupMask(ghost_right_id, -1, 0, 0)
        p.setCollisionFilterGroupMask(ghost_tail_id, -1, 0, 0)
    
    try:
        # 2. Get State
        h_pos, h_orn = p.getBasePositionAndOrientation(drone_id)
        
        # 3. Smooth Input
        alpha = 0.15 
        smoothed_action = (smoothed_action * (1 - alpha)) + (ai_action * alpha)
        
        # 4. Calculate Orientation
        MAX_BANK = 0.78 # 45 degrees
        d_roll  = smoothed_action[0] * MAX_BANK
        d_pitch = -smoothed_action[1] * MAX_BANK 
        d_yaw   = 0.0 
        
        delta_orn = p.getQuaternionFromEuler([d_roll, d_pitch, d_yaw])
        _, ghost_orn = p.multiplyTransforms([0,0,0], h_orn, [0,0,0], delta_orn)
        
        # 5. Position Parts
        center_pos = h_pos 

        def place_part(body_id, local_offset, local_euler=[0,0,0]):
            local_orn = p.getQuaternionFromEuler(local_euler)
            p_pos, p_orn = p.multiplyTransforms(center_pos, ghost_orn, local_offset, local_orn)
            p.resetBasePositionAndOrientation(body_id, p_pos, p_orn)

        # Main Wing
        place_part(ghost_left_id,  [0, 0, 0])  
        
        # Fuselage (Body)
        place_part(ghost_right_id, [0, 0, -0.05]) 
        
        # Tail (Orange!)
        place_part(ghost_tail_id,  [-0.5, 0, 0.15])
        
    except Exception: 
        pass

# --- VISUAL FUNCTIONS ---

def force_pygame_focus():
    """Forces the Pygame window to the front (Windows Only)."""
    if sys.platform.startswith('win'):
        try:
            import ctypes
            hwnd = pygame.display.get_wm_info()['window']
            # Force the window to the foreground
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        except:
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
if is_vectorized:
    obs = env.reset()
else:
    obs, _ = env.reset()

# force_pygame_focus()
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
                accumulated_time = 0.0
                last_capture_time = -100.0
                total_crashes_in_task = 0 # Reset counters
                total_waypoints_in_task = 0
                
                # Reset Env & Ghost
                if is_vectorized:
                    obs = env.reset()
                else:
                    obs, _ = env.reset()
                # force_pygame_focus()
                ghost_left_id = None # Reset Ghost Bodies
                ghost_right_id = None
                ghost_tail_id = None
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
            
        # 2. Get AI Input (With Safety Clip)
        if agent_model:
            raw_ai, _ = agent_model.predict(obs, deterministic=True)
            # CLIP AI ACTION
            if is_vectorized:
                # The AI returns [[roll, pitch...]], we need just [roll, pitch...]
                raw_ai = raw_ai[0]
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
            accumulated_time += dt_sec
            if current_phase["ghost"] and agent_model and drone_id is not None:
                update_ghost_plane(p, drone_id, ai_action)

            current_episode["observations"].append(obs)
            current_episode["actions"].append(final_action.copy()) 
            # --- STEP 4 FIX: Handle Environment Type ---
            if is_vectorized:
                # VecEnv expects a LIST of actions: [action]
                # It returns LISTS of results: ([obs], [rews], [dones], [infos])
                obs, rewards, dones, infos = env.step([final_action])
                
                # Extract the single values for our script
                reward = rewards[0]
                terminated = dones[0]
                truncated = False  # VecEnv handles truncation internally
                info = infos[0]
            else:
                # Standard Environment (Old Style)
                obs, reward, terminated, truncated, info = env.step(final_action)
            # obs, reward, terminated, truncated, info = env.step(final_action)
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
                smoothed_action = np.zeros(4) # Reset smoothing filter

                if is_vectorized:
                    obs = env.reset()
                else:
                    obs, _ = env.reset()
                # force_pygame_focus()
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
                if current_phase["show_hud"]:
                    # Arrow
                    if current_phase["arrow"]:
                        draw_hud_arrow(screen, pos, orn, current_targets, args.unordered)
                    # Ghost
                    if current_phase["ghost"] and agent_model:
                        # Update Ghost Physics (Run this every frame regardless of draw)
                        # We do this in the physics step, but draw here
                        pass 
                        # Note: Ghost drawing is handled by 'update_ghost_plane' visual updates 
                        # which are physically updated in the main loop. 
                        # We just need to ensure the physics update ONLY happens if current_phase['ghost'] is True.

            # --- STANDARD HUD (Non-Experiment) ---
            else:    
                if args.assist_arrow:
                    draw_hud_arrow(screen, pos, orn, current_targets, args.unordered)
                
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
            #         paused = True

except KeyboardInterrupt:
    print("Interrupted.")
finally:
    env.close()
    pygame.quit()
    save_data(session_data, current_episode, args, current_phase["tag"])