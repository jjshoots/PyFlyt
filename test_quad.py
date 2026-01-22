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
    print("Warning: Stable Baselines3 not found. Agent pilot will not work.")

# --- ARGUMENTS ---
parser = argparse.ArgumentParser(description="QuadX Drone System: Train, Fly, Record")

# RL & Mode Args
parser.add_argument("--train", action="store_true", help="Training Mode (No Graphics)")
parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], default="PPO", help="RL Algorithm")
parser.add_argument("--steps", type=int, default=100000, help="Training timesteps")
parser.add_argument("--pilot", type=str, choices=["human", "agent"], default="human", help="Who flies?")
parser.add_argument("--model_path", type=str, default="quadx_agent", help="Agent file path (no ext)")
parser.add_argument("--render-mode", type=str, choices=["human", "none"], default="human", help="Render Mode")

# Assist Args
parser.add_argument("--assist-agent", action="store_true", help="Spawn an AI 'Ghost Quad' to guide the human")
parser.add_argument("--assist-arrow", action="store_true", help="Show HUD 'Navigation Arrow' to target")

# Visual & Sim Args
parser.add_argument("--zone", type=float, default=0.0, help="Zone Radius (0 = Auto)")
parser.add_argument("--disable-hud", action="store_true", help="Disable ALL HUD")
parser.add_argument("--show-data", action="store_true", help="Show text telemetry")
parser.add_argument("--unordered", action="store_true", help="Use unordered waypoints")
args = parser.parse_args()

# --- CONFIG (LOGITECH EXTREME 3D PRO) ---
AXIS_ROLL, AXIS_PITCH, AXIS_YAW, AXIS_THROTTLE = 0, 1, 2, 3
BTN_PAUSE, BTN_RESET = 0, 1  # Trigger=0, Thumb=1

# QUADCOPTER SENSITIVITY
EXPO_VALUE = 0.0      # 0.0 = Linear
MAX_RATE = 1.0        # Max Angular Rate (Rad/s)
HOVER_THRUST = 0.6    # Approx hover point

# --- DATA RECORDING & SESSION SETUP ---
output_dir = "flight_data"
os.makedirs(output_dir, exist_ok=True)
session_data = []
current_episode = {"observations": [], "actions": [], "rewards": [], "terminals": []}

TIME_LIMIT_SECONDS = 300 
start_ticks = pygame.time.get_ticks()

# --- 1. TRAINING MODE ---
if args.train:
    print(f"\n=== TRAINING MODE ===")
    try:
        env = gym.make("PyFlyt/QuadX-Waypoints-v4", render_mode=None)
    except:
        env = gym.make("PyFlyt/QuadX-Waypoints-v0", render_mode=None)
    
    # Clip actions to prevent crashes during exploration
    env = gym.wrappers.ClipAction(env)
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
    env = gym.make("PyFlyt/QuadX-Waypoints-v4", render_mode=render_mode)
except:
    env = gym.make("PyFlyt/QuadX-Waypoints-v0", render_mode=render_mode)

# Clip Actions for Safety
env = gym.wrappers.ClipAction(env)
env = FlattenWaypointEnv(env, context_length=2)

# Load Agent
agent_model = None
if args.pilot == "agent" or args.assist_agent:
    path = f"{args.model_path}.zip"
    if os.path.exists(path):
        print(f"Loading Agent from {path}...")
        try:
            ModelClass = SAC if args.algo == "SAC" else PPO
            agent_model = ModelClass.load(path)
        except: pass
    else:
        print(f"Warning: Agent {path} not found. Assist disabled.")

# Detect Zone
ZONE_RADIUS = args.zone
if ZONE_RADIUS == 0.0:
    try: ZONE_RADIUS = env.unwrapped.env.flight_dome_size
    except: ZONE_RADIUS = 50.0

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
pygame.display.set_caption(f"QuadX Pilot: {args.pilot.upper()}")

font = pygame.font.SysFont("monospace", 20, bold=True)
small_font = pygame.font.SysFont("monospace", 16)

joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Joystick Detected: {joystick.get_name()}")

# --- GHOST QUAD LOGIC ---
ghost_body_id = None
ghost_arm1_id = None
ghost_arm2_id = None
ghost_smoother = np.zeros(4)

def update_ghost_quad(p, drone_id, ai_action):
    """
    Renders a 'Ghost Quad' (+) that shows AI intent.
    """
    global ghost_body_id, ghost_arm1_id, ghost_arm2_id, ghost_smoother

    # 1. Create the Quad (One-time)
    if ghost_body_id is None:
        # Center Body (White)
        body_s = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 1, 1, 0.5])
        ghost_body_id = p.createMultiBody(baseVisualShapeIndex=body_s)
        
        # Arm 1 (Red - Front/Back)
        a1_s = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.02, 0.02], rgbaColor=[1, 0, 0, 0.5])
        ghost_arm1_id = p.createMultiBody(baseVisualShapeIndex=a1_s)
        
        # Arm 2 (Green - Left/Right)
        a2_s = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, 0.3, 0.02], rgbaColor=[0, 1, 0, 0.5])
        ghost_arm2_id = p.createMultiBody(baseVisualShapeIndex=a2_s)

    try:
        # 2. Get Human State
        h_pos, h_orn = p.getBasePositionAndOrientation(drone_id)
        
        # 3. Smooth AI Action
        alpha = 0.15 
        ghost_smoother = (ghost_smoother * (1 - alpha)) + (ai_action * alpha)
        
        # 4. Calculate Visual Offset (Direct Mapping)
        # Full stick = 45 degree tilt visual
        MAX_VISUAL_TILT = 0.5 # Radians
        
        d_roll  = ghost_smoother[0] * MAX_VISUAL_TILT
        d_pitch = ghost_smoother[1] * MAX_VISUAL_TILT 
        d_yaw   = 0.0 # Locked Yaw
        
        delta_orn = p.getQuaternionFromEuler([d_roll, d_pitch, d_yaw])
        _, target_orn = p.multiplyTransforms([0,0,0], h_orn, [0,0,0], delta_orn)
        
        # Tether 2m ahead
        rot_mat = np.array(p.getMatrixFromQuaternion(h_orn)).reshape(3, 3)
        tether_pos = np.array(h_pos) + rot_mat.dot([2.0, 0, 0])
        
        # 6. Update Physics Bodies
        p.resetBasePositionAndOrientation(ghost_body_id, tether_pos, target_orn)
        p.resetBasePositionAndOrientation(ghost_arm1_id, tether_pos, target_orn)
        p.resetBasePositionAndOrientation(ghost_arm2_id, tether_pos, target_orn)
        
    except Exception:
        pass

def draw_hud_arrow(screen, drone_pos, drone_orn, targets, unordered):
    if not targets or len(targets) == 0: return
    
    if unordered:
        min_dist = float('inf')
        active_target = None
        for t in targets:
            d = np.linalg.norm(t) 
            if d < min_dist: min_dist = d; active_target = t
    else:
        active_target = targets[0]
        min_dist = np.linalg.norm(active_target)

    if active_target is None: return

    rot_mat = np.array(p.getMatrixFromQuaternion(drone_orn)).reshape(3, 3)
    inv_rot = rot_mat.T
    local_vec = inv_rot.dot(np.array(active_target))
    
    cx, cy = WINDOW_W // 2, WINDOW_H // 2
    scale = 800.0
    norm = np.linalg.norm(local_vec)
    if norm < 0.1: return
    direction = local_vec / norm
    
    dx, dy = -direction[1], -direction[2]
    angle = math.atan2(dy, dx)

    arrow_x = cx + (dx * scale)
    arrow_y = cy + (dy * scale)
    hud_radius = 350
    screen_dist = math.sqrt((arrow_x - cx)**2 + (arrow_y - cy)**2)
    
    if screen_dist > hud_radius:
        ratio = hud_radius / screen_dist
        arrow_x = cx + (arrow_x - cx) * ratio
        arrow_y = cy + (arrow_y - cy) * ratio

    size = 20
    points = [(size, 0), (-size, -size * 0.6), (-size, size * 0.6)]
    rot_points = []
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    for px, py in points:
        rot_points.append((arrow_x + (px * cos_a - py * sin_a), arrow_y + (px * sin_a + py * cos_a)))

    pygame.draw.polygon(screen, (255, 0, 255), rot_points)
    pygame.draw.polygon(screen, (255, 255, 255), rot_points, 2)
    screen.blit(font.render(f"{min_dist:.0f}m", True, (255, 255, 255)), (arrow_x - 20, arrow_y + 30))

def get_drone_state(env):
    try:
        drone = env.unwrapped.env.drones[0]
        pos, orn = p.getBasePositionAndOrientation(drone.Id)
        euler = p.getEulerFromQuaternion(orn)
        return drone.Id, pos, orn, euler
    except: return None, None, None, None

def render_camera(drone_id, pos, orn):
    if drone_id is None: return None
    try:
        # CHASE CAMERA: Higher and Closer for Quad
        rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        cam_pos = np.array(pos) + rot_mat.dot([-2.0, 0, 1.0]) # 2m back, 1m up
        cam_target = np.array(pos)
        view_matrix = p.computeViewMatrix(cam_pos, cam_target, rot_mat.dot([0, 0, 1]))
        proj_matrix = p.computeProjectionMatrixFOV(90, float(MAIN_RENDER_W)/MAIN_RENDER_H, 0.1, 1000.0)
        _, _, rgb, _, _ = p.getCameraImage(width=MAIN_RENDER_W, height=MAIN_RENDER_H, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        return pygame.surfarray.make_surface(np.transpose(np.array(rgb, dtype=np.uint8).reshape(MAIN_RENDER_H, MAIN_RENDER_W, 4)[:, :, :3], (1, 0, 2)))
    except: return None

def draw_radar(screen, drone_pos, drone_yaw, targets, zone_radius):
    RADAR_SIZE, CENTER = 200, (WINDOW_W - 120, WINDOW_H - 120)
    PX_PER_METER = (RADAR_SIZE / 2) / (zone_radius * 1.5)
    s = pygame.Surface((RADAR_SIZE, RADAR_SIZE), pygame.SRCALPHA)
    pygame.draw.circle(s, (0, 40, 0, 200), (RADAR_SIZE//2, RADAR_SIZE//2), RADAR_SIZE // 2)
    screen.blit(s, (CENTER[0] - RADAR_SIZE//2, CENTER[1] - RADAR_SIZE//2))
    pygame.draw.circle(screen, (150, 150, 150), CENTER, RADAR_SIZE // 2, 2)
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
            
        for i, p in enumerate(radar_points):
            color = (0, 255, 255) if i == 0 else (255, 255, 0)
            pygame.draw.circle(screen, color, p, 5)

def save_data(session_data, incomplete_episode):
    if len(incomplete_episode["observations"]) > 0:
        session_data.append(incomplete_episode)
    if len(session_data) == 0: return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("flight_data", f"quad_log_{ts}.npz")
    save_dict = {}
    for i, ep in enumerate(session_data):
        save_dict[f"ep_{i}_obs"] = ep["observations"]
        save_dict[f"ep_{i}_act"] = ep["actions"]
        save_dict[f"ep_{i}_rew"] = ep["rewards"]
    np.savez(filename, **save_dict)
    print(f"\nLog Saved: {filename}")

# --- MAIN LOOP ---
clock = pygame.time.Clock()
print("Resetting environment...")
obs, _ = env.reset()
print("Env Ready.")

# Count Targets
total_targets = 0
if hasattr(env.unwrapped, "waypoints") and hasattr(env.unwrapped.waypoints, "targets"):
    total_targets = len(env.unwrapped.waypoints.targets)

control_smoother = np.zeros(4) 
current_targets = []
paused = True 
running = True

try:
    while running:
        clock.tick(30)
        elapsed_sec = (pygame.time.get_ticks() - start_ticks) / 1000.0
        if elapsed_sec > TIME_LIMIT_SECONDS: running = False
        
        drone_id, pos, orn, euler = get_drone_state(env)
        pygame.event.pump() 
        
        # --- INPUTS ---
        human_action = np.array([0.0, 0.0, 0.0, 0.0])
        ai_action = np.array([0.0, 0.0, 0.0, 0.0])

        if joystick:
            # AXIS MAPPING for QuadX
            # Roll (0), Pitch (1), Yaw (2), Thrust (3 - Slider)
            r_roll = joystick.get_axis(AXIS_ROLL)
            r_pitch = joystick.get_axis(AXIS_PITCH)
            r_yaw = joystick.get_axis(AXIS_YAW)
            r_thr = joystick.get_axis(AXIS_THROTTLE)
            
            def expo(v, e): return (v**3 * e) + (v * (1-e))
            
            # Quad Actions: [RollRate, PitchRate, YawRate, Thrust]
            # Thrust: Logitech Slider is -1 (Up) to 1 (Down). We want -1 maps to 1.0 (Full), 1 maps to 0.0 (Off)
            # Actually PyFlyt Thrust is usually [-1, 1] or [0, 1]. Let's assume [-1, 1] range for env step.
            # But usually 0 is idle. Let's map slider -1 (UP) to 1.0, and 1 (DOWN) to -1.0.
            
            thrust_val = -r_thr # Simple invert. Up (-1) becomes 1.0. Down (1) becomes -1.0.
            
            raw_human = np.array([
                expo(r_roll, EXPO_VALUE) * MAX_RATE,
                expo(r_pitch, EXPO_VALUE) * MAX_RATE, # Check invert if needed
                expo(r_yaw, EXPO_VALUE) * MAX_RATE,
                thrust_val
            ])
            human_action = np.clip(raw_human, -1.0, 1.0)

        # AI INPUT
        if agent_model:
            raw_ai, _ = agent_model.predict(obs, deterministic=True)
            ai_action = np.clip(raw_ai, -1.0, 1.0)
        
        # --- CONTROL SELECTION ---
        if args.pilot == "agent":
            alpha = 0.15 
            control_smoother = (control_smoother * (1 - alpha)) + (ai_action * alpha)
            final_action = control_smoother
        else:
            final_action = human_action.copy()
            control_smoother = final_action.copy()

        # --- STEP ---
        if not paused:
            if args.assist_agent and agent_model and drone_id:
                update_ghost_quad(p, drone_id, ai_action)

            current_episode["observations"].append(obs)
            current_episode["actions"].append(final_action.copy()) 
            obs, reward, terminated, truncated, _ = env.step(final_action)
            current_episode["rewards"].append(reward)
            current_episode["terminals"].append(terminated or truncated)

            if hasattr(env.unwrapped, "waypoints"):
                current_targets = [t - pos for t in env.unwrapped.waypoints.targets]
            else: current_targets = []

            if terminated or truncated:
                session_data.append(current_episode)
                print(f"Flight Completed. Targets Reached: {np.sum(np.array(current_episode['rewards']) >= 90.0)}")
                current_episode = {"observations": [], "actions": [], "rewards": [], "terminals": []}
                
                # Reset Ghosts
                ghost_body_id = None
                ghost_smoother = np.zeros(4)
                control_smoother = np.zeros(4)
                
                obs, _ = env.reset()
                paused = True

        # --- RENDER ---
        screen.fill((0, 0, 0))
        main_surf = render_camera(drone_id, pos, orn)
        if main_surf:
            screen.blit(pygame.transform.scale(main_surf, (WINDOW_W, WINDOW_H)), (0, 0))

        if not args.disable_hud:
            if args.assist_arrow:
                draw_hud_arrow(screen, pos, orn, current_targets, args.unordered)
            
            draw_radar(screen, pos, euler[2], current_targets, ZONE_RADIUS)
            
            # Telemetry Box
            if args.show_data and pos is not None:
                BOX_X, BOX_Y = 20, 80
                s = pygame.Surface((300, 180)); s.set_alpha(150); s.fill((0, 0, 0))
                screen.blit(s, (BOX_X, BOX_Y))
                pygame.draw.rect(screen, (255, 255, 255), (BOX_X, BOX_Y, 300, 180), 2)
                
                num_captured = total_targets - len(current_targets)
                dist = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
                
                # Thrust Bar
                thr_pct = (final_action[3] + 1.0) / 2.0 * 100 # Map -1..1 to 0..100%
                
                lines = [
                    f"PILOT: {args.pilot.upper()}",
                    f"ALGO:  {args.algo}",
                    f"ALT:   {pos[2]:.1f} m",
                    f"DIST:  {dist:.1f}m",
                    f"THR:   {thr_pct:.0f}%",
                    f"GOALS: {num_captured} / {total_targets}"
                ]
                for i, line in enumerate(lines):
                    color = (0, 255, 0)
                    screen.blit(small_font.render(line, True, color), (BOX_X + 20, BOX_Y + 15 + (i * 24)))

        if paused:
            screen.blit(font.render("PAUSED (Trigger/Space to Fly)", True, (255, 255, 0)), (WINDOW_W//2 - 150, WINDOW_H//2))
        
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE: paused = not paused
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == BTN_PAUSE: paused = not paused
                if event.button == BTN_RESET: 
                    ghost_body_id = None
                    obs, _ = env.reset()
                    paused = True

except KeyboardInterrupt:
    print("Interrupted.")
finally:
    env.close()
    pygame.quit()
    save_data(session_data, current_episode)