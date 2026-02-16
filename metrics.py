import numpy as np
import pandas as pd

# The total number of target waypoints successfully triggered by the aircraft during the episode.
def calculate_waypoints_captured(rewards, threshold=90.0):
    return np.sum(rewards >= threshold)

# Binary flag indicating if the pilot completed the entire course (1 = Success, 0 = Fail).
def calculate_success(infos):
    if len(infos) == 0: return 0
    # Checks the last frame's info dict for the completion flag
    # Assumes PyFlyt standard key 'env_complete'
    return 1 if infos[-1].get('env_complete', False) else 0


# Binary flag indicating if the flight ended in a collision.
def calculate_crashes(infos):
    if len(infos) == 0: return 0
    # Checks the last frame for collision flag
    # Assumes PyFlyt standard key 'collision'
    return 1 if infos[-1].get('collision', False) else 0

# The count of times the aircraft entered the "Warning Zone" of the active waypoint 
# but exited without capturing it (failed attempts).
def calculate_near_misses(positions, waypoints, rewards, warn_dist=10.0):
    if len(waypoints) == 0: return 0
    
    near_miss_count = 0
    active_idx = 0
    in_zone = False # State flag: Are we currently inside the bubble?
    
    for t, pos in enumerate(positions):
        # Stop if we have finished all waypoints
        if active_idx >= len(waypoints): break
        
        # 1. Check for Capture Event (Reward spike)
        if rewards[t] >= 90.0:
            # We captured it! 
            # Even if we were 'in_zone', this is a success, not a miss.
            active_idx += 1
            in_zone = False # Reset state for the NEXT waypoint
            continue
            
        # 2. Check Distance to the CURRENT Active Waypoint
        target = waypoints[active_idx]
        dist = np.linalg.norm(pos - target)
        
        if dist < warn_dist:
            # We are inside the warning bubble
            if not in_zone:
                in_zone = True # Mark entry
        else:
            # We are outside the warning bubble
            if in_zone:
                # We WERE inside, but now we left, and we did NOT capture it.
                # This counts as a "Missed Pass" or "Failed Approach"
                near_miss_count += 1
                in_zone = False # Reset state
                
    return near_miss_count

# The number of frames where the aircraft flew dangerously close to the ground (e.g., Altitude < 2m) or breached the flight ceiling without actually crashing.
def calculate_near_crashes(positions, floor_threshold=2.0, ceiling_threshold=19.0):
    # PyFlyt Ground is usually Z=0
    altitudes = positions[:, 2] # Z axis

    # Count frames where we are dangerously low but positive (alive)
    low_danger = np.sum((altitudes > 0.05) & (altitudes < floor_threshold))

    # Optional: Count frames dangerously high (Ceiling breach risk)
    high_danger = np.sum(altitudes > ceiling_threshold)

    return low_danger + high_danger


# The total duration of the episode, valid only if the task was successfully completed.
def calculate_time_to_completion(duration, is_success):
    if not is_success:
        return np.nan  # Or None
    return duration

# The average perpendicular distance between the aircraft and the ideal straight-line path between the previous and active waypoint.
def calculate_cte(positions, waypoints, rewards):
    if len(waypoints) == 0: return 0.0
    cte_sum, count = 0.0, 0
    prev_wp = np.array([0.0, 0.0, 10.0]) # Start
    active_idx = 0

    for t, pos in enumerate(positions):
        if active_idx >= len(waypoints): break
        if rewards[t] >= 90.0:
            prev_wp = waypoints[active_idx]
            active_idx += 1
            continue

        path_vec = waypoints[active_idx] - prev_wp
        drone_vec = pos - prev_wp
        path_len = np.linalg.norm(path_vec)

        if path_len < 1e-3: dist = np.linalg.norm(pos - waypoints[active_idx])
        else: dist = np.linalg.norm(np.cross(path_vec, drone_vec)) / path_len

        cte_sum += dist
        count += 1
    return cte_sum / count if count > 0 else 0.0


# The total Euclidean distance flown by the aircraft.
def calculate_flight_distance(positions):
    if len(positions) < 2: return 0.0
    return np.sum(np.linalg.norm(positions[1:] - positions[:-1], axis=1))

# The percentage of flight time where the aircraft’s bank angle exceeded 90 degrees (inverted).
def calculate_inverted_time(quaternions):
    # Quat [x, y, z, w]. Up_z = 1 - 2(x^2 + y^2)
    qx, qy = quaternions[:, 0], quaternions[:, 1]
    up_z = 1.0 - 2.0 * (qx**2 + qy**2)
    return (np.sum(up_z < 0.0) / len(quaternions)) * 100.0

# The maximum load factor experienced, estimated using forward speed and pitch rate.
def calculate_max_g_force(lin_vel, ang_vel):
    """
    lin_vel: [u, v, w] (Forward, Right, Down speed in body frame)
    ang_vel: [p, q, r] (Roll, Pitch, Yaw rates in body frame)
    """
    if len(lin_vel) == 0: return 1.0
    
    g = 9.81
    
    # Extract Forward Speed (u) and Pitch Rate (q)
    # In fixed-wing, G-force is dominated by Pitching maneuvers (Lift)
    u = lin_vel[:, 0]  # Forward speed
    q = ang_vel[:, 1]  # Pitch rate (rad/s)
    
    # Calculate Vertical Load Factor (n_z)
    # Formula: n = (Centripetal_Accel / g) + 1
    # We add 1.0 because straight-and-level flight is 1G.
    # Pulling up (+q) increases Gs; Pushing down (-q) decreases Gs.
    vertical_gs = (u * q) / g + 1.0
    
    # Optional: Add Lateral Gs from Yaw (sideslip forces)
    # v = lin_vel[:, 1]
    # r = ang_vel[:, 2]
    # lateral_gs = (u * r) / g
    # total_gs = np.sqrt(vertical_gs**2 + lateral_gs**2)

    # Return the maximum absolute G-force experienced
    return np.max(np.abs(vertical_gs))

# The Shannon entropy of the pilot's joystick inputs, representing "control disorder."
from scipy.stats import entropy
def calculate_control_entropy(actions):
    # Magnitude of Roll/Pitch stick deflection
    magnitudes = np.linalg.norm(actions[:, :2], axis=1)
    hist, _ = np.histogram(magnitudes, bins=20, range=(0, 1), density=True)
    return entropy(hist + 1e-10, base=2)

# The count of rapid stick reversals (sign flips) in the control input derivative.
def calculate_pio(action_channel):
    deltas = action_channel[1:] - action_channel[:-1]
    signs = np.sign(deltas)
    signs[np.abs(deltas) < 0.01] = 0 # Filter noise
    signs_clean = signs[signs != 0]
    if len(signs_clean) < 2: return 0
    return np.sum(signs_clean[1:] * signs_clean[:-1] < 0)

# The percentage of time the control stick is saturated (>90% deflection).
def calculate_volatility(actions):
    is_bang = np.any(np.abs(actions[:, :2]) > 0.9, axis=1)
    return (np.sum(is_bang) / len(actions)) * 100.0

# The variance of the specific energy state, measuring energy management stability.
def calculate_energy_variance(positions, lin_vel):
    G = 9.81
    h = positions[:, 2]
    v = np.linalg.norm(lin_vel, axis=1)
    Es = h + (v**2) / (2 * G)
    return np.var(Es)

# The cosine similarity between the Human's control vector and the AI's control vector.
def calculate_trust_cosine(human_act, ai_act):
    h, a = human_act[:, :2], ai_act[:, :2]
    dot = np.sum(h * a, axis=1)
    norms = (np.linalg.norm(h, axis=1) * np.linalg.norm(a, axis=1)) + 1e-6
    return np.mean(dot / norms)

# The time lag that maximizes the cross-correlation between Human and AI signals.
from scipy.signal import correlate
def calculate_latency(h_sig, a_sig, fps=60.0):
    # Normalize
    h_norm = (h_sig - np.mean(h_sig)) / (np.std(h_sig) + 1e-6)
    a_norm = (a_sig - np.mean(a_sig)) / (np.std(a_sig) + 1e-6)
    corr = correlate(h_norm, a_norm, mode='full')
    lags = np.arange(-len(h_norm) + 1, len(h_norm))
    lag_frames = lags[np.argmax(corr)]
    return (lag_frames / fps) * 1000.0 # ms

# The difference in control magnitude between the Human and the AI. Positive values indicate the Human is pushing harder (aggressive); negative values indicate the Human is pushing softer (timid) than the AI.
def calculate_trust_intensity_gap(human_act, ai_act):
    # Calculate magnitudes for Roll/Pitch (indices 0, 1)
    mag_h = np.linalg.norm(human_act[:, :2], axis=1)
    mag_a = np.linalg.norm(ai_act[:, :2], axis=1)

    # Return average difference
    return np.mean(mag_h - mag_a)

