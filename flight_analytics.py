import numpy as np

class FlightAnalytics:
    def __init__(self, data_buffer, hz=30):
        self.obs = np.array(data_buffer["observations"])
        self.acts = np.array(data_buffer["actions"])
        self.rews = np.array(data_buffer["rewards"])
        self.hz = hz
        self.total_steps = len(self.obs)
        
        # --- INDICES (Based on PyFlyt Flattened Observation) ---
        # 0-2: AngVel, 3-6: Quat, 7-9: LinVel, 10-12: LinPos
        # 13+: Target Deltas (Target_1_X, Target_1_Y, ...)
        self.idx_ang_vel = slice(0, 3)
        self.idx_lin_vel = slice(7, 10)
        self.idx_lin_pos = slice(10, 13)
        # Targets start AFTER the attitude block (Index 23)
        self.idx_target = slice(23, 26)  # <--- CHANGED FROM (13, 16)

    def calculate_all(self):
        print("\n" + "="*50)
        print("          FLIGHT QUALITY REPORT")
        print("="*50)
        
        # --- 1. MISSION METRICS (Time & Score) ---
        captures = np.sum(self.rews >= 90.0) 
        crashes = np.sum(self.rews <= -90.0)
        duration = self.total_steps / self.hz
        
        print(f"MISSION STATUS:")
        print(f"  Time Aloft:      {duration:.2f} s")
        print(f"  Waypoints Hit:   {captures}")
        print(f"  Crashes:         {crashes}")

        # --- 2. ATTITUDE STABILITY (Mean/SD Angular Velocity) ---
        # "Is the ride smooth or jerky?"
        ang_vels = self.obs[:, self.idx_ang_vel]
        ang_mags = np.linalg.norm(ang_vels, axis=1)
        
        print(f"\nSTABILITY (Lower is Better):")
        print(f"  Mean Ang. Vel:   {np.mean(ang_mags):.4f} rad/s")
        print(f"  SD Ang. Vel:     {np.std(ang_mags):.4f} rad/s")

        # --- 3. CROSS-TRACK PROXY (Heading Alignment Error) ---
        # "Did we fly straight at the target or snake around?"
        # Angle between Velocity Vector and Target Vector
        lin_vels = self.obs[:, self.idx_lin_vel]
        target_vecs = self.obs[:, self.idx_target]
        
        # Normalize vectors
        v_norm = np.linalg.norm(lin_vels, axis=1)
        t_norm = np.linalg.norm(target_vecs, axis=1)
        
        # Avoid divide by zero
        v_norm[v_norm < 0.01] = 1.0
        t_norm[t_norm < 0.01] = 1.0
        
        # Dot product angle calculation
        dot = np.sum(lin_vels * target_vecs, axis=1)
        cosine = np.clip(dot / (v_norm * t_norm), -1.0, 1.0)
        heading_errors = np.degrees(np.arccos(cosine))
        
        print(f"\nPRECISION (Heading Error):")
        print(f"  Mean Error:      {np.mean(heading_errors):.2f} deg")
        print(f"  SD Error:        {np.std(heading_errors):.2f} deg")

        # --- 4. ACTION VOLATILITY (% Destabilizing Actions) ---
        # "Did the pilot bang the sticks?"
        # Destabilizing = changing input by > 50% in a single frame (0.03s)
        act_deltas = np.diff(self.acts, axis=0)
        max_jerks = np.max(np.abs(act_deltas), axis=1)
        destabilizing_count = np.sum(max_jerks > 0.5)
        destab_pct = (destabilizing_count / len(max_jerks)) * 100
        
        print(f"\nCONTROL SMOOTHNESS:")
        print(f"  Destabilizing:   {destab_pct:.2f}% (Jerky Inputs)")
        print(f"  RMS Volatility:  {np.sqrt(np.mean(act_deltas**2)):.4f}")

        # --- 5. EFFICIENCY (Path Tortuosity & Energy) ---
        # "Did we take the long way?"
        # Total Distance Flown (Sum of velocity * dt)
        dist_flown = np.sum(v_norm) * (1/self.hz)
        
        # Net Displacement (Start to End)
        start_pos = self.obs[0, self.idx_lin_pos]
        end_pos = self.obs[-1, self.idx_lin_pos]
        displacement = np.linalg.norm(end_pos - start_pos)
        
        # Tortuosity Ratio (1.0 = Straight Line, >1.0 = Curvy)
        tortuosity = dist_flown / max(displacement, 1.0)
        
        # RMS Velocity (Energy metric)
        rms_vel = np.sqrt(np.mean(v_norm**2))

        print(f"\nEFFICIENCY:")
        print(f"  RMS Velocity:    {rms_vel:.2f} m/s")
        print(f"  Path Tortuosity: {tortuosity:.2f} (1.0 is perfect)")
        print(f"  Total Distance:  {dist_flown:.1f} m")
        print("="*50)