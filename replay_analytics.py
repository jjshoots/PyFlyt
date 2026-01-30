import numpy as np
import argparse
import os
from flight_analytics import FlightAnalytics

# --- ARGUMENTS ---
parser = argparse.ArgumentParser(description="Generate Report from Saved Flight Data")
parser.add_argument("file", type=str, help="Path to the .npz file (e.g. flight_data/log_20260129.npz)")
args = parser.parse_args()

def generate_report_from_file(filename):
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found.")
        return

    print(f"Loading data from: {filename}...")
    try:
        data = np.load(filename)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Reconstruct the 'combined_buffer' 
    combined_buffer = {
        "observations": [],
        "actions": [],
        "human_actions": [],
        "ai_actions": [],
        "rewards": [],
        "terminals": [],
        "boundary_hits": []  # <--- NEW FIELD
    }
    
    # 1. Count Episodes
    episode_indices = set()
    for key in data.files:
        if key.startswith("ep_") and "_obs" in key:
            # Extract index "0" from "ep_0_obs"
            try:
                idx = int(key.split("_")[1])
                episode_indices.add(idx)
            except: pass
            
    sorted_indices = sorted(list(episode_indices))
    print(f"Found {len(sorted_indices)} episodes.")
    
    # 2. Merge Data
    for i in sorted_indices:
        # Required Fields
        if f"ep_{i}_obs" in data:
            combined_buffer["observations"].extend(data[f"ep_{i}_obs"])
            combined_buffer["actions"].extend(data[f"ep_{i}_act"])
            combined_buffer["rewards"].extend(data[f"ep_{i}_rew"])
        
        # Optional Fields
        if f"ep_{i}_human_act" in data:
            combined_buffer["human_actions"].extend(data[f"ep_{i}_human_act"])
        if f"ep_{i}_ai_act" in data:
            combined_buffer["ai_actions"].extend(data[f"ep_{i}_ai_act"])
            
        # <--- NEW: Load Boundary Hits --->
        if f"ep_{i}_wall" in data:
            combined_buffer["boundary_hits"].extend(data[f"ep_{i}_wall"])
            
    # 3. Run Analytics
    print("\n" + "="*50)
    print(f"       REGENERATED SESSION REPORT")
    print("="*50)
    
    # Calculate Custom Metrics (Since FlightAnalytics might not have them)
    total_time = len(combined_buffer["rewards"]) / 30.0 # Assuming 30Hz
    total_wall_hits = int(sum(combined_buffer["boundary_hits"])) if combined_buffer["boundary_hits"] else 0
    
    print(f"GLOBAL SESSION TOTALS:")
    print(f"  Total Time:      {total_time:.1f} s")
    print(f"  Total Wall Hits: {total_wall_hits}")  # <--- PRINT METRIC
    
    # Run Standard Analytics
    try:
        analytics = FlightAnalytics(combined_buffer)
        analytics.calculate_all()
    except Exception as e:
        print(f"\nNote: Standard analytics could not run ({e})")
        print("This might happen if 'flight_analytics.py' is missing or incompatible.")

if __name__ == "__main__":
    generate_report_from_file(args.file)