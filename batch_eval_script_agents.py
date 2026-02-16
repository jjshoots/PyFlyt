import pandas as pd
import numpy as np
import subprocess
import glob
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# --- CONFIGURATION ---
# List the filenames of the agents you want to test (NO .zip extension)
AGENTS_TO_TEST = ["fixedwing_agent", "airl_agent_v1"] 
HUMAN_DATA_PATH = "metrics_per_flight.csv"
NUM_EPISODES = 30  # Episodes per agent
RENDER_MODE = "none" # 'human' to watch, 'none' for speed

# --- METRIC CALCULATOR ---
def calculate_flight_metrics(npz_file):
    """
    Reads the raw .npz data from test_easy.py and calculates metrics.
    """
    try:
        data = np.load(npz_file, allow_pickle=True)
        
        # Extract Arrays (Robust get)
        # Note: Adjust keys if your npz structure differs (e.g. 'obs', 'actions')
        actions = data.get('action', [])
        rewards = data.get('reward', [])
        
        if len(actions) == 0: return None

        # 1. Waypoints (Score)
        # Estimate based on reward spikes (assuming +10 for waypoint)
        waypoints = np.sum(rewards > 5.0)
        
        # 2. Crash Detection (Large negative reward at end)
        crashed = 1 if rewards[-1] < -10 else 0 
        
        # 3. Energy Variance (Physical Effort)
        energy_variance = np.var(actions, axis=0).mean()
            
        # 4. Action Volatility (Jerky Inputs)
        action_vol = np.sum(np.abs(np.diff(actions, axis=0))) / len(actions)
            
        # 5. PIO (Pilot Induced Oscillation) - Pitch Reversals
        pitch_actions = actions[:, 1] # Assuming Index 1 is Pitch
        reversals = np.sum(np.diff(np.sign(np.diff(pitch_actions))) != 0)
        pio_score = reversals / (len(actions) / 30.0) # Reversals per second

        return {
            "Waypoints": waypoints,
            "Crashed": crashed,
            "Energy_Variance": energy_variance,
            "Action_Vol_Bang": action_vol,
            "PIO_Count_Pitch": pio_score,
            "Duration_Sec": len(actions) / 30.0
        }
        
    except Exception as e:
        print(f"[Error] Could not process {npz_file}: {e}")
        return None

# --- BATCH RUNNER ---
def run_multi_agent_batch():
    all_results = []
    
    print(f">>> Starting Evaluation for {len(AGENTS_TO_TEST)} Agents...")
    print(f">>> Config: {NUM_EPISODES} Episodes per Agent. Mode: {RENDER_MODE}\n")
    
    for agent_name in AGENTS_TO_TEST:
        print(f"--- Testing Agent: {agent_name} ---")
        
        for i in range(NUM_EPISODES):
            print(f"   [Ep {i+1}/{NUM_EPISODES}] Flying...", end="\r")
            
            # Capture start time to identify the new file
            start_time = time.time()
            
            # Run test_easy.py
            cmd = [
                "python", "test_easy.py", 
                "--pilot", "agent", 
                "--model-path", agent_name,
                "--render-mode", RENDER_MODE
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"\n[!] Error in Ep {i+1}: {result.stderr}")
                continue
                
            # Find the newest .npz file
            list_of_files = glob.glob('*.npz') 
            if not list_of_files:
                print("\n[!] No NPZ file found.")
                continue
                
            latest_file = max(list_of_files, key=os.path.getctime)
            
            # Verify file is new
            if os.path.getctime(latest_file) < start_time:
                print(f"\n[!] Warning: Latest file {latest_file} seems old. Skipping.")
                continue
                
            # Calculate Metrics
            metrics = calculate_flight_metrics(latest_file)
            if metrics:
                metrics['Episode'] = i + 1
                metrics['Condition'] = agent_name # Tag with Agent Name
                all_results.append(metrics)
        
        print(f"\n   -> Agent {agent_name} Complete.\n")
            
    # Save Combined Results
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_df.to_csv("multi_agent_metrics.csv", index=False)
        print(f">>> All Evaluations Complete. Saved to 'multi_agent_metrics.csv'.")
        return final_df
    else:
        print("[!] No data collected.")
        return None

# --- COMPARISON ---
def compare_all_groups(agent_df):
    print("\n>>> Comparing Humans vs. All Agents...")
    
    # 1. Load Human Data
    try:
        human_df = pd.read_csv(HUMAN_DATA_PATH)
        # Standardize Condition Name
        human_baseline = human_df[human_df['Condition'].astype(str).str.contains('Alone', case=False)].copy()
        human_baseline['Condition'] = 'Human (Alone)'
        
        # Filter for relevant columns
        cols = ['Condition', 'Waypoints', 'Energy_Variance', 'PIO_Count_Pitch', 'Action_Vol_Bang']
        human_data = human_baseline[cols]
        
    except FileNotFoundError:
        print(f"[!] Human data file '{HUMAN_DATA_PATH}' not found. Plotting agents only.")
        human_data = pd.DataFrame()

    # 2. Combine Data
    # Ensure agent_df has the same columns
    if not human_data.empty:
        combined_df = pd.concat([human_data, agent_df[cols]], ignore_index=True)
    else:
        combined_df = agent_df[cols]

    # 3. Statistical Analysis (T-Test against Human)
    if not human_data.empty:
        print(f"\n{'Agent':<20} | {'Metric':<15} | {'Diff (Agent-Human)':<20} | {'P-Value':<10}")
        print("-" * 75)
        
        human_means = human_data.mean(numeric_only=True)
        
        for agent in AGENTS_TO_TEST:
            agent_sub = agent_df[agent_df['Condition'] == agent]
            if agent_sub.empty: continue
            
            for m in ['Waypoints', 'Energy_Variance']:
                t, p = ttest_ind(agent_sub[m], human_data[m], equal_var=False)
                diff = agent_sub[m].mean() - human_means[m]
                print(f"{agent:<20} | {m:<15} | {diff:+.4f}             | {p:.4f}")

    # 4. Plotting
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Performance (Waypoints)
    plt.subplot(1, 2, 1)
    sns.barplot(data=combined_df, x='Condition', y='Waypoints', errorbar='se', palette='magma')
    plt.title("Performance Comparison (Score)")
    plt.xticks(rotation=15)
    
    # Plot 2: Control Style (Energy)
    plt.subplot(1, 2, 2)
    sns.barplot(data=combined_df, x='Condition', y='Energy_Variance', errorbar='se', palette='viridis')
    plt.title("Control Strategy (Energy Variance)")
    plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.savefig("multi_agent_comparison.png")
    print("\n-> Saved plot 'multi_agent_comparison.png'")

if __name__ == "__main__":
    # 1. Run Batch
    results_df = run_multi_agent_batch()
    
    # 2. Compare
    if results_df is not None:
        compare_all_groups(results_df)