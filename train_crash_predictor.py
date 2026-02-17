import numpy as np
import pandas as pd
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import joblib 
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
DATA_ROOT = "flight_data"
FPS = 60
PREDICTION_WINDOW_SEC = 0.5 
PREDICTION_STEPS = int(PREDICTION_WINDOW_SEC * FPS) # 60 Frames

# Train/Test Split
TRAIN_SUBJECTS = [i for i in range(1, 25)] 
TEST_SUBJECTS = [i for i in range(25, 31)]

# Feature Selection Indices (Physics + Actions)
KEEP_OBS_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# --- 1. DATA LOADER ---
def load_and_process_data():
    print(">>> 1. Loading Flight Data (Filtered)...")
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    target_tasks = ['task1', 'task_arrow', 'task_ghost']
    files = []
    
    for task in target_tasks:
        pattern = os.path.join(DATA_ROOT, "*", "session1", task, "*.npz")
        found = glob.glob(pattern)
        files.extend(found)
        
    print(f"   Found {len(files)} log files matching 'session1' & tasks {target_tasks}")
    
    stats = {"Crashes_Found": 0}

    for fpath in files:
        parts = fpath.split(os.sep)
        try:
            if 'flight_data' in parts:
                idx = parts.index('flight_data') + 1
                subj_id = int(parts[idx])
            else:
                subj_id = parts[1] 
        except:
            continue

        try:
            data = np.load(fpath, allow_pickle=True)
            keys = list(data.keys())
            obs_keys = [k for k in keys if k.endswith('_obs')]
            for o_key in obs_keys:
                prefix = o_key.replace('_obs', '')
                a_key = f"{prefix}_act"
                r_key = f"{prefix}_rew" 
                if r_key not in keys or a_key not in keys: continue

                obs = data[o_key]
                actions = data[a_key]
                rewards = data[r_key]

                min_len = min(len(obs), len(actions), len(rewards))
                if min_len < PREDICTION_STEPS: continue
                
                obs = obs[:min_len]
                actions = actions[:min_len]
                rewards = rewards[:min_len]

                # Features
                obs = obs[:, KEEP_OBS_INDICES]
                features = np.hstack([obs, actions])

                # Labels
                labels = np.zeros(min_len)
                is_crash = np.min(rewards[-5:]) < -10.0 
                
                if is_crash:
                    stats["Crashes_Found"] += 1
                    start_idx = max(0, min_len - PREDICTION_STEPS)
                    labels[start_idx:] = 1 
                
                if subj_id in TRAIN_SUBJECTS:
                    X_train.append(features)
                    y_train.append(labels)
                elif subj_id in TEST_SUBJECTS:
                    X_test.append(features)
                    y_test.append(labels)
                
        except Exception as e:
            continue
    
    if not X_train: raise ValueError("No data found.")

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)
    
    print(f"   Train Rows: {len(y_train)} | Test Rows: {len(y_test)}")
    print(f"   Crashes:    {stats['Crashes_Found']}")
    print(f"   Input Dim:  {X_train.shape[1]}")
    return X_train, y_train, X_test, y_test

# --- 2. RANDOM FOREST (BASELINE) ---
def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n>>> 2. Training Random Forest (Baseline)...")
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    joblib.dump(scaler, 'crash_scaler.joblib')

    rf = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train_s, y_train)
    
    y_pred = rf.predict(X_test_s)
    
    print("\n[Random Forest Results]")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Danger']))
    
    joblib.dump(rf, 'crash_rf_model.joblib')
    return rf, scaler

# --- 3. FLEXIBLE RNN (LSTM/GRU) ---
class CrashRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, model_type='LSTM'):
        super(CrashRNN, self).__init__()
        self.model_type = model_type.upper()
        
        # ADDED DROPOUT to prevent overfitting to "Safe" states
        if self.model_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
            
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        last_out = out[:, -1, :] 
        return self.fc(last_out) 

def create_sequences(X, y, seq_len):
    xs, ys = [], []
    stride = 5 
    for i in range(0, len(X) - seq_len, stride):
        xs.append(X[i:(i+seq_len)])
        ys.append(y[i+seq_len]) 
    return np.array(xs), np.array(ys)

def train_rnn_experiments(X_train, y_train, X_test, y_test, scaler):
    print("\n>>> 3. Running RNN Experiments (Nuclear Mode)...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using Device: {device}")

    # 1. NUCLEAR WEIGHTING (6x Standard)
    num_safe = (y_train == 0).sum()
    num_danger = (y_train == 1).sum()
    
    # Multiplying by 6.0 effectively says: "One missed crash is as bad as 120 false alarms"
    pos_weight_val = (num_safe / (num_danger + 1e-6)) * 6.0
    
    print(f"   Class Imbalance: {num_safe} Safe vs {num_danger} Danger")
    print(f"   Nuclear Weight: {pos_weight_val:.2f} (6x Standard)")
    
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    SEQ_LEN = PREDICTION_STEPS 
    BATCH_SIZE = 2048 
    
    print(f"   Generating Sequences (Len={SEQ_LEN})...")
    train_x, train_y = create_sequences(X_train_s, y_train, SEQ_LEN)
    test_x, test_y = create_sequences(X_test_s, y_test, SEQ_LEN)
    
    train_dataset = TensorDataset(torch.FloatTensor(train_x).to(device), torch.FloatTensor(train_y).to(device))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    input_dim = X_train.shape[1]
    
    # LSTM Won last time, so we focus on LSTM variants
    configs = [
        ('LSTM', 64, 2),
        ('LSTM', 128, 2),
        ('GRU', 64, 2),
        ('GRU', 128, 2)
    ]
    
    results = {}
    best_recall = 0.0 
    best_model_name = ""

    for (m_type, h_dim, n_layers) in configs:
        config_name = f"{m_type}_H{h_dim}_L{n_layers}"
        print(f"\n   [Training {config_name}]")
        
        model = CrashRNN(input_dim, hidden_dim=h_dim, num_layers=n_layers, model_type=m_type).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(100): 
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                out = model(bx).squeeze()
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        
        model.to('cpu')
        model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(test_x)
            logits = model(test_tensor).squeeze()
            preds = torch.sigmoid(logits).numpy()
            
        print("   --- Threshold Analysis ---")
        
        # Scan lower thresholds to find where Recall beats Random Forest (0.76)
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            y_pred_bin = (preds > thresh).astype(int)
            try:
                report = classification_report(test_y, y_pred_bin, output_dict=True)
                recall = report['1.0']['recall']
                precision = report['1.0']['precision']
                f1 = report['1.0']['f1-score']
                
                print(f"   [Thresh {thresh}] Recall: {recall:.4f} | Prec: {precision:.4f} | F1: {f1:.4f}")
                
                if thresh == 0.3: # Use 0.3 as the benchmark for saving
                    current_metric = recall
                    results[config_name] = recall
                    
                    if current_metric > best_recall:
                        best_recall = current_metric
                        best_model_name = config_name
                        
                        save_path = "best_crash_rnn.pth"
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'config': {'input_dim': input_dim, 'hidden_dim': h_dim, 'num_layers': n_layers, 'model_type': m_type},
                            'threshold': 0.3 # Save optimal threshold
                        }, save_path)
                        print(f"      [New Best Recall Model Saved]")
            except:
                pass

    print("\n>>> Experiment Summary (Danger Class Recall @ 0.3 Thresh):")
    for k, v in results.items():
        print(f"   {k}: {v:.4f}")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_and_process_data()
    rf_model, scaler = train_random_forest(X_train, y_train, X_test, y_test)
    train_rnn_experiments(X_train, y_train, X_test, y_test, scaler)