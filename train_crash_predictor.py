import numpy as np
import pandas as pd
import glob
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
DATA_ROOT = "flight_data"
FPS = 30  # Assumed Hz
PREDICTION_WINDOW_SEC = 1.0  # Lead time
PREDICTION_STEPS = int(PREDICTION_WINDOW_SEC * FPS)

# Train/Test Split by Subject ID
TRAIN_SUBJECTS = [str(i) for i in range(1, 25)] # Subjects 1-24
TEST_SUBJECTS = [str(i) for i in range(25, 31)] # Subjects 25-30

# Obs (29) + Action (4)
INPUT_DIM = 29 + 4 

# --- 1. DATA LOADER ---
def load_and_process_data():
    print(">>> 1. Loading Flight Data (Robust Mode)...")
    
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    files = glob.glob(os.path.join(DATA_ROOT, "**", "*.npz"), recursive=True)
    print(f"   Found {len(files)} log files.")
    
    stats = {"Safe_Frames": 0, "Danger_Frames": 0, "Crashes_Found": 0}

    for fpath in files:
        # Extract Subject ID
        parts = fpath.split(os.sep)
        try:
            # logic to find subject id (assuming "flight_data/{id}/...")
            if 'flight_data' in parts:
                idx = parts.index('flight_data') + 1
                subj_id = parts[idx]
            else:
                subj_id = parts[1]
        except:
            continue

        try:
            data = np.load(fpath, allow_pickle=True)
            keys = list(data.keys())
            
            # Find all observation keys (e.g., 'ep_0_obs', 'ep_1_obs')
            obs_keys = [k for k in keys if k.endswith('_obs')]
            
            for o_key in obs_keys:
                # Extract prefix (e.g., 'ep_0')
                prefix = o_key.replace('_obs', '')
                
                # Construct Action and Reward keys
                a_key = f"{prefix}_human_act"
                
                # Reward key might vary, check common variants
                r_key = f"{prefix}_reward"
                if r_key not in keys: r_key = f"{prefix}_rew"
                if r_key not in keys: continue # Skip if no reward (can't label crash)
                if a_key not in keys: continue # Skip if no action

                # Extract Data
                obs = data[o_key]
                actions = data[a_key]
                rewards = data[r_key]

                # Sync lengths
                min_len = min(len(obs), len(actions), len(rewards))
                obs = obs[:min_len]
                actions = actions[:min_len]
                rewards = rewards[:min_len]

                if min_len < PREDICTION_STEPS: continue

                # --- LABEL ENGINEERING ---
                labels = np.zeros(min_len)
                
                # Check for Crash (Large Negative Reward at end)
                # Ensure we check the last few frames in case of padding
                is_crash = np.min(rewards[-5:]) < -10.0 
                
                if is_crash:
                    stats["Crashes_Found"] += 1
                    start_idx = max(0, min_len - PREDICTION_STEPS)
                    labels[start_idx:] = 1
                
                stats["Safe_Frames"] += (labels == 0).sum()
                stats["Danger_Frames"] += (labels == 1).sum()

                # --- FEATURE ENGINEERING ---
                features = np.hstack([obs, actions])
                
                # Append
                if subj_id in TRAIN_SUBJECTS:
                    X_train.append(features)
                    y_train.append(labels)
                elif subj_id in TEST_SUBJECTS:
                    X_test.append(features)
                    y_test.append(labels)
                
        except Exception as e:
            print(f"   [!] Skipped {fpath}: {e}")
            continue

    if not X_train:
        raise ValueError("No training data found! Check file paths and keys.")

    # Concatenate
    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    X_test = np.vstack(X_test)
    y_test = np.hstack(y_test)
    
    print(f"\n   Data Split Complete:")
    print(f"   Train Rows: {len(y_train)}")
    print(f"   Test Rows:  {len(y_test)}")
    print(f"   Crashes:    {stats['Crashes_Found']}")
    
    return X_train, y_train, X_test, y_test

# --- 2. RANDOM FOREST ---
def train_random_forest(X_train, y_train, X_test, y_test):
    print("\n>>> 2. Training Random Forest...")
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train_s, y_train)
    
    y_pred = rf.predict(X_test_s)
    print("\n[Random Forest Results]")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Danger']))
    
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, "Random Forest")
    
    return rf

# --- 3. LSTM ---
class CrashLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(CrashLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :] 
        prediction = self.sigmoid(self.fc(last_out))
        return prediction

def create_sequences(X, y, seq_len=30):
    # Optimization: Stride to reduce memory usage if needed
    stride = 5 
    xs, ys = [], []
    for i in range(0, len(X) - seq_len, stride):
        xs.append(X[i:(i+seq_len)])
        ys.append(y[i+seq_len]) 
    return np.array(xs), np.array(ys)

def train_lstm(X_train, y_train, X_test, y_test):
    print("\n>>> 3. Training LSTM...")
    SEQ_LEN = 30
    BATCH_SIZE = 1024
    EPOCHS = 5
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print("   Generating sequences...")
    train_x, train_y = create_sequences(X_train_s, y_train, SEQ_LEN)
    test_x, test_y = create_sequences(X_test_s, y_test, SEQ_LEN)
    
    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(train_x), torch.FloatTensor(train_y))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = CrashLSTM(input_dim=INPUT_DIM)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("   Training...")
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x).squeeze()
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"   Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
        
    model.eval()
    with torch.no_grad():
        test_x_tensor = torch.FloatTensor(test_x)
        preds = model(test_x_tensor).squeeze().numpy()
        
    y_pred_bin = (preds > 0.5).astype(int)
    
    print("\n[LSTM Results]")
    print(classification_report(test_y, y_pred_bin, target_names=['Safe', 'Danger']))
    
    cm = confusion_matrix(test_y, y_pred_bin)
    plot_confusion_matrix(cm, "LSTM")
    
    return model

def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{title} Confusion Matrix")
    plt.savefig(f"confusion_{title.lower().replace(' ','_')}.png")

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_and_process_data()
    train_random_forest(X_train, y_train, X_test, y_test)
    train_lstm(X_train, y_train, X_test, y_test)