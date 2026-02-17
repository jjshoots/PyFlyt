import numpy as np
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DATA_ROOT = "flight_data"
FPS = 60
PREDICTION_WINDOW_SEC = 1.0  # Increased to 1.5s to make it "predict into the future" harder/more useful
PREDICTION_STEPS = int(PREDICTION_WINDOW_SEC * FPS)
SEQ_LEN = 60                 # Look at past 1.5s to predict next 1.5s
BATCH_SIZE = 2048            # Bigger batch for faster training
MAX_EPOCHS = 100
PATIENCE = 10                # Early stopping patience

# Train/Test Split
TRAIN_SUBJECTS = [str(i) for i in range(1, 25)]
TEST_SUBJECTS = [str(i) for i in range(25, 31)]

# Device Config
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"   Using Device: {device}")

# --- 1. DATA LOADING (Optimized) ---
def load_data():
    print(">>> 1. Loading Flight Data...")
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    files = glob.glob(os.path.join(DATA_ROOT, "**", "*.npz"), recursive=True)
    print(f"   Found {len(files)} logs. Extracting features...")

    # Statistics
    total_safe = 0
    total_danger = 0

    for fpath in files:
        # Subject ID extraction logic
        parts = fpath.split(os.sep)
        try:
            if 'flight_data' in parts: subj_id = parts[parts.index('flight_data') + 1]
            else: subj_id = parts[1]
        except: continue

        try:
            data = np.load(fpath, allow_pickle=True)
            keys = list(data.keys())
            obs_key = next((k for k in keys if k.endswith('_obs')), None)
            if not obs_key: continue
            
            prefix = obs_key.replace('_obs', '')
            act_key = f"{prefix}_human_act"
            rew_key = next((k for k in keys if k.startswith(prefix) and 'rew' in k), None)
            
            if not act_key or not rew_key: continue

            obs = data[obs_key]
            act = data[act_key]
            rew = data[rew_key]

            # Sync lengths
            min_len = min(len(obs), len(act), len(rew))
            if min_len < PREDICTION_STEPS + SEQ_LEN: continue
            
            obs, act, rew = obs[:min_len], act[:min_len], rew[:min_len]

            # Labeling
            labels = np.zeros(min_len)
            is_crash = np.min(rew[-10:]) < -10.0 # Check last 10 frames for crash
            
            if is_crash:
                # Mark prediction window as danger
                labels[max(0, min_len - PREDICTION_STEPS):] = 1

            # Features
            features = np.hstack([obs, act])
            
            # Bucketing
            if subj_id in TRAIN_SUBJECTS:
                X_train_list.append(features)
                y_train_list.append(labels)
            elif subj_id in TEST_SUBJECTS:
                X_test_list.append(features)
                y_test_list.append(labels)
                
            total_safe += (labels == 0).sum()
            total_danger += (labels == 1).sum()

        except Exception: continue

    # Concatenate
    X_train = np.vstack(X_train_list)
    y_train = np.hstack(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.hstack(y_test_list)
    
    # Calculate Class Weight for Loss function
    pos_weight = total_safe / (total_danger + 1e-5)
    print(f"   Data Loaded. Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   Danger Ratio: {total_danger/(total_safe+total_danger):.4f}")
    print(f"   Calculated Pos_Weight: {pos_weight:.2f}")

    return X_train, y_train, X_test, y_test, pos_weight

# --- 2. SEQUENCE GENERATOR ---
def create_sequences(X, y, seq_len):
    # Stride of 5 to reduce memory but keep coverage
    stride = 5
    xs, ys = [], []
    for i in range(0, len(X) - seq_len, stride):
        xs.append(X[i:(i+seq_len)])
        ys.append(y[i+seq_len]) 
    return np.array(xs), np.array(ys)

# --- 3. MODEL ARCHITECTURE ---
class CrashRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, rnn_type="LSTM"):
        super(CrashRNN, self).__init__()
        self.rnn_type = rnn_type
        
        if rnn_type == "GRU":
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
            
        self.fc = nn.Linear(hidden_dim, 1)
        # No sigmoid here, we use BCEWithLogitsLoss for stability
        
    def forward(self, x):
        # x: (batch, seq, feature)
        out, _ = self.rnn(x)
        # Take last time step
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        return logits

# --- 4. TRAINING LOOP ---
def train_model(X_train_seq, y_train_seq, X_test_seq, y_test_seq, config, pos_weight):
    print(f"\n--- Training {config['type']} (Hidden: {config['hidden']}, Layers: {config['layers']}) ---")
    
    # Convert to Tensor
    train_data = TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train_seq))
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model Setup
    model = CrashRNN(input_dim=33, hidden_dim=config['hidden'], num_layers=config['layers'], rnn_type=config['type']).to(device)
    
    # Loss with Class Weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch).squeeze()
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # Validation Check (Use Test Set as Val for this script)
        model.eval()
        with torch.no_grad():
            X_val = torch.FloatTensor(X_test_seq).to(device)
            y_val = torch.FloatTensor(y_test_seq).to(device)
            val_logits = model(X_val).squeeze()
            val_loss = criterion(val_logits, y_val).item()
            
        print(f"   Ep {epoch+1:03d} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}", end="\r")
        
        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"\n   -> Early stopping at epoch {epoch+1}")
            break
            
    # Load best weights
    model.load_state_dict(best_model_state)
    return model

# --- 5. EVALUATION ---
def evaluate_model(model, X_test_seq, y_test_seq):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_seq).to(device)
        logits = model(X_tensor).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
        
    y_true = y_test_seq
    y_pred = (probs > 0.5).astype(int)
    
    # Calculate PR-AUC (Area Under Precision-Recall Curve) - Best single metric for imbalanced data
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)
    
    print("\n   [Results]")
    print(classification_report(y_true, y_pred, target_names=['Safe', 'Danger']))
    print(f"   PR-AUC: {pr_auc:.4f}")
    
    return pr_auc, model

# --- MAIN ---
if __name__ == "__main__":
    # 1. Load Data (Once)
    X_train, y_train, X_test, y_test, pos_weight = load_data()
    
    # Scale Data
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Create Sequences (Once)
    print(">>> Generating Sequences...")
    X_train_seq, y_train_seq = create_sequences(X_train_s, y_train, SEQ_LEN)
    X_test_seq, y_test_seq = create_sequences(X_test_s, y_test, SEQ_LEN)
    print(f"   Seq Data Shape: {X_train_seq.shape}")

    # 2. Hyperparameter Grid
    configs = [
        {'type': 'LSTM', 'hidden': 64, 'layers': 2},
        {'type': 'LSTM', 'hidden': 128, 'layers': 4},
        {'type': 'GRU',  'hidden': 64, 'layers': 2},
        {'type': 'GRU',  'hidden': 128, 'layers': 4},
    ]
    
    results = []
    
    # 3. Training Loop
    best_overall_auc = 0
    best_overall_model = None
    best_config_name = ""

    for conf in configs:
        model = train_model(X_train_seq, y_train_seq, X_test_seq, y_test_seq, conf, pos_weight)
        auc_score, _ = evaluate_model(model, X_test_seq, y_test_seq)
        
        conf_name = f"{conf['type']}_{conf['hidden']}u_{conf['layers']}L"
        results.append({'name': conf_name, 'auc': auc_score})
        
        if auc_score > best_overall_auc:
            best_overall_auc = auc_score
            best_overall_model = model
            best_config_name = conf_name
            
            # Save Checkpoint
            torch.save(model.state_dict(), f"best_crash_model_{conf_name}.pth")
            joblib.dump(scaler, f"scaler_{conf_name}.joblib")

    # 4. Final Leaderboard
    print("\n>>> LEADERBOARD (PR-AUC) <<<")
    results.sort(key=lambda x: x['auc'], reverse=True)
    for r in results:
        print(f"   {r['name']:<15} | {r['auc']:.4f}")
        
    print(f"\n>>> Winner: {best_config_name}")
    print(f"   Saved to 'best_crash_model_{best_config_name}.pth'")