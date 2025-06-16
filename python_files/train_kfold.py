import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# === 配置项 ===
NPY_DIR = "npy_data"
X_PATH = os.path.join(NPY_DIR, "X.npy")
Y_PATH = os.path.join(NPY_DIR, "view_count_log.npy")
SAVE_MODEL_PREFIX = os.path.join(NPY_DIR, "mlp_fold")

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
KFOLDS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 数据加载 ===
X = np.load(X_PATH).astype(np.float32)
y = np.load(Y_PATH).astype(np.float32).reshape(-1, 1)

print(f"[INFO] X shape: {X.shape}, y shape: {y.shape}")

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)
dataset = TensorDataset(X_tensor, y_tensor)

# === 模型定义 ===
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        self._init_weights()

    def forward(self, x):
        return self.model(x)

    def _init_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

# === K-Fold 训练 ===
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
rmse_list = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n[FOLD {fold+1}]")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE)

    model = MLPRegressor(input_dim=X.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 验证
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                preds = model(xb).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(yb.numpy())
        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        rmse = np.sqrt(mean_squared_error(targets, preds))
        print(f"[Epoch {epoch:02d}] Val RMSE: {rmse:.4f}")

    # 保存模型
    model_path = f"{SAVE_MODEL_PREFIX}{fold+1}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[Saved] Fold {fold+1} model -> {model_path}")
    rmse_list.append(rmse)

# === 结果汇总 ===
print("\n=== K-Fold Summary ===")
for i, rmse in enumerate(rmse_list):
    print(f"Fold {i+1}: RMSE = {rmse:.4f}")
print(f"Average RMSE: {np.mean(rmse_list):.4f}")
