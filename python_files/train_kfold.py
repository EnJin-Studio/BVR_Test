import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import os

# === 配置 ===
NPY_DIR = "npy_data"
X_PATH = os.path.join(NPY_DIR, "X.npy")
Y_PATH = os.path.join(NPY_DIR, "y.npy")
SAVE_MODEL_PREFIX = os.path.join(NPY_DIR, "mlp_fold")

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
KFOLDS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 数据加载 ===
X = np.load(X_PATH).astype(np.float32)
y = np.load(Y_PATH).astype(np.float32).reshape(-1, 1)
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
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

# === 交叉验证训练 ===
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
rmse_list = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n[FOLD {fold+1}/{KFOLDS}]")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

    model = MLPRegressor(input_dim=X.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每轮输出验证 RMSE
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                pred = model(xb).cpu().numpy()
                preds.append(pred)
                targets.append(yb.numpy())
        preds = np.vstack(preds)
        targets = np.vstack(targets)
        rmse = np.sqrt(mean_squared_error(targets, preds))
        print(f"[Epoch {epoch:02d}] Val RMSE: {rmse:.4f}")

    # 保存最后一轮模型
    torch.save(model.state_dict(), f"{SAVE_MODEL_PREFIX}_fold{fold+1}.pth")
    print(f"[Saved] Fold {fold+1} model to {SAVE_MODEL_PREFIX}_fold{fold+1}.pth")

    rmse_list.append(rmse)

# === 总结 ===
print("\n=== Cross-Validation Summary ===")
for i, rmse in enumerate(rmse_list):
    print(f"Fold {i+1}: RMSE = {rmse:.4f}")
print(f"Average RMSE: {np.mean(rmse_list):.4f}")
