import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Configuration
NPY_DIR = "npy_data"
IMAGE_PATH = os.path.join(NPY_DIR, "picture_features_aligned.npy")      # [N, 512]
TITLE_PATH = os.path.join(NPY_DIR, "title.npy")                         # [N, 768]
UPLOADER_PATH = os.path.join(NPY_DIR, "uploader.npy")                   # [N, 768]
NUMERIC_PATH = os.path.join(NPY_DIR, "numeric_features.npy")            # [N, 3]
Y_PATH = os.path.join(NPY_DIR, "view_count.npy")                        # [N]
SAVE_MODEL_PREFIX = os.path.join("model", "model_fold")

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
KFOLDS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load data
image_feat = np.load(IMAGE_PATH).astype(np.float32)
title_feat = np.load(TITLE_PATH).astype(np.float32)
uploader_feat = np.load(UPLOADER_PATH).astype(np.float32)
numeric_feat = np.load(NUMERIC_PATH).astype(np.float32)
y = np.load(Y_PATH).astype(np.float32).reshape(-1, 1)

assert all(x.shape[0] == y.shape[0] for x in [image_feat, title_feat, uploader_feat, numeric_feat])

print(f"[INFO] Dataset size: {y.shape[0]}")

dataset = TensorDataset(
    torch.tensor(image_feat),
    torch.tensor(title_feat),
    torch.tensor(uploader_feat),
    torch.tensor(numeric_feat),
    torch.tensor(y)
)

# Model
class MultiModalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_branch = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.title_branch = nn.Sequential(
            nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.uploader_branch = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.3)
        )
        self.numeric_branch = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(), nn.Dropout(0.3)
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256 + 128 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, img, title, up, num):
        x1 = self.image_branch(img)
        x2 = self.title_branch(title)
        x3 = self.uploader_branch(up)
        x4 = self.numeric_branch(num)
        fused = torch.cat([x1, x2, x3, x4], dim=1)
        return self.fusion(fused)

# K-Fold Training
kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
rmse_list = []

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"\n[FOLD {fold+1}]")

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE)

    model = MultiModalMLP().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_rmse = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for img, title, up, num, target in train_loader:
            img, title, up, num, target = img.to(DEVICE), title.to(DEVICE), up.to(DEVICE), num.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            pred = model(img, title, up, num)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for img, title, up, num, target in val_loader:
                img, title, up, num = img.to(DEVICE), title.to(DEVICE), up.to(DEVICE), num.to(DEVICE)
                pred = model(img, title, up, num).cpu().numpy()
                all_preds.append(pred)
                all_targets.append(target.numpy())
        preds = np.vstack(all_preds)
        targets = np.vstack(all_targets)
        rmse = np.sqrt(mean_squared_error(targets, preds))
        print(f"[Epoch {epoch:02d}] Val RMSE: {rmse:.4f}")

        y_train_fold = y[train_idx]
        y_train_mean = y_train_fold.mean()
        baseline_pred = np.full_like(targets, y_train_mean)
        baseline_rmse = np.sqrt(mean_squared_error(targets, baseline_pred))
        print(f"[Baseline] RMSE if always predict mean of y_train: {baseline_rmse:.4f}")

        # EarlyStopping 判断
        if rmse < best_rmse:
            best_rmse = rmse
            patience_counter = 0
            torch.save(model.state_dict(), f"{SAVE_MODEL_PREFIX}{fold+1}_state_dict.pt")
            print(f"[Saved] New best model with RMSE {rmse:.4f}")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. Patience {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("[EarlyStopping] No improvement, stopping training.")
                break

    rmse_list.append(best_rmse)

# Summary
print("\n=== K-Fold Summary ===")
for i, rmse in enumerate(rmse_list):
    print(f"Fold {i+1}: RMSE = {rmse:.4f}")
print(f"Average RMSE: {np.mean(rmse_list):.4f}")
