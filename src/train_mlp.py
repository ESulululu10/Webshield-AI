import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import os, json

# Load data
X = np.load("data/X.npy").astype("float32")
meta = pd.read_csv("data/meta.csv")

labels = meta["label"].astype(str).tolist()
classes = sorted(list(set(labels)))
label2id = {c:i for i,c in enumerate(classes)}
y = np.array([label2id[l] for l in labels], dtype="int64")

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl   = DataLoader(val_ds, batch_size=128, shuffle=False)

inp = X.shape[1]
num_classes = len(classes)

# MLP model
class MLP(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, k)
        )
    def forward(self, x): return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(inp, num_classes).to(device)

# Weighted loss for imbalance
counts = Counter(y_train.tolist())
weights = torch.tensor([len(y_train)/counts[i] for i in range(num_classes)], dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
opt = optim.Adam(model.parameters(), lr=1e-3)
best_acc, patience, bad = 0.0, 5, 0

def accuracy(dl):
    model.eval(); correct=total=0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred==yb).sum().item()
            total += yb.size(0)
    return correct/total

# Training loop
for epoch in range(30):
    model.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        opt.step()

    acc_tr = accuracy(train_dl)
    acc_va = accuracy(val_dl)
    print(f"epoch {epoch+1:02d} | train {acc_tr:.4f} | val {acc_va:.4f}")

    if acc_va > best_acc:
        best_acc, bad = acc_va, 0
        os.makedirs("models", exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "inp": inp, "classes": classes}, "models/mlp_embeddings.pt")
        with open("models/label_map.json","w") as f: json.dump({"classes": classes, "label2id": label2id}, f)
        print("*** Saved Best Model ***")
    else:
        bad += 1
        if bad >= patience:
            print("Early stopping")
            break

print("Best validation accuracy:", best_acc)

