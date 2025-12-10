import os, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ---- Paths
X_PATH = "data/X.npy"
META_PATH = "data/meta.csv"
CKPT_PATH = "models/mlp_embeddings.pt"
LABEL_MAP_JSON = "models/label_map.json"
OUT_CM = "data/confusion_matrix.png"
OUT_TXT = "data/classification_report.txt"

# ---- Model (same architecture as training)
class MLP(torch.nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, 512), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, k)
        )
    def forward(self, x): return self.net(x)

# ---- Load data
X = np.load(X_PATH).astype("float32")
meta = pd.read_csv(META_PATH)
y_true_labels = meta["label"].astype(str).values

# ---- Load model + label mapping
ckpt = torch.load(CKPT_PATH, map_location="cpu")
with open(LABEL_MAP_JSON, "r") as f:
    lm = json.load(f)
classes = lm["classes"]
inp_dim = ckpt["inp"]

model = MLP(inp_dim, len(classes))
model.load_state_dict(ckpt["state_dict"])
model.eval()

# ---- Predict
with torch.no_grad():
    logits = model(torch.from_numpy(X))
    probs = torch.softmax(logits, dim=1).numpy()
    y_pred_ids = probs.argmax(1)

id2label = {i:c for i, c in enumerate(classes)}
y_pred_labels = np.array([id2label[i] for i in y_pred_ids])

# ---- Metrics
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — WebShield-AI")
plt.savefig(OUT_CM, bbox_inches="tight")
plt.close()

report = classification_report(y_true_labels, y_pred_labels, target_names=classes, digits=3)
with open(OUT_TXT, "w") as f:
    f.write(report)

print("✅ Saved:", OUT_CM)
print("✅ Saved:", OUT_TXT)
print("\n", report)
