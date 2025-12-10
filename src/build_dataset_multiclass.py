import os, glob, numpy as np, pandas as pd

# Base project directory (this script assumes you run it from the Project root)
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE, ".."))

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

ROOTS = {
    "entertainment": os.path.join(DATA_ROOT, "Entertainment_Venues_Activities"),
    "news": os.path.join(DATA_ROOT, "News"),
    "arts_&_culture": os.path.join(DATA_ROOT, "Arts_&_Culture_Events"),  # <- NEW
}


OUT_X = os.path.join(DATA_ROOT, "X.npy")
OUT_META = os.path.join(DATA_ROOT, "meta.csv")

os.makedirs(DATA_ROOT, exist_ok=True)

rows, vecs = [], []

def load_vec(domain_dir: str):
    """Load and concatenate available embeddings; L2-normalize."""
    paths = [
        os.path.join(domain_dir, "site.img.npy"),
        os.path.join(domain_dir, "site.text.npy"),
    ]
    parts = []
    for p in paths:
        if os.path.exists(p):
            try:
                parts.append(np.load(p).ravel())
            except Exception as e:
                print(f"[skip] {p}: {e}")
    if not parts:
        return None
    v = np.concatenate(parts)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v.astype("float32")

for label, root in ROOTS.items():
    if not os.path.isdir(root):
        print(f"[warn] Missing folder: {root}")
        continue
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        v = load_vec(d)
        if v is None:
            continue
        vecs.append(v)
        rows.append({
            "domain": os.path.basename(d),
            "path": d,
            "label": label,
            "dim": int(v.shape[0]),
        })

if not vecs:
    raise SystemExit("No embeddings found. Check your data folders and file names.")

X = np.vstack(vecs)
meta = pd.DataFrame(rows)

np.save(OUT_X, X)
meta.to_csv(OUT_META, index=False)

print("Saved embeddings:", X.shape, "->", OUT_X)
print("Saved meta:", OUT_META)
print("\nLabel counts:\n", meta["label"].value_counts())
