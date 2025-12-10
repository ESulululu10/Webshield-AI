import os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shap, joblib, matplotlib.pyplot as plt

os.makedirs("explainability", exist_ok=True)

X = np.load("data/X.npy").astype("float32")
y = pd.read_csv("data/meta.csv")["label"].astype(str).values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr)

print(classification_report(y_te, rf.predict(X_te), digits=3))
joblib.dump(rf, "models/rf_explainer.pkl")

explainer = shap.TreeExplainer(rf)
idx = np.random.choice(len(X_te), size=min(400, len(X_te)), replace=False)
shap_vals = explainer.shap_values(X_te[idx])

plt.figure()
shap.summary_plot(shap_vals, X_te[idx], show=False)
plt.savefig("explainability/shap_summary.png", bbox_inches="tight")
plt.close()
print("✅ Saved: explainability/shap_summary.png")
