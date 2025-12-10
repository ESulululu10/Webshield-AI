# src/app_streamlit.py

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import streamlit as st

# ==========================
# Paths
# ==========================
X_PATH = "data/X.npy"
META_PATH = "data/meta.csv"
CKPT_PATH = "models/mlp_embeddings.pt"
LABEL_MAP_JSON = "models/label_map.json"

SHAP_IMG = "explainability/shap_summary.png"
REPORT_PATH = "data/classification_report.txt"
FEEDBACK_CSV = "data/feedback.csv"


# ==========================
# Model definition (must match training)
# ==========================
class MLP(torch.nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, k),
        )

    def forward(self, x):
        return self.net(x)


def pretty_label(raw: str) -> str:
    """Make label names nice for humans."""
    return (
        raw.replace("_&_", " & ")
        .replace("_", " ")
        .title()
    )


# ==========================
# Cached loaders
# ==========================
@st.cache_resource
def load_model_and_data():
    X = np.load(X_PATH).astype("float32")
    meta = pd.read_csv(META_PATH)

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    with open(LABEL_MAP_JSON, "r") as f:
        lm = json.load(f)
    classes = lm["classes"]

    model = MLP(ckpt["inp"], len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return X, meta, model, classes


@st.cache_data
def load_classification_report():
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r") as f:
            return f.read()
    return None


# ==========================
# Inference helper
# ==========================
def predict_vec(model, vec: np.ndarray):
    with torch.no_grad():
        logits = model(torch.from_numpy(vec[None, :]))
        probs = torch.softmax(logits, dim=1).numpy().ravel()
    top = int(probs.argmax())
    return top, probs


# ==========================
# Streamlit UI
# ==========================
st.set_page_config(
    page_title="WebShield-AI",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ WebShield-AI: Web Content Filtering Dashboard")
st.caption("Categories: Arts & Culture • Entertainment • News")

X, meta, model, classes = load_model_and_data()

# ------- Layout: two columns (left = interaction, right = explainability/metrics)
left, right = st.columns([1.1, 1])

# ----------------- LEFT COLUMN: Prediction + Feedback -----------------
with left:
    st.subheader("1. Choose a domain")
    domain = st.selectbox(
        "Domain (type to search):",
        options=sorted(meta["domain"].tolist()),
        index=0,
    )

    # Find row/index for selected domain
    row = meta[meta["domain"] == domain].iloc[0]
    idx = int(row.name)

    st.write("")  # small spacing

    classify_clicked = st.button("🚀 Classify this domain", use_container_width=True)

    if classify_clicked:
        label_id, probs = predict_vec(model, X[idx])
        pred_raw = classes[label_id]
        pred_nice = pretty_label(pred_raw)
        conf = float(probs[label_id])

        # Prediction banner
        st.success(
            f"**Prediction:** {pred_nice}  •  **Confidence:** {conf:.3f}",
            icon="✅",
        )

        st.markdown(f"**Domain:** `{domain}`")

        # Probabilities table
        st.markdown("**Class probabilities:**")
        prob_rows = []
        for i, c in enumerate(classes):
            prob_rows.append(
                {
                    "Class": pretty_label(c),
                    "Probability": f"{probs[i]:.3f}",
                }
            )
        st.table(pd.DataFrame(prob_rows))

        st.markdown("---")
        st.subheader("2. Human-in-the-loop feedback")

        col_yes, col_no = st.columns(2)

        def append_feedback(correct_flag: int):
            os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)
            fb = {
                "timestamp": datetime.utcnow().isoformat(),
                "domain": domain,
                "pred_raw": pred_raw,
                "pred_display": pred_nice,
                "correct": int(correct_flag),
            }
            df = pd.DataFrame([fb])
            df.to_csv(
                FEEDBACK_CSV,
                mode="a",
                header=not os.path.exists(FEEDBACK_CSV),
                index=False,
            )

        with col_yes:
            if st.button("✅ Yes, prediction is correct", use_container_width=True):
                append_feedback(1)
                st.toast("Thanks! Feedback recorded. ✅")

        with col_no:
            if st.button("❌ No, prediction is wrong", use_container_width=True):
                append_feedback(0)
                st.toast("Got it! Marked as incorrect. ❌")


# ----------------- RIGHT COLUMN: Explainability + Metrics -----------------
with right:
    st.subheader("3. Model explanation & validation")

    # SHAP explanation
    if os.path.exists(SHAP_IMG):
        with st.expander("🔍 View SHAP global explanation (feature importance)", expanded=False):
            st.image(
                SHAP_IMG,
                use_column_width=True,
                caption="SHAP summary plot over embedding features",
            )
    else:
        st.info("SHAP explanation image not found yet.", icon="ℹ️")

    # Validation metrics
    report_text = load_classification_report()
    if report_text:
        with st.expander("📊 View validation metrics (classification report)", expanded=False):
            st.text(report_text)
    else:
        st.info("Classification report file not found.", icon="ℹ️")

    st.markdown(
        "<small>Note: This demo runs on the labeled dataset you created in the "
        "NetSTAR/Cloudflare-based pipeline.</small>",
        unsafe_allow_html=True,
    )
