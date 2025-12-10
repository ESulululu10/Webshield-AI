# WebShield-AI  
### A Human-in-the-Loop Explainable Web Filtering System (CYB 501 – Final Project)

WebShield-AI is an AI-powered website classification system that categorizes domains into:

- **Arts & Culture**
- **Entertainment**
- **News**

The system is built end-to-end using machine learning, explainability tools (SHAP), and a fully functional Streamlit user interface that supports **human feedback** to improve the model over time.

This project was completed individually as part of **CYB 501 – Foundations of Cybersecurity** at the **University of Michigan–Flint**.

---

## 🌐 Project Highlights

### ✔ End-to-End Pipeline
This project includes all stages of an ML workflow:

1. **Dataset Preparation**
   - Manually labeled domains across 3 categories  
   - Image + text embeddings provided by UM-Flint research crawler  
   - Metadata stored in `meta.csv`

2. **Preprocessing**
   - Load `.npy` embeddings  
   - L2 normalization  
   - Combine image + text vectors into unified embeddings  

3. **Model Training**
   - PyTorch **MLP classifier**
   - Handles ~6.9K labeled samples  
   - Achieves **99% overall accuracy**

4. **Evaluation**
   - Classification report & confusion matrix  
   - Handles significant dataset imbalance (only 49 news samples)

5. **Explainability (SHAP)**
   - KernelExplainer on normalized embeddings  
   - Global SHAP plot to understand feature influence  

6. **Streamlit App**
   - Select domain from dropdown  
   - Run classification + confidence  
   - View SHAP explainability plots  
   - Provide **human-in-the-loop feedback** saved in `feedback.csv`

---

## 📁 Repository Structure
WebShield-AI/
│ README.md
│ requirements.txt
│ .gitignore
│
├── src/
│ ├── train_mlp.py
│ ├── evaluate_model.py
│ ├── build_dataset_multiclass.py
│ └── app_streamlit.py
│
├── models/
│ ├── mlp_embeddings.pt
│ ├── rf_explainer.pkl
│ └── label_map.json
│
├── explainability/
│ ├── explain_shap.py
│ └── shap_summary.png
│
└── data/
├── classification_report.txt
├── confusion_matrix.png
└── meta.csv


⚠ **Raw dataset folders (Arts, Entertainment, News) are intentionally excluded** to keep the repository lightweight and avoid uploading large files.

---

## 🧠 Model Performance

| Class            | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| Arts & Culture   | 1.000     | 1.000  | 1.000    | 3723    |
| Entertainment    | 0.999     | 0.999  | 0.999    | 2454    |
| News             | 0.959     | 0.959  | 0.959    | 49      |

- **Overall Accuracy:** 0.999  
- News class performs slightly lower due to dataset imbalance  
- Still produces strong generalization across all categories  

---

## 🔍 Explainability with SHAP

- Uses **KernelExplainer** on normalized embeddings  
- Generates a global summary plot (`shap_summary.png`)  
- Helps understand how embedding dimensions influence predictions  
- Enables transparency for cybersecurity analysts  

---

## 🖥 Streamlit Web App

### Features:
- Domain dropdown  
- One-click classification  
- Probability distribution table  
- Confidence score  
- SHAP visualization  
- Human validation buttons:  
  - ✅ Prediction correct  
  - ❌ Prediction wrong  
- Feedback saved to `feedback.csv` for future retraining  

To run:

```bash
streamlit run src/app_streamlit.py

📦 Installation

Clone the repository:
git clone https://github.com/<username>/WebShield-AI.git
cd WebShield-AI

Install dependencies:
pip install -r requirements.txt

🚧 Limitations

Dataset imbalance (only 49 news samples)

Only 3 categories (real web filtering needs 50+ categories)

Embedding-only approach (no HTML, JS, layout features)

SHAP is global-only (no per-domain explanation)

Manual labeling required (time-consuming)

Streamlit UI is a demo, not integrated with proxy/SOC tools

🚀 Future Work

Expand dataset coverage

Add URL-level lexical features

Integrate screenshot-based models (ViT, CLIP)

Improve explainability with local SHAP

Deploy as real-time filtering API

Implement adversarial robustness testing

🙏 Acknowledgements

Special thanks to Dr. Khalid Malik and TA JJ Ryan for their guidance, support, and the opportunity to work on a research-aligned project for CYB-501.
