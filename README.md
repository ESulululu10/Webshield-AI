# WebShield-AI  
### A Human-in-the-Loop Explainable Web Filtering System  
**CYB 501 – Final Project | University of Michigan–Flint**

WebShield-AI is an AI-powered website classification system that categorizes domains into:

- **Arts & Culture**
- **Entertainment**
- **News**

The system combines machine learning, explainability (SHAP), and a Streamlit interface that supports **human-in-the-loop feedback** to improve accuracy over time.

This project was completed individually as part of **CYB 501 – Foundations of Cybersecurity**.

---

## 🌐 Project Highlights

### ✔ End-to-End Machine Learning Pipeline

#### **1. Dataset Preparation**
- Manually labeled domains into 3 categories  
- Image + text embeddings (from UM-Flint research crawler)  
- Metadata stored in `meta.csv`

#### **2. Preprocessing**
- Load `.npy` embedding files  
- L2 normalize  
- Merge image + text vectors into unified embeddings  

#### **3. Model Training**
- PyTorch **Multi-Layer Perceptron (MLP)**  
- Trained on ~6.9K samples  
- Achieved **99% overall accuracy**

#### **4. Model Evaluation**
- Classification report + confusion matrix  
- Handles severe imbalance (only 49 News samples)

#### **5. Explainability (SHAP)**
- KernelExplainer applied to normalized embeddings  
- Global summary plot for feature contributions  

#### **6. Streamlit UI**
- Domain dropdown selector  
- Classification + confidence  
- SHAP explainability plot  
- Feedback buttons:  
  - ✅ Prediction Correct  
  - ❌ Prediction Wrong  
- Feedback saved to `feedback.csv`

---

## 📁 Repository Structure
```
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
```
---

⚠ **Raw dataset folders (Arts, Entertainment, News) are intentionally excluded using `.gitignore`.  
These folders contain large files and are not required to run the model.**

---

## 🧠 Model Performance

| Class            | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| Arts & Culture   | 1.000     | 1.000  | 1.000    | 3723    |
| Entertainment    | 0.999     | 0.999  | 0.999    | 2454    |
| News             | 0.959     | 0.959  | 0.959    | 49      |

- **Overall Accuracy:** 0.999  
- News class underperforms because of extremely small sample size  
- Model still shows strong generalization across categories  

---

## 🔍 Explainability with SHAP

- Uses **SHAP KernelExplainer**  
- Shows global feature importance  
- Helps analysts understand how embedding dimensions influence classification  
- Builds trust by making model decisions transparent  

---

## 🖥 Streamlit Web Application

### Features:
- Domain dropdown  
- One-click classification  
- Confidence + class probability distribution  
- SHAP explainability visualization  
- Human feedback (correct/incorrect)  
- Saves feedback to `feedback.csv` for retraining loops  
---
### Run the App

```bash
streamlit run src/app_streamlit.py
```
📦 Installation
```
1. Clone the repository
git clone https://github.com/<your-username>/WebShield-AI.git
cd WebShield-AI

2. Install dependencies
pip install -r requirements.txt
```
🚧 Limitations
-Severe dataset imbalance (only 49 News samples)
- Only 3 categories — real web filtering requires ~50+
- Model uses embeddings only (no HTML/JS structural analysis)
- SHAP used globally — no local force plots
- Manual labeling required (time-consuming)
- Streamlit UI is a demo, not deployed in production or SOC tools

🚀 Future Work
- Expand dataset (especially News)
- Add URL lexical feature extraction
- Integrate screenshot/image-based models (CLIP, ViT)
- Improve explainability with local SHAP
- Deploy as a real-time filtering service (Flask API / Proxy integration)
- Add adversarial robustness testing
- Add retraining loop using user feedback

🙏 Acknowledgements
- Special thanks to Dr. Khalid Malik and TA JJ Ryan for guidance, feedback, and the opportunity to work on this project.
