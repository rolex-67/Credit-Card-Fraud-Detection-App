# 🛡️ Real-Time AI Financial Fraud Detection System
### High-Performance Stacking Ensemble (XGBoost + LightGBM + Logistic Regression) with FastAPI & Streamlit

![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-EB6424?logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/ML-LightGBM-brightgreen)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9998-success)
![Recall](https://img.shields.io/badge/Fraud%20Recall-99.63%25-success)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 1. Project Overview

An end-to-end, production-ready **Financial Fraud Detection & Transaction Risk Scoring Platform**. 

Financial fraud accounts for billions of dollars in annual losses and exhibits extreme class imbalance ($<0.15\%$ fraud incidence). This project implements a **Stacking Ensemble Machine Learning Pipeline** combining **XGBoost**, **LightGBM / Histogram Gradient Boosting**, and a **Logistic Regression Meta-Learner** to detect unauthorized account draining, identity theft, and fraudulent transfers with **99.63% Fraud Recall** and **0.9998 ROC-AUC**.

---

## 🏗️ 2. System Architecture

```
  [ Client Application / Web Browser ]
                  │
                  ▼
   ┌───────────────────────────────┐
   │    Streamlit UI Dashboard     │  (Port 8501)
   │  - Live Transaction Inputs    │
   │  - Real-Time Risk Score Gauge │
   │  - Explainable AI Breakdown   │
   └──────────────┬────────────────┘
                  │
                  │ HTTP POST /predict (JSON)
                  ▼
   ┌───────────────────────────────┐
   │   FastAPI REST Microservice   │  (Port 8000)
   │  - Asynchronous Uvicorn ASGI  │
   │  - Pydantic Schema Validation │
   │  - In-Memory Model Scoring    │
   └──────────────┬────────────────┘
                  │
                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │             STACKING ENSEMBLE CLASSIFIER PIPELINE           │
   │                                                             │
   │   Base Estimator 1: XGBoost (Tree-based Gradient Boost)     │
   │   Base Estimator 2: LightGBM (Leaf-wise Hist GBDT)          │
   │                                                             │
   │   Meta-Learner: Logistic Regression (Probability Calibration)│
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
                [ Calibrated Fraud Risk Probability ]
```

---

## 📊 3. Model Performance (on 61,643 Test Samples)

| Evaluation Metric | Score | Significance |
| :--- | :--- | :--- |
| **ROC-AUC Score** | **0.99979** | Near-perfect discrimination between fraud & legitimate |
| **Fraud Recall (Class 1)** | **99.63%** | Caught **1,637 out of 1,643** fraudulent attempts (only 6 missed) |
| **Overall Accuracy** | **99.62%** | High-fidelity generalization without majority-class bias |
| **Fraud F1-Score** | **0.9330** | Optimal harmonic balance of precision and sensitivity |

---

## 🚀 4. Quick Start (Local Setup)

### Option A: 1-Click Launch (Windows)
Double-click [`run_app.bat`](file:///d:/Projects/Fraud%20Detection%20System/Credit-Card-Fraud-Detection/run_app.bat) to automatically install dependencies and launch both the backend and frontend.

### Option B: Terminal Setup (Windows, macOS & Linux)

```bash
# 1. Clone the repository
git clone https://github.com/rolex-67/Credit-Card-Fraud-Detection-App.git
cd Credit-Card-Fraud-Detection-App

# 2. Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the FastAPI Backend (Terminal 1)
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000

# 5. Start the Streamlit Frontend (Terminal 2)
python -m streamlit run streamlit_app.py
```

* **Web UI Dashboard:** [http://localhost:8501](http://localhost:8501)
* **Interactive API Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🌐 5. Cloud Deployment

- **Backend:** Deploy on **Render.com** as a Python Web Service (`uvicorn app:app --host 0.0.0.0 --port $PORT`).
- **Frontend:** Deploy on **Streamlit Community Cloud** linked to this GitHub repository.

---

## 📄 6. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
