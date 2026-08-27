import os
import sys
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from model_def import FraudStackingEnsemble

def train_and_save():
    csv_file = "PS_20174392719_1491204439457_log.csv"
    if not os.path.exists(csv_file):
        for f in os.listdir("."):
            if f.endswith(".csv") and ("PS_" in f or "log" in f):
                csv_file = f
                break
                
    print(f"Loading data from: {csv_file} ...", flush=True)
    use_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud', 'isFraud']
    df = pd.read_csv(csv_file, usecols=use_cols)
    print(f"Total Records: {len(df):,} | Actual Frauds: {df['isFraud'].sum():,}", flush=True)

    type_map = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
    df['type'] = df['type'].map(type_map).fillna(0).astype(int)

    feature_cols = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
    
    # Stratified balance: Include ALL 8,213 frauds + 300,000 legitimate transactions
    fraud_df = df[df['isFraud'] == 1]
    non_fraud_df = df[df['isFraud'] == 0].sample(n=min(300000, len(df[df['isFraud'] == 0])), random_state=42)
    sample_df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1.0, random_state=42)

    X = sample_df[feature_cols].values
    y = sample_df['isFraud'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train):,} samples | Testing on {len(X_test):,} samples...", flush=True)
    
    model = FraudStackingEnsemble()
    model.fit(X_train, y_train)

    print("\n=================== MODEL EVALUATION ===================", flush=True)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test, threshold=0.5)

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.5f}", flush=True)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4), flush=True)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred), flush=True)
    print("========================================================\n", flush=True)

    # Save to credit_fraud.pkl
    out_paths = ["credit_fraud.pkl"]
    if os.path.exists("backend"):
        out_paths.append(os.path.join("backend", "credit_fraud.pkl"))
    for p in out_paths:
        joblib.dump(model, p, compress=3)
        print(f"Saved new stacked model to: {p}", flush=True)


    # Verification on sample fraud case
    test_sample = np.array([[1, 4, 181.0, 181.0, 0.0, 0.0, 0.0, 0]], dtype=np.float64)
    sample_prob = model.predict_proba(test_sample)[0][1]
    sample_pred = model.predict(test_sample)[0]
    print(f"\nVerification on Test Vector [Transfer, Empty Balance $181]:")
    print(f"  Predicted Fraud Probability: {sample_prob*100:.2f}% | Binary Classification: {sample_pred} ({'FRAUD' if sample_pred == 1 else 'NOT FRAUD'})", flush=True)
    print("\nTraining and replacement completed successfully!", flush=True)

if __name__ == "__main__":
    train_and_save()
