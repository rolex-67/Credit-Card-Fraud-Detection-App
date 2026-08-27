import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

class FraudStackingEnsemble:
    """
    Stacked Ensemble Model for Credit Card Fraud Detection:
    - Base Model 1: XGBoost (Tree-based Gradient Boosting)
    - Base Model 2: LightGBM / Fast Hist Gradient Boosting (Leaf-wise Gradient Boosting)
    - Meta Model: Logistic Regression (Optimal Probability Stacker)
    """
    def __init__(self):
        self.xgb = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=4.0,
            tree_method='hist',
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
        self.hgb = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=6,
            learning_rate=0.08,
            class_weight='balanced',
            random_state=42
        )
        self.meta_learner = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )

    def fit(self, X, y):
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        y_arr = np.ascontiguousarray(y, dtype=np.int32)
        
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof_meta_features = np.zeros((len(X_arr), 2), dtype=np.float64)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
            X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
            X_va, y_va = X_arr[val_idx], y_arr[val_idx]
            
            fold_xgb = XGBClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=4.0,
                tree_method='hist',
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
            fold_xgb.fit(X_tr, y_tr)
            
            fold_hgb = HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=6,
                learning_rate=0.08,
                class_weight='balanced',
                random_state=42
            )
            fold_hgb.fit(X_tr, y_tr)
            
            oof_meta_features[val_idx, 0] = fold_xgb.predict_proba(X_va)[:, 1]
            oof_meta_features[val_idx, 1] = fold_hgb.predict_proba(X_va)[:, 1]

        self.meta_learner.fit(oof_meta_features, y_arr)
        self.xgb.fit(X_arr, y_arr)
        self.hgb.fit(X_arr, y_arr)
        return self

    def predict_proba(self, X):
        X_arr = np.ascontiguousarray(X, dtype=np.float64)
        p_xgb = self.xgb.predict_proba(X_arr)[:, 1]
        p_hgb = self.hgb.predict_proba(X_arr)[:, 1]
        meta_features = np.column_stack([p_xgb, p_hgb])
        return self.meta_learner.predict_proba(meta_features)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)
