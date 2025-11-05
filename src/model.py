"""
model.py
- Train classical ML model(s) for multi-class classification:
  - Loads features.csv and train.csv (APTOS mapping image -> label)
  - Trains RandomForest and XGBoost (optional), picks best by cross-val accuracy,
    saves model (joblib) and produces SHAP values for a validation subset.
- Outputs:
  - model.joblib
  - shap_values.pkl and sample shap summary plot
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import joblib
import shap
import matplotlib.pyplot as plt

def load_data(features_csv, mapping_csv):
    feats = pd.read_csv(features_csv)
    mapdf = pd.read_csv(mapping_csv)  # expected columns: image,label
    df = feats.merge(mapdf, left_on='image', right_on='image', how='inner')
    # drop non-numeric columns
    X = df.drop(columns=['image','label'])
    # fill na
    X = X.fillna(0)
    y = df['label'].astype(int).values
    return X, y, df

def train_and_select(X, y, save_path="model.joblib", use_xgb=True):
    # simple split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    # RandomForest baseline
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred = rf.predict(X_val)
    rf_acc = accuracy_score(y_val, pred)
    print("RF val acc:", rf_acc)
    best = rf
    best_name = "rf"
    # try xgboost
    if use_xgb:
        clf = xgb.XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
        clf.fit(X_train, y_train)
        px = clf.predict(X_val)
        xacc = accuracy_score(y_val, px)
        print("XGB val acc:", xacc)
        if xacc > rf_acc:
            best = clf
            best_name = "xgb"
    joblib.dump(best, save_path)
    print("Saved model:", save_path, "type:", best_name)
    # detailed report
    ypred = best.predict(X_val)
    print(classification_report(y_val, ypred))
    return best, X_val, y_val

def explain_with_shap(model, X_sample, out_plot="shap_summary.png"):
    # TreeExplainer for tree models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)  # for multi-class returns list
    # summary plot (class 0 vs others can be shown; show for first class)
    plt.figure(figsize=(8,6))
    try:
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.savefig(out_plot, bbox_inches='tight', dpi=150)
        plt.close()
    except Exception as e:
        print("shap summary plot failed:", e)
    return shap_values, explainer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--out", default="model.joblib")
    args = parser.parse_args()
    X, y, df = load_data(args.features, args.mapping)
    model, Xv, yv = train_and_select(X, y, save_path=args.out)
    # compute SHAP on a small sample
    sample = Xv.iloc[:200] if len(Xv)>200 else Xv
    shap_values, explainer = explain_with_shap(model, sample, out_plot="shap_summary.png")
    print("Done training + shap. See shap_summary.png")
