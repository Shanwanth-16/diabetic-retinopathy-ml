"""
ui/app.py
Streamlit UI:
- Load model.joblib, features.csv as reference
- Upload an image or pick from processed/, run single-image preprocessing, extract features, predict class,
  show SHAP explanation (feature bar) and overlay masks (visual report).
"""

import streamlit as st
from pathlib import Path
import sys, os
import numpy as np
from PIL import Image
import joblib
import pandas as pd
import cv2
import json
# import our modules (ensure python path)
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from preprocessing import load_image, crop_retina, resize_and_pad, clahe_green, background_subtract, vessel_map, detect_dark_blobs, detect_bright_spots
from features import color_moments, compute_glcm_features, sharpness_score, entropy_score, lesion_stats, vessel_stats
from visualize import build_visual_report

st.set_page_config(page_title="DR (classical ML) — APTOS", layout="centered")

st.title("Diabetic Retinopathy (Classical ML) — Demo")
st.markdown("Multi-class (0–4) classifier using handcrafted features (vessel, lesions, texture).")

model_path = st.text_input("Model path (joblib)", "model.joblib")
mapping_csv = st.text_input("Mapping CSV (image,label)", "train.csv")
processed_dir = st.text_input("Processed dir (for masks)", "data/processed")
masks_dir = st.text_input("Masks dir", "data/masks")

if not Path(model_path).exists():
    st.warning("Model not found. Train first with src/model.py")
model = None
if Path(model_path).exists():
    model = joblib.load(model_path)
    st.success("Loaded model")

uploaded = st.file_uploader("Upload APTOS PNG image", type=['png','jpg','jpeg'])
if uploaded is not None and model is not None:
    # save temp and run single-image preprocess
    temp = Path("temp_upload.png")
    with open(temp, "wb") as f:
        f.write(uploaded.getbuffer())
    img = load_image(str(temp))
    crop, _ = crop_retina(img)
    if crop is None:
        st.error("Could not detect retina circle.")
    else:
        p = resize_and_pad(crop, size=512)
        p_clahe = clahe_green(p)
        green = p_clahe[...,1]
        green_bs = background_subtract(green, sigma=50)
        p_clahe[...,1] = green_bs
        # save to show
        st.image(Image.fromarray(p_clahe), caption="Preprocessed image", use_column_width=True)
        # masks
        fr, vessel_thr = vessel_map(p_clahe[...,1])
        _, dark_mask, kp = detect_dark_blobs(p_clahe[...,1])
        _, bright_mask = detect_bright_spots(p_clahe)
        # build features for this one image
        feats = {}
        feats.update(color_moments(p_clahe))
        feats['sharpness'] = sharpness_score(cv2.cvtColor(p_clahe, cv2.COLOR_RGB2GRAY))
        feats['entropy'] = entropy_score(cv2.cvtColor(p_clahe, cv2.COLOR_RGB2GRAY))
        feats.update(compute_glcm_features(cv2.cvtColor(p_clahe, cv2.COLOR_RGB2GRAY)))
        vstats = vessel_stats(vessel_thr)
        feats.update(vstats)
        dstats = lesion_stats(dark_mask)
        bstats = lesion_stats(bright_mask)
        for k,v in dstats.items(): feats[f'dark_{k}']=v
        for k,v in bstats.items(): feats[f'bright_{k}']=v
        X_single = pd.DataFrame([feats]).fillna(0)
        # ensure columns match training features - load features.csv and align
        if Path("features.csv").exists():
            ref = pd.read_csv("features.csv").drop(columns=['image'], errors='ignore')
            # add missing cols with 0, drop extras
            for c in ref.columns:
                if c not in X_single.columns: X_single[c]=0
            X_single = X_single[ref.columns]
        # predict
        pred = model.predict(X_single)[0]
        proba = model.predict_proba(X_single)[0] if hasattr(model, "predict_proba") else None
        st.write(f"Predicted class: **{int(pred)}**")
        if proba is not None:
            st.write("Probabilities:", [round(float(x),3) for x in proba.tolist()])
        # save masks and build visual report
        tmp_masks = Path("temp_masks")
        tmp_masks.mkdir(exist_ok=True)
        base="upload"
        cv2.imwrite(str(tmp_masks/f"{base}_vessel.png"), vessel_thr)
        cv2.imwrite(str(tmp_masks/f"{base}_dark.png"), dark_mask)
        cv2.imwrite(str(tmp_masks/f"{base}_bright.png"), bright_mask)
        out = Path("temp_report.png")
        build_visual_report(str(Path("temp_upload.png")), tmp_masks, str(out))
        st.image(str(out), caption="Visual explanation (vessels=green, dark lesions=red, exudates=yellow)")
        # SHAP explanation (if tree model)
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            # we need training data to compute base value; try to load features.csv
            if Path("features.csv").exists():
                Xref = pd.read_csv("features.csv").drop(columns=['image'], errors='ignore').fillna(0)
                # align columns
                for c in Xref.columns:
                    if c not in X_single.columns:
                        X_single[c]=0
                X_single = X_single[Xref.columns]
                shap_values = explainer.shap_values(Xref.sample(min(200, len(Xref)), random_state=1))
                # use summary on top features
                st.markdown("**Feature importance (approx via SHAP)**")
                shap.initjs()
                # show bar plot of absolute mean SHAP for each feature using our model and small sample
                abs_mean = np.abs(shap_values).mean(axis=1) if isinstance(shap_values, np.ndarray) else np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                # fallback simple bar - use pandas
                if isinstance(abs_mean, np.ndarray):
                    feat_names = Xref.columns
                    dfm = pd.Series(abs_mean, index=feat_names).sort_values(ascending=False)[:20]
                    st.bar_chart(dfm)
        except Exception as e:
            st.write("SHAP not available or failed:", e)
