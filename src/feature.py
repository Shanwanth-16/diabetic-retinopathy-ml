"""
features.py
- Load processed images + masks and compute a hybrid feature vector per image:
  - Vessel features (density)
  - Lesion counts/areas (dark blobs, bright exudates)
  - Texture (GLCM Haralick)
  - Color moments, sharpness, entropy
- Saves features.csv with one row per image.
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.feature import greycomatrix, greycoprops
from skimage import exposure
from scipy import ndimage as ndi

def compute_glcm_features(gray):
    # convert to 8-bit
    g = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # quantize to reduce levels
    levels = 8
    gq = (g / (256//levels)).astype(np.uint8)
    # compute glcm
    distances = [1, 2, 4]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = greycomatrix(gq, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    feats = {}
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for p in props:
        vals = greycoprops(glcm, p)
        feats[f'glcm_{p}_mean'] = float(np.mean(vals))
        feats[f'glcm_{p}_std'] = float(np.std(vals))
    return feats

def color_moments(img):
    # mean, std, skewness per channel (RGB)
    feats = {}
    for i, ch in enumerate(['r','g','b']):
        c = img[...,i].astype(np.float32)
        feats[f'{ch}_mean'] = float(np.mean(c))
        feats[f'{ch}_std'] = float(np.std(c))
        # skewness
        m3 = np.mean((c - feats[f'{ch}_mean'])**3)
        sigma = feats[f'{ch}_std'] + 1e-9
        feats[f'{ch}_skew'] = float(m3 / (sigma**3))
    return feats

def sharpness_score(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def entropy_score(gray):
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()
    hist = hist / (hist.sum()+1e-9)
    ent = -np.sum([p*np.log2(p) for p in hist if p>0])
    return float(ent)

def lesion_stats(mask):
    labels, n = ndi.label(mask>0)
    areas = ndi.sum(mask>0, labels, index=range(1, n+1))
    if len(areas)==0:
        return {"lesion_count":0, "lesion_total_area":0, "lesion_max_area":0}
    areas = np.array(areas)
    return {"lesion_count": int(len(areas)), "lesion_total_area": float(np.sum(areas)), "lesion_max_area": float(np.max(areas))}

def vessel_stats(vessel_mask):
    vessel_frac = float(np.mean(vessel_mask>0))
    # skeleton length approx
    skel = cv2.ximgproc.thinning(vessel_mask) if hasattr(cv2.ximgproc, 'thinning') else vessel_mask
    vessel_pixels = int(np.sum(vessel_mask>0))
    return {"vessel_frac": vessel_frac, "vessel_pixels": vessel_pixels}

def build_features(processed_dir, masks_dir, out_csv="features.csv"):
    processed_dir = Path(processed_dir)
    masks_dir = Path(masks_dir)
    rows = []
    for p in sorted(processed_dir.glob("*.png")):
        base = p.stem
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # masks
        vpath = masks_dir/f"{base}_vessel.png"
        dpath = masks_dir/f"{base}_dark.png"
        bpath = masks_dir/f"{base}_bright.png"
        vessel_mask = cv2.imread(str(vpath), cv2.IMREAD_GRAYSCALE) if vpath.exists() else np.zeros_like(gray)
        dark_mask = cv2.imread(str(dpath), cv2.IMREAD_GRAYSCALE) if dpath.exists() else np.zeros_like(gray)
        bright_mask = cv2.imread(str(bpath), cv2.IMREAD_GRAYSCALE) if bpath.exists() else np.zeros_like(gray)
        feats = {}
        feats['image'] = base + ".png"
        # color moments
        feats.update(color_moments(img))
        # sharpness, entropy
        feats['sharpness'] = sharpness_score(gray)
        feats['entropy'] = entropy_score(gray)
        # glcm
        feats.update(compute_glcm_features(gray))
        # vessel and lesion stats
        feats.update(vessel_stats(vessel_mask))
        dark_stats = lesion_stats(dark_mask)
        bright_stats = lesion_stats(bright_mask)
        # prefix
        for k,v in dark_stats.items():
            feats[f'dark_{k}'] = v
        for k,v in bright_stats.items():
            feats[f'bright_{k}'] = v
        rows.append(feats)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("Saved features:", out_csv)
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--out", default="features.csv")
    args = parser.parse_args()
    build_features(args.processed, args.masks, args.out)
