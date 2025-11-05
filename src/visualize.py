"""
visualize.py
- Given processed image + masks + model + SHAP info, create overlay visuals:
  - overlay vessels (green), dark lesions (red), bright lesions (yellow)
  - annotate top-k lesions relevant according to SHAP-feature mapping (simple heuristic)
- Provides a function `visualize_prediction(image_path, masks_dir, features_row, shap_values_row, out_path)`
"""

import cv2
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

def overlay_masks(image_rgb, vessel_mask, dark_mask, bright_mask, alpha=0.5):
    canvas = image_rgb.copy().astype(np.float32)/255.0
    overlay = canvas.copy()
    # green vessels
    overlay[vessel_mask>0] = [0.0, 1.0, 0.0]
    # dark lesions -> red
    overlay[dark_mask>0] = [1.0, 0.0, 0.0]
    # bright -> yellow
    overlay[bright_mask>0] = [1.0, 1.0, 0.0]
    out = (canvas*(1-alpha) + overlay*alpha)
    out = np.clip(out*255, 0, 255).astype(np.uint8)
    return out

def annotate_top_lesions(img, dark_mask, bright_mask, top_n=5):
    out = img.copy()
    # find contours of dark_mask and bright_mask
    def annotate(mask, color):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)[:top_n]
        i = 1
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(out, (x,y), (x+w, y+h), color, 2)
            cv2.putText(out, str(i), (x, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            i += 1
    annotate(dark_mask, (255,0,0))
    annotate(bright_mask, (0,255,255))
    return out

def build_visual_report(processed_img_path, masks_dir, out_path):
    base = Path(processed_img_path).stem
    img = cv2.imread(processed_img_path)[..., ::-1]
    vessels = cv2.imread(str(Path(masks_dir)/f"{base}_vessel.png"), cv2.IMREAD_GRAYSCALE) if Path(masks_dir/f"{base}_vessel.png").exists() else np.zeros(img.shape[:2], dtype=np.uint8)
    dark = cv2.imread(str(Path(masks_dir)/f"{base}_dark.png"), cv2.IMREAD_GRAYSCALE) if Path(masks_dir/f"{base}_dark.png").exists() else np.zeros(img.shape[:2], dtype=np.uint8)
    bright = cv2.imread(str(Path(masks_dir)/f"{base}_bright.png"), cv2.IMREAD_GRAYSCALE) if Path(masks_dir/f"{base}_bright.png").exists() else np.zeros(img.shape[:2], dtype=np.uint8)
    overlay = overlay_masks(img, vessels, dark, bright, alpha=0.55)
    annotated = annotate_top_lesions(overlay, dark, bright, top_n=5)
    cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    return out_path

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--out", default="report.png")
    args = parser.parse_args()
    build_visual_report(args.img, args.masks, args.out)
    print("Saved visual report:", args.out)
