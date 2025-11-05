"""
preprocessing.py
- Batch preprocess APTOS PNG images: load, RGBA->RGB, detect/crop retina circle, resize, CLAHE on green channel,
  background subtraction, denoise, vessel enhancement (Frangi), detect bad images.
- Outputs:
  - processed images (512x512) as PNGs in processed_dir/
  - vessel masks and lesion candidate masks saved alongside
  - bad_images.csv listing images skipped
"""

import os
import cv2
import numpy as np
from skimage.filters import frangi
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import pandas as pd

def ensure_folder(p): 
    Path(p).mkdir(parents=True, exist_ok=True)

def load_image(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot load: {path}")
    # RGBA -> RGB
    if img.shape[-1] == 4:
        alpha = img[...,3] / 255.0
        rgb = np.empty(img[..., :3].shape, dtype=np.uint8)
        for c in range(3):
            rgb[..., c] = (img[..., c] * alpha + (1-alpha)*0)  # composite over black
        img = rgb
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img[..., ::-1]  # BGR->RGB

def crop_retina(img_rgb, debug=False):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    # blur then Otsu
    blur = cv2.medianBlur(gray, 5)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # find largest connected component
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    cnt = max(contours, key=cv2.contourArea)
    (x,y), radius = cv2.minEnclosingCircle(cnt)
    x, y, radius = int(x), int(y), int(radius)
    # create square crop around circle
    r = int(radius*1.05)
    h, w = img_rgb.shape[:2]
    x1, x2 = max(0, x-r), min(w, x+r)
    y1, y2 = max(0, y-r), min(h, y+r)
    crop = img_rgb[y1:y2, x1:x2].copy()
    mask = np.zeros_like(gray)
    cv2.circle(mask, (x-x1,y-y1), int(radius), 255, -1)
    return crop, mask[y1:y2, x1:x2]

def resize_and_pad(img, size=512):
    h, w = img.shape[:2]
    # resize keeping aspect then pad
    scale = size / max(h,w)
    nh, nw = int(h*scale), int(w*scale)
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    pad_top = (size - nh)//2
    pad_bottom = size - nh - pad_top
    pad_left = (size - nw)//2
    pad_right = size - nw - pad_left
    img_p = cv2.copyMakeBorder(img_r, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
    return img_p

def clahe_green(img):
    g = img[...,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    g2 = clahe.apply(g)
    out = img.copy()
    out[...,1] = g2
    return out

def background_subtract(img_gray, sigma=50):
    # large gaussian blur as background estimate
    bg = cv2.GaussianBlur(img_gray, (0,0), sigma)
    res = cv2.subtract(img_gray, bg)
    # normalize
    res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
    return res

def vessel_map(g_channel):
    # Frangi expects float in [0,1]
    imf = g_channel.astype(np.float32) / 255.0
    # Frangi tends to highlight vessels as bright lines
    fr = frangi(imf, scale_range=(1,6), scale_step=2)
    # normalize to 0-255
    fr = (255 * (fr - np.min(fr)) / (np.ptp(fr)+1e-9)).astype(np.uint8)
    _, thr = cv2.threshold(fr, 30, 255, cv2.THRESH_BINARY)  # threshold adjustable
    return fr, thr

def detect_dark_blobs(img_gray):
    # black top-hat to emphasize dark spots (microaneurysms/hemorrhages)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    tophat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
    # blob detector params
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 5
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 2000
    params.filterByCircularity = False
    params.filterByInertia = False
    params.filterByColor = True
    params.blobColor = 255
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(tophat)
    mask = np.zeros_like(img_gray)
    for kp in keypoints:
        cv2.circle(mask, (int(kp.pt[0]), int(kp.pt[1])), int(max(2,int(kp.size/2))), 255, -1)
    return tophat, mask, keypoints

def detect_bright_spots(img_rgb):
    # exudates often bright/yellow: work in LAB space
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L = lab[...,0]
    # top-hat transform to capture bright spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    tophat = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, kernel)
    _, thr = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)  # adjustable
    return tophat, thr

def is_bad_image(img_rgb):
    # detect near-black or extremely low variance
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mean = float(np.mean(gray))/255.0
    std = float(np.std(gray))/255.0
    if mean < 0.02 or std < 0.02:
        return True
    return False

def preprocess_folder(src_dir, processed_dir, masks_dir, size=512, save_masks=True):
    ensure_folder(processed_dir)
    ensure_folder(masks_dir)
    bad = []
    files = sorted(Path(src_dir).glob("*.png"))
    for f in tqdm(files, desc="Preprocessing"):
        try:
            img = load_image(str(f))
        except Exception as e:
            bad.append((str(f), "load_error"))
            continue
        if is_bad_image(img):
            bad.append((str(f), "dark_or_low_variance"))
            continue
        crop, mask = crop_retina(img)
        if crop is None or crop.size == 0:
            bad.append((str(f), "no_retina"))
            continue
        img_r = resize_and_pad(crop, size=size)
        img_clahe = clahe_green(img_r)
        # background subtract on green channel
        green = img_clahe[...,1]
        green_bs = background_subtract(green, sigma=max(30, size//10))
        # denoise
        green_bs = cv2.medianBlur(green_bs, 3)
        # reconstruct a 3-channel preprocessed image (RGB but green enhanced)
        proc = img_r.copy()
        proc[...,1] = green_bs
        # vessel map
        fr, vessel_thr = vessel_map(proc[...,1])
        # lesions
        dark_tophat, dark_mask, kp = detect_dark_blobs(proc[...,1])
        bright_tophat, bright_mask = detect_bright_spots(proc)
        # save outputs
        base = Path(f).stem
        cv2.imwrite(str(Path(processed_dir)/f"{base}.png"), cv2.cvtColor(proc, cv2.COLOR_RGB2BGR))
        if save_masks:
            cv2.imwrite(str(Path(masks_dir)/f"{base}_vessel.png"), vessel_thr)
            cv2.imwrite(str(Path(masks_dir)/f"{base}_dark.png"), dark_mask)
            cv2.imwrite(str(Path(masks_dir)/f"{base}_bright.png"), bright_mask)
        # optionally save metadata per-image (skip for speed)
    # write bad images csv
    if bad:
        df = pd.DataFrame(bad, columns=["path","reason"])
        df.to_csv(Path(processed_dir)/"bad_images.csv", index=False)
    print("Done preprocess. processed:", len(list(Path(processed_dir).glob("*.png"))))
