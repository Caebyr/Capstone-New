"""
Driver for the xView-vs-VisDrone drift comparison (expanded vocabulary: car/bus/truck/van).

xView is streamed from the HichTala/xview HF mirror -- no local download, see
PROJECT_CONTEXT.md for why. This is a *preparation* pass: xView has no dedicated "van" category
("Pickup Truck" is the closest match and is rare), so van is measured but likely to be excluded
from the reported drift if it doesn't clear MIN_SAMPLES_FOR_DRIFT -- a bigger scan later can
revisit that specifically. car/bus/truck are well-populated and are the real payoff here: three
genuinely shared categories between xView and VisDrone, vs. DOTA's fixed ceiling of two
(small-vehicle/large-vehicle) that no amount of extra scanning can raise.

No fine-tuning is involved in this comparison (both encoders are off-the-shelf, frozen), so
there's no train/eval leakage concern and no need for a held-out split here.
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
from numpy.linalg import norm

sys.path.insert(0, os.path.dirname(__file__))
from src.dataset import EXPANDED_VOCAB, VISDRONE_TO_EXPANDED, get_visdrone_crops, get_xview_crops, subsample_balanced
from src.embeddings import encode_with_clip, encode_with_remoteclip, load_clip, load_remoteclip

MAX_XVIEW_IMAGES = 5000  # bounded stream scan; enough for car/bus/truck at 300/class, van excluded
MAX_CROPS_PER_CLASS = 300
MIN_SAMPLES_FOR_DRIFT = 30  # classes below this on either side are reported but excluded from drift
RANDOM_SEED = 42
BATCH_SIZE = 16

VISDRONE_BASE = "data/raw/VisDrone/VisDrone2019-DET-train"
VISDRONE_IMG_DIR = os.path.join(VISDRONE_BASE, "images")
VISDRONE_ANN_DIR = os.path.join(VISDRONE_BASE, "annotations")

EMBED_OUT = "embeddings/xview_vs_visdrone"
RESULTS_OUT = "results/xview_vs_visdrone"
os.makedirs(EMBED_OUT, exist_ok=True)
os.makedirs(RESULTS_OUT, exist_ok=True)


def log(msg, flush=True):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=flush)


def compute_drift(xview_feats, xview_labels, vis_feats, vis_labels):
    scores = {}
    for cls in EXPANDED_VOCAB:
        xv = xview_feats[xview_labels == cls]
        uav = vis_feats[vis_labels == cls]
        if len(xv) < MIN_SAMPLES_FOR_DRIFT or len(uav) < MIN_SAMPLES_FOR_DRIFT:
            log(f"  skipping {cls}: xview={len(xv)} visdrone={len(uav)} (below MIN_SAMPLES_FOR_DRIFT={MIN_SAMPLES_FOR_DRIFT})")
            continue
        xv_c, uav_c = xv.mean(axis=0), uav.mean(axis=0)
        l2 = float(norm(xv_c - uav_c))
        cos = float(1 - np.dot(xv_c, uav_c) / (norm(xv_c) * norm(uav_c)))
        xv_intra = float(np.mean(norm(xv - xv_c, axis=1)))
        uav_intra = float(np.mean(norm(uav - uav_c, axis=1)))
        scores[cls] = {
            "l2_drift": round(l2, 4),
            "cosine_drift": round(cos, 4),
            "drift_ratio": round(l2 / ((xv_intra + uav_intra) / 2), 4),
            "xview_samples": int(len(xv)),
            "visdrone_samples": int(len(uav)),
        }
    return scores


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")

    log(f"streaming xView (up to {MAX_XVIEW_IMAGES} images, no local download)...")
    t0 = time.time()
    xview_crops_all = get_xview_crops(max_images=MAX_XVIEW_IMAGES, log=log)
    log(f"xView raw crops: {len(xview_crops_all)}  ({time.time()-t0:.1f}s)  {dict(Counter(c for _, c in xview_crops_all))}")
    xview_crops = subsample_balanced(xview_crops_all, MAX_CROPS_PER_CLASS, RANDOM_SEED)
    log(f"xView subsampled: {len(xview_crops)}  {dict(Counter(c for _, c in xview_crops))}")

    log("extracting VisDrone ground-truth crops (full scan, expanded vocab)...")
    t0 = time.time()
    visdrone_crops_all = get_visdrone_crops(VISDRONE_IMG_DIR, VISDRONE_ANN_DIR, class_map=VISDRONE_TO_EXPANDED)
    log(f"VisDrone raw crops: {len(visdrone_crops_all)}  ({time.time()-t0:.1f}s)  {dict(Counter(c for _, c in visdrone_crops_all))}")
    visdrone_crops = subsample_balanced(visdrone_crops_all, MAX_CROPS_PER_CLASS, RANDOM_SEED)
    log(f"VisDrone subsampled: {len(visdrone_crops)}  {dict(Counter(c for _, c in visdrone_crops))}")

    log("loading RemoteCLIP ViT-L/14...")
    remoteclip_model, remoteclip_preprocess = load_remoteclip("models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt", device)
    log("loading CLIP ViT-B/32...")
    clip_model, clip_processor = load_clip(device)

    log("encoding xView with RemoteCLIP...")
    t0 = time.time()
    xview_rc_feats, xview_rc_labels = encode_with_remoteclip(xview_crops, remoteclip_model, remoteclip_preprocess, device, BATCH_SIZE, log=log)
    log(f"xView RemoteCLIP done: {xview_rc_feats.shape}  ({time.time()-t0:.1f}s)")

    log("encoding VisDrone with RemoteCLIP...")
    t0 = time.time()
    vis_rc_feats, vis_rc_labels = encode_with_remoteclip(visdrone_crops, remoteclip_model, remoteclip_preprocess, device, BATCH_SIZE, log=log)
    log(f"VisDrone RemoteCLIP done: {vis_rc_feats.shape}  ({time.time()-t0:.1f}s)")

    log("encoding xView with CLIP...")
    t0 = time.time()
    xview_clip_feats, xview_clip_labels = encode_with_clip(xview_crops, clip_model, clip_processor, device, BATCH_SIZE, log=log)
    log(f"xView CLIP done: {xview_clip_feats.shape}  ({time.time()-t0:.1f}s)")

    log("encoding VisDrone with CLIP...")
    t0 = time.time()
    vis_clip_feats, vis_clip_labels = encode_with_clip(visdrone_crops, clip_model, clip_processor, device, BATCH_SIZE, log=log)
    log(f"VisDrone CLIP done: {vis_clip_feats.shape}  ({time.time()-t0:.1f}s)")

    log("computing drift (RemoteCLIP, native 768-dim space)...")
    remoteclip_drift = compute_drift(xview_rc_feats, xview_rc_labels, vis_rc_feats, vis_rc_labels)
    for cls, s in remoteclip_drift.items():
        log(f"  {cls}: {s}")

    log("computing drift (CLIP ViT-B/32, native 512-dim space)...")
    clip_drift = compute_drift(xview_clip_feats, xview_clip_labels, vis_clip_feats, vis_clip_labels)
    for cls, s in clip_drift.items():
        log(f"  {cls}: {s}")

    np.save(os.path.join(EMBED_OUT, "xview_remoteclip_feats.npy"), xview_rc_feats)
    np.save(os.path.join(EMBED_OUT, "xview_remoteclip_labels.npy"), xview_rc_labels)
    np.save(os.path.join(EMBED_OUT, "visdrone_remoteclip_feats.npy"), vis_rc_feats)
    np.save(os.path.join(EMBED_OUT, "visdrone_remoteclip_labels.npy"), vis_rc_labels)
    np.save(os.path.join(EMBED_OUT, "xview_clip_feats.npy"), xview_clip_feats)
    np.save(os.path.join(EMBED_OUT, "xview_clip_labels.npy"), xview_clip_labels)
    np.save(os.path.join(EMBED_OUT, "visdrone_clip_feats.npy"), vis_clip_feats)
    np.save(os.path.join(EMBED_OUT, "visdrone_clip_labels.npy"), vis_clip_labels)

    with open(os.path.join(RESULTS_OUT, "drift_comparison.json"), "w") as f:
        json.dump({
            "clip_vit_b32": clip_drift,
            "remoteclip_vit_l14": remoteclip_drift,
            "max_xview_images": MAX_XVIEW_IMAGES,
            "max_crops_per_class": MAX_CROPS_PER_CLASS,
            "min_samples_for_drift": MIN_SAMPLES_FOR_DRIFT,
            "random_seed": RANDOM_SEED,
            "notes": "xView has no dedicated van category (Pickup Truck is the closest match "
                     "and is rare); van is measured but likely excluded here for insufficient "
                     "samples. car/bus/truck are the real result of this pass.",
        }, f, indent=2)

    log(f"DONE. Saved embeddings to {EMBED_OUT} and results to {RESULTS_OUT}")


if __name__ == "__main__":
    main()
