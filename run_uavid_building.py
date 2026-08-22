"""
Adds "building" as a 4th shared class to the xView-vs-VisDrone comparison, using UAVid as the
UAV-side source for buildings (VisDrone has no building annotations at all -- it's a pure
traffic/pedestrian dataset). xView's "Building" category is the satellite-side source.

UAVid crops come from local files (data/raw/UAVid, downloaded from Kaggle -- the official DLR/UTwente
direct-download links were unreachable from this environment). xView is streamed (no download),
same as run_xview_vs_visdrone.py.

Merges the result with the existing car/bus/truck drift_comparison.json from
run_xview_vs_visdrone.py into a single updated 4-class comparison -- does not recompute
car/bus/truck (no need, nothing about that measurement changed).
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
from src.dataset import get_uavid_crops, get_xview_crops, subsample_balanced
from src.embeddings import encode_with_clip, encode_with_remoteclip, load_clip, load_remoteclip

MAX_XVIEW_IMAGES_FOR_BUILDING = 500  # buildings are common in xView; should be plenty
MAX_CROPS_PER_CLASS = 300
MIN_SAMPLES_FOR_DRIFT = 30
RANDOM_SEED = 42
BATCH_SIZE = 16

UAVID_BASE = "data/raw/UAVid/modified_uavid_dataset"

EMBED_OUT = "embeddings/xview_vs_visdrone"
RESULTS_OUT = "results/xview_vs_visdrone"
os.makedirs(EMBED_OUT, exist_ok=True)
os.makedirs(RESULTS_OUT, exist_ok=True)


def log(msg, flush=True):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=flush)


def compute_drift(xview_feats, uav_feats):
    xv_c, uav_c = xview_feats.mean(axis=0), uav_feats.mean(axis=0)
    l2 = float(norm(xv_c - uav_c))
    cos = float(1 - np.dot(xv_c, uav_c) / (norm(xv_c) * norm(uav_c)))
    xv_intra = float(np.mean(norm(xview_feats - xv_c, axis=1)))
    uav_intra = float(np.mean(norm(uav_feats - uav_c, axis=1)))
    return {
        "l2_drift": round(l2, 4),
        "cosine_drift": round(cos, 4),
        "drift_ratio": round(l2 / ((xv_intra + uav_intra) / 2), 4),
        "xview_samples": int(len(xview_feats)),
        "visdrone_samples": int(len(uav_feats)),  # keeping key name consistent with the other file (uav side)
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")

    log("collecting UAVid building crops (train + val, local files)...")
    t0 = time.time()
    uavid_crops = get_uavid_crops(
        os.path.join(UAVID_BASE, "train_data", "Images"), os.path.join(UAVID_BASE, "train_data", "Labels"), log=log,
    )
    uavid_crops += get_uavid_crops(
        os.path.join(UAVID_BASE, "val_data", "Images"), os.path.join(UAVID_BASE, "val_data", "Labels"), log=log,
    )
    log(f"UAVid raw building crops: {len(uavid_crops)}  ({time.time()-t0:.1f}s)")
    uavid_crops = subsample_balanced(uavid_crops, MAX_CROPS_PER_CLASS, RANDOM_SEED)
    log(f"UAVid subsampled: {len(uavid_crops)}")

    log(f"streaming xView for buildings (up to {MAX_XVIEW_IMAGES_FOR_BUILDING} images)...")
    t0 = time.time()
    xview_crops = get_xview_crops(class_map={"Building": "building"}, max_images=MAX_XVIEW_IMAGES_FOR_BUILDING, log=log)
    log(f"xView raw building crops: {len(xview_crops)}  ({time.time()-t0:.1f}s)")
    xview_crops = subsample_balanced(xview_crops, MAX_CROPS_PER_CLASS, RANDOM_SEED)
    log(f"xView subsampled: {len(xview_crops)}")

    if len(uavid_crops) < MIN_SAMPLES_FOR_DRIFT or len(xview_crops) < MIN_SAMPLES_FOR_DRIFT:
        log(f"INSUFFICIENT SAMPLES: xview={len(xview_crops)} uavid={len(uavid_crops)} (need >= {MIN_SAMPLES_FOR_DRIFT}). Stopping.")
        return

    log("loading RemoteCLIP ViT-L/14...")
    remoteclip_model, remoteclip_preprocess = load_remoteclip("models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt", device)
    log("loading CLIP ViT-B/32...")
    clip_model, clip_processor = load_clip(device)

    log("encoding xView buildings with RemoteCLIP...")
    xview_rc_feats, _ = encode_with_remoteclip(xview_crops, remoteclip_model, remoteclip_preprocess, device, BATCH_SIZE, log=log)
    log("encoding UAVid buildings with RemoteCLIP...")
    uavid_rc_feats, _ = encode_with_remoteclip(uavid_crops, remoteclip_model, remoteclip_preprocess, device, BATCH_SIZE, log=log)

    log("encoding xView buildings with CLIP...")
    xview_clip_feats, _ = encode_with_clip(xview_crops, clip_model, clip_processor, device, BATCH_SIZE, log=log)
    log("encoding UAVid buildings with CLIP...")
    uavid_clip_feats, _ = encode_with_clip(uavid_crops, clip_model, clip_processor, device, BATCH_SIZE, log=log)

    remoteclip_building_drift = compute_drift(xview_rc_feats, uavid_rc_feats)
    clip_building_drift = compute_drift(xview_clip_feats, uavid_clip_feats)
    log(f"RemoteCLIP building drift: {remoteclip_building_drift}")
    log(f"CLIP building drift: {clip_building_drift}")

    np.save(os.path.join(EMBED_OUT, "xview_building_remoteclip_feats.npy"), xview_rc_feats)
    np.save(os.path.join(EMBED_OUT, "uavid_building_remoteclip_feats.npy"), uavid_rc_feats)
    np.save(os.path.join(EMBED_OUT, "xview_building_clip_feats.npy"), xview_clip_feats)
    np.save(os.path.join(EMBED_OUT, "uavid_building_clip_feats.npy"), uavid_clip_feats)

    # Merge with the existing car/bus/truck comparison into one 4-class result
    existing_path = os.path.join(RESULTS_OUT, "drift_comparison.json")
    with open(existing_path) as f:
        existing = json.load(f)

    existing["clip_vit_b32"]["building"] = clip_building_drift
    existing["remoteclip_vit_l14"]["building"] = remoteclip_building_drift
    existing.setdefault("notes", "")
    existing["notes"] += (
        " | building added 2026-08-20: xView (satellite, streamed) vs UAVid (UAV, local Kaggle "
        "download of the official train+val split, VisDrone has no building class so UAVid substitutes "
        "for the UAV side on this one class only). Crops derived from UAVid's pixel-level masks via "
        "connected-component analysis, not native bounding boxes."
    )

    with open(os.path.join(RESULTS_OUT, "drift_comparison_4class.json"), "w") as f:
        json.dump(existing, f, indent=2)

    log(f"DONE. Merged 4-class result saved to {os.path.join(RESULTS_OUT, 'drift_comparison_4class.json')}")


if __name__ == "__main__":
    main()
