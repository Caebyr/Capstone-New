"""
Standalone driver for the RemoteCLIP-vs-CLIP drift comparison (fixes PROJECT_CONTEXT.md Section 14a/14b).

Run directly with the hf env's python for real streaming progress output (unlike
`jupyter nbconvert --execute`, which only serializes cell output once the whole notebook
finishes). Saves embeddings + drift results to disk; notebooks/03_remoteclip.ipynb loads
these results for analysis/plotting rather than recomputing them.
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
from src.dataset import COMMON_VOCAB, get_dota_crops, get_visdrone_crops, split_visdrone_images, subsample_balanced
from src.embeddings import encode_with_clip, encode_with_remoteclip, load_clip, load_remoteclip

MAX_CROPS_PER_CLASS = 300  # reduced from 800 after the 30-min timeout; re-tune once real throughput is known
RANDOM_SEED = 42
BATCH_SIZE = 16

# Must match notebooks/04_finetune_visdrone_clip.ipynb exactly -- both this script and that
# notebook need to evaluate on the *same* VisDrone crops for the 2-way and 3-way results to be
# directly comparable (rather than two different measurements of "the same" thing).
HOLDOUT_FRACTION = 0.2
SPLIT_SEED = 1234

DOTA_IMG_DIR = "data/raw/DOTA/images"
DOTA_LABEL_DIR = "data/raw/DOTA/labelTxt"
VISDRONE_BASE = "data/raw/VisDrone/VisDrone2019-DET-train"
VISDRONE_IMG_DIR = os.path.join(VISDRONE_BASE, "images")
VISDRONE_ANN_DIR = os.path.join(VISDRONE_BASE, "annotations")

EMBED_OUT = "embeddings/remoteclip_vs_clip"
RESULTS_OUT = "results/remoteclip_vs_clip"
os.makedirs(EMBED_OUT, exist_ok=True)
os.makedirs(RESULTS_OUT, exist_ok=True)


def log(msg, flush=True):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=flush)


def compute_drift(dota_feats, dota_labels, vis_feats, vis_labels):
    scores = {}
    for cls in COMMON_VOCAB:
        sat = dota_feats[dota_labels == cls]
        uav = vis_feats[vis_labels == cls]
        if len(sat) == 0 or len(uav) == 0:
            log(f"  skipping {cls}: sat={len(sat)} uav={len(uav)}")
            continue
        sat_c, uav_c = sat.mean(axis=0), uav.mean(axis=0)
        l2 = float(norm(sat_c - uav_c))
        cos = float(1 - np.dot(sat_c, uav_c) / (norm(sat_c) * norm(uav_c)))
        sat_intra = float(np.mean(norm(sat - sat_c, axis=1)))
        uav_intra = float(np.mean(norm(uav - uav_c, axis=1)))
        scores[cls] = {
            "l2_drift": round(l2, 4),
            "cosine_drift": round(cos, 4),
            "drift_ratio": round(l2 / ((sat_intra + uav_intra) / 2), 4),
            "sat_samples": int(len(sat)),
            "uav_samples": int(len(uav)),
            "sat_intra_var": round(sat_intra, 4),
            "uav_intra_var": round(uav_intra, 4),
        }
    return scores


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")

    log("extracting DOTA ground-truth crops (all 469 local images)...")
    t0 = time.time()
    dota_crops_all = get_dota_crops(DOTA_IMG_DIR, DOTA_LABEL_DIR)
    log(f"DOTA raw crops: {len(dota_crops_all)}  ({time.time()-t0:.1f}s)  {dict(Counter(c for _, c in dota_crops_all))}")
    dota_crops = subsample_balanced(dota_crops_all, MAX_CROPS_PER_CLASS, RANDOM_SEED)
    log(f"DOTA subsampled: {len(dota_crops)}  {dict(Counter(c for _, c in dota_crops))}")

    log("splitting VisDrone into train/held-out pools (must match the fine-tuning notebook)...")
    train_files, holdout_files = split_visdrone_images(VISDRONE_ANN_DIR, HOLDOUT_FRACTION, SPLIT_SEED)
    log(f"VisDrone: {len(train_files)} train-pool images, {len(holdout_files)} held-out-pool images")

    log("extracting VisDrone ground-truth crops (held-out pool only, for comparability with the 3-way result)...")
    t0 = time.time()
    visdrone_crops_all = get_visdrone_crops(VISDRONE_IMG_DIR, VISDRONE_ANN_DIR, allowed_files=holdout_files)
    log(f"VisDrone raw crops: {len(visdrone_crops_all)}  ({time.time()-t0:.1f}s)  {dict(Counter(c for _, c in visdrone_crops_all))}")
    visdrone_crops = subsample_balanced(visdrone_crops_all, MAX_CROPS_PER_CLASS, RANDOM_SEED)
    log(f"VisDrone subsampled: {len(visdrone_crops)}  {dict(Counter(c for _, c in visdrone_crops))}")

    log("loading RemoteCLIP ViT-L/14...")
    remoteclip_model, remoteclip_preprocess = load_remoteclip("models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt", device)
    log("loading CLIP ViT-B/32...")
    clip_model, clip_processor = load_clip(device)

    log("encoding DOTA with RemoteCLIP...")
    t0 = time.time()
    dota_rc_feats, dota_rc_labels = encode_with_remoteclip(dota_crops, remoteclip_model, remoteclip_preprocess, device, BATCH_SIZE, log=log)
    log(f"DOTA RemoteCLIP done: {dota_rc_feats.shape}  ({time.time()-t0:.1f}s)")

    log("encoding VisDrone with RemoteCLIP...")
    t0 = time.time()
    vis_rc_feats, vis_rc_labels = encode_with_remoteclip(visdrone_crops, remoteclip_model, remoteclip_preprocess, device, BATCH_SIZE, log=log)
    log(f"VisDrone RemoteCLIP done: {vis_rc_feats.shape}  ({time.time()-t0:.1f}s)")

    log("encoding DOTA with CLIP...")
    t0 = time.time()
    dota_clip_feats, dota_clip_labels = encode_with_clip(dota_crops, clip_model, clip_processor, device, BATCH_SIZE, log=log)
    log(f"DOTA CLIP done: {dota_clip_feats.shape}  ({time.time()-t0:.1f}s)")

    log("encoding VisDrone with CLIP...")
    t0 = time.time()
    vis_clip_feats, vis_clip_labels = encode_with_clip(visdrone_crops, clip_model, clip_processor, device, BATCH_SIZE, log=log)
    log(f"VisDrone CLIP done: {vis_clip_feats.shape}  ({time.time()-t0:.1f}s)")

    log("computing drift (RemoteCLIP, native 768-dim space)...")
    remoteclip_drift = compute_drift(dota_rc_feats, dota_rc_labels, vis_rc_feats, vis_rc_labels)
    for cls, s in remoteclip_drift.items():
        log(f"  {cls}: {s}")

    log("computing drift (CLIP ViT-B/32, native 512-dim space)...")
    clip_drift = compute_drift(dota_clip_feats, dota_clip_labels, vis_clip_feats, vis_clip_labels)
    for cls, s in clip_drift.items():
        log(f"  {cls}: {s}")

    np.save(os.path.join(EMBED_OUT, "dota_remoteclip_feats.npy"), dota_rc_feats)
    np.save(os.path.join(EMBED_OUT, "dota_remoteclip_labels.npy"), dota_rc_labels)
    np.save(os.path.join(EMBED_OUT, "visdrone_remoteclip_feats.npy"), vis_rc_feats)
    np.save(os.path.join(EMBED_OUT, "visdrone_remoteclip_labels.npy"), vis_rc_labels)
    np.save(os.path.join(EMBED_OUT, "dota_clip_feats.npy"), dota_clip_feats)
    np.save(os.path.join(EMBED_OUT, "dota_clip_labels.npy"), dota_clip_labels)
    np.save(os.path.join(EMBED_OUT, "visdrone_clip_feats.npy"), vis_clip_feats)
    np.save(os.path.join(EMBED_OUT, "visdrone_clip_labels.npy"), vis_clip_labels)

    with open(os.path.join(RESULTS_OUT, "drift_comparison.json"), "w") as f:
        json.dump({
            "clip_vit_b32": clip_drift,
            "remoteclip_vit_l14": remoteclip_drift,
            "max_crops_per_class": MAX_CROPS_PER_CLASS,
            "random_seed": RANDOM_SEED,
            "holdout_fraction": HOLDOUT_FRACTION,
            "split_seed": SPLIT_SEED,
        }, f, indent=2)

    log(f"DONE. Saved embeddings to {EMBED_OUT} and results to {RESULTS_OUT}")


if __name__ == "__main__":
    main()
