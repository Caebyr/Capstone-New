"""
Robustness check for the three-way drift comparison (results/remoteclip_vs_clip/drift_comparison_3way.json).

Reuses the already-trained fine-tuned checkpoint (no retraining) and re-runs the fast
encoding/drift-computation step across several different random seeds, to check whether the
reported ranking (fine-tuned-CLIP < RemoteCLIP < CLIP) holds up across different random samples
of evaluation crops, not just the one seed (42) already reported.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from numpy.linalg import norm
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, os.path.dirname(__file__))
from src.dataset import COMMON_VOCAB, get_dota_crops, get_visdrone_crops, subsample_balanced
from src.embeddings import encode_with_clip, encode_with_remoteclip, load_clip, load_remoteclip

SEEDS = [42, 123, 7, 2024, 99]
MAX_CROPS_PER_CLASS = 300

DOTA_IMG_DIR = "data/raw/DOTA/images"
DOTA_LABEL_DIR = "data/raw/DOTA/labelTxt"
VISDRONE_BASE = "data/raw/VisDrone/VisDrone2019-DET-train"
VISDRONE_IMG_DIR = os.path.join(VISDRONE_BASE, "images")
VISDRONE_ANN_DIR = os.path.join(VISDRONE_BASE, "annotations")

FINETUNED_CKPT = "models/finetuned_clip_visdrone/finetuned_clip_visdrone_epoch3.pt"
OUT_PATH = "results/remoteclip_vs_clip/seed_sensitivity.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_drift(dota_feats, dota_labels, vis_feats, vis_labels):
    scores = {}
    for cls in COMMON_VOCAB:
        sat = dota_feats[dota_labels == cls]
        uav = vis_feats[vis_labels == cls]
        if len(sat) == 0 or len(uav) == 0:
            continue
        sat_c, uav_c = sat.mean(axis=0), uav.mean(axis=0)
        scores[cls] = float(norm(sat_c - uav_c))
    return scores


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")

    log("scanning full DOTA + VisDrone crop sets once (reused across all seeds)...")
    dota_crops_all = get_dota_crops(DOTA_IMG_DIR, DOTA_LABEL_DIR)
    visdrone_crops_all = get_visdrone_crops(VISDRONE_IMG_DIR, VISDRONE_ANN_DIR)
    log(f"DOTA: {len(dota_crops_all)} crops, VisDrone: {len(visdrone_crops_all)} crops")

    log("loading CLIP, RemoteCLIP, and the fine-tuned checkpoint...")
    clip_model, clip_processor = load_clip(device)
    remoteclip_model, remoteclip_preprocess = load_remoteclip("models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt", device)
    finetuned_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    finetuned_model.load_state_dict(torch.load(FINETUNED_CKPT, map_location=device))
    finetuned_model.eval()
    finetuned_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    results = {"clip_vit_b32": {}, "remoteclip_vit_l14": {}, "finetuned_clip_visdrone": {}}

    for seed in SEEDS:
        log(f"--- seed {seed} ---")
        dota_eval = subsample_balanced(dota_crops_all, MAX_CROPS_PER_CLASS, seed)
        vis_eval = subsample_balanced(visdrone_crops_all, MAX_CROPS_PER_CLASS, seed)

        d_feats, d_labels = encode_with_clip(dota_eval, clip_model, clip_processor, device, log=None)
        v_feats, v_labels = encode_with_clip(vis_eval, clip_model, clip_processor, device, log=None)
        results["clip_vit_b32"][seed] = compute_drift(d_feats, d_labels, v_feats, v_labels)

        d_feats, d_labels = encode_with_remoteclip(dota_eval, remoteclip_model, remoteclip_preprocess, device, log=None)
        v_feats, v_labels = encode_with_remoteclip(vis_eval, remoteclip_model, remoteclip_preprocess, device, log=None)
        results["remoteclip_vit_l14"][seed] = compute_drift(d_feats, d_labels, v_feats, v_labels)

        d_feats, d_labels = encode_with_clip(dota_eval, finetuned_model, finetuned_processor, device, log=None)
        v_feats, v_labels = encode_with_clip(vis_eval, finetuned_model, finetuned_processor, device, log=None)
        results["finetuned_clip_visdrone"][seed] = compute_drift(d_feats, d_labels, v_feats, v_labels)

        for cls in COMMON_VOCAB:
            c = results["clip_vit_b32"][seed][cls]
            r = results["remoteclip_vit_l14"][seed][cls]
            f = results["finetuned_clip_visdrone"][seed][cls]
            ranking = "OK (finetuned < remoteclip < clip)" if f < r < c else f"DIFFERENT: clip={c:.4f} remoteclip={r:.4f} finetuned={f:.4f}"
            log(f"  {cls}: clip={c:.4f} remoteclip={r:.4f} finetuned={f:.4f}  -> {ranking}")

    log("summary across seeds:")
    for arm in results:
        for cls in COMMON_VOCAB:
            vals = [results[arm][s][cls] for s in SEEDS]
            log(f"  {arm} / {cls}: mean={np.mean(vals):.4f} std={np.std(vals):.4f} min={min(vals):.4f} max={max(vals):.4f}")

    with open(OUT_PATH, "w") as f:
        json.dump({"seeds": SEEDS, "max_crops_per_class": MAX_CROPS_PER_CLASS, "results": results}, f, indent=2)
    log(f"saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
