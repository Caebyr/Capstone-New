"""
Rebuilds a crop manifest (source image, box, label, cluster IDs) for every cached embedding file used
in the paper, and proves the manifest lines up row-for-row with the cached embeddings.

The original drivers (run_remoteclip_vs_clip.py, notebooks/04_finetune_visdrone_clip.ipynb,
run_xview_vs_visdrone.py, run_uavid_building.py) saved embeddings + labels but not which image each
crop came from. Crop extraction and subsample_balanced are deterministic (sorted file lists, seeded
random.Random), so replaying them with the same settings reproduces the same crop lists in the same
order. src/dataset.py now tags every crop with its provenance (PIL info dict, no effect on pixels or
ordering), which is what this script writes out.

Two checks per list, both must pass:
  1. the replayed label sequence equals the cached *_labels.npy exactly (every row);
  2. every replayed crop is re-encoded with each model and compared with ALL cached rows: >= 98% of
     crops must be nearest to their OWN row and the median own-row cosine must be >= 0.99. A row
     misalignment would collapse both. Not 100%: JPEG sources (VisDrone, xView) decode slightly
     differently under today's Pillow/libjpeg-turbo than in the original runs (cosine ~0.99, PNG
     sources DOTA/UAVid are exact), and in crowded embedding spaces (notably the fine-tuned model,
     where almost every crop has a near-twin above 0.97) that wobble can make a near-duplicate crop
     the nearest neighbour. Mismatched rows were inspected: all same-looking vehicles, own-row
     cosine >= 0.93. The report also records how much the drift values would move if re-encoded.

Image-level cluster IDs are what the bootstrap in drift_stats.py / classify_crops.py resamples.

Usage (hf env):  python build_crop_manifests.py
"""
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from src.crop_sets import LISTS
from src.dataset import (
    VISDRONE_TO_COMMON, VISDRONE_TO_EXPANDED, XVIEW_TO_EXPANDED, get_dota_crops, get_uavid_crops,
    get_visdrone_crops, get_xview_crops, split_visdrone_images, subsample_balanced,
)
from src.embeddings import encode_with_clip, encode_with_remoteclip, load_clip, load_remoteclip

# Settings copied from the drivers that produced the cached embeddings -- do not change.
MAX_CROPS_PER_CLASS = 300
RANDOM_SEED = 42
HOLDOUT_FRACTION, SPLIT_SEED = 0.2, 1234
MAX_XVIEW_IMAGES = 5000
MAX_XVIEW_IMAGES_FOR_BUILDING = 500
# The car/bus/truck run predates "Building" being added to XVIEW_TO_EXPANDED (2026-08-20); the cached
# xView labels contain car/bus/truck/van and no building, so replay with that earlier mapping.
XVIEW_TO_EXPANDED_PRE_BUILDING = {k: v for k, v in XVIEW_TO_EXPANDED.items() if v != "building"}

DOTA_IMG_DIR, DOTA_LABEL_DIR = "data/raw/DOTA/images", "data/raw/DOTA/labelTxt"
VISDRONE_BASE = "data/raw/VisDrone/VisDrone2019-DET-train"
VISDRONE_IMG_DIR, VISDRONE_ANN_DIR = os.path.join(VISDRONE_BASE, "images"), os.path.join(VISDRONE_BASE, "annotations")
UAVID_BASE = "data/raw/UAVid/modified_uavid_dataset"
REMOTECLIP_CKPT = "models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt"
FINETUNED_CKPT = "models/finetuned_clip_visdrone_heldout/finetuned_clip_visdrone_heldout_epoch3.pt"

OUT_DIR = "results/crop_manifests"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def replay_crop_lists(raw_counts=None):
    """Same calls, same order of RNG use, as the original drivers. If raw_counts is a dict it is
    filled with the per-class crop counts BEFORE subsampling, for comparison with the original logs."""
    lists = {}
    raw = raw_counts if raw_counts is not None else {}
    dota_all = get_dota_crops(DOTA_IMG_DIR, DOTA_LABEL_DIR)
    raw["p1_dota"] = dict(Counter(c for _, c in dota_all))
    lists["p1_dota"] = subsample_balanced(dota_all, MAX_CROPS_PER_CLASS, RANDOM_SEED)
    del dota_all
    log(f"p1_dota: {len(lists['p1_dota'])}")

    _, holdout_files = split_visdrone_images(VISDRONE_ANN_DIR, HOLDOUT_FRACTION, SPLIT_SEED)
    vis_holdout = get_visdrone_crops(VISDRONE_IMG_DIR, VISDRONE_ANN_DIR, class_map=VISDRONE_TO_COMMON, allowed_files=holdout_files)
    raw["p1_visdrone_heldout"] = dict(Counter(c for _, c in vis_holdout))
    lists["p1_visdrone_heldout"] = subsample_balanced(vis_holdout, MAX_CROPS_PER_CLASS, RANDOM_SEED)
    log(f"p1_visdrone_heldout: {len(lists['p1_visdrone_heldout'])} (from {len(holdout_files)} held-out images)")

    vis_all = get_visdrone_crops(VISDRONE_IMG_DIR, VISDRONE_ANN_DIR, class_map=VISDRONE_TO_EXPANDED)
    raw["p2_visdrone"] = dict(Counter(c for _, c in vis_all))
    lists["p2_visdrone"] = subsample_balanced(vis_all, MAX_CROPS_PER_CLASS, RANDOM_SEED)
    del vis_all
    log(f"p2_visdrone: {len(lists['p2_visdrone'])}")

    log(f"streaming xView ({MAX_XVIEW_IMAGES} images)...")
    xv = get_xview_crops(class_map=XVIEW_TO_EXPANDED_PRE_BUILDING, max_images=MAX_XVIEW_IMAGES, log=None)
    raw["p2_xview"] = dict(Counter(c for _, c in xv))
    lists["p2_xview"] = subsample_balanced(xv, MAX_CROPS_PER_CLASS, RANDOM_SEED)
    del xv
    log(f"p2_xview: {len(lists['p2_xview'])}")

    uavid = get_uavid_crops(os.path.join(UAVID_BASE, "train_data", "Images"), os.path.join(UAVID_BASE, "train_data", "Labels"), log=None)
    uavid += get_uavid_crops(os.path.join(UAVID_BASE, "val_data", "Images"), os.path.join(UAVID_BASE, "val_data", "Labels"), log=None)
    raw["p3_uavid_building"] = dict(Counter(c for _, c in uavid))
    lists["p3_uavid_building"] = subsample_balanced(uavid, MAX_CROPS_PER_CLASS, RANDOM_SEED)
    log(f"p3_uavid_building: {len(lists['p3_uavid_building'])}")

    xb = get_xview_crops(class_map={"Building": "building"}, max_images=MAX_XVIEW_IMAGES_FOR_BUILDING, log=None)
    raw["p3_xview_building"] = dict(Counter(c for _, c in xb))
    lists["p3_xview_building"] = subsample_balanced(xb, MAX_CROPS_PER_CLASS, RANDOM_SEED)
    log(f"p3_xview_building: {len(lists['p3_xview_building'])}")
    return lists


def write_manifest(name, crops):
    path = os.path.join(OUT_DIR, f"{name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "crop_id", "label", "dataset", "source_image", "sequence", "x1", "y1", "x2", "y2", "width", "height"])
        for i, (crop, label) in enumerate(crops):
            m = crop.info
            x1, y1, x2, y2 = m["bbox"]
            w.writerow([i, f"{name}:{i}", label, m["dataset"], m["source"], m.get("sequence", ""),
                        x1, y1, x2, y2, crop.size[0], crop.size[1]])
    return path


def reencode_sensitivity(lists, reencoded):
    """How much the reported per-category L2 drift would move if every crop were re-encoded today
    (JPEG decode differences). Reported values use the cached embeddings; this is a robustness note."""
    from src.crop_sets import TASKS
    from src.stats import drift
    out = {}
    for task, spec in TASKS.items():
        for model in spec["models"]:
            for cat in spec["categories"]:
                sides = {}
                for dom in ("overhead", "uav"):
                    for list_name, cats in spec[dom]:
                        if cat in cats:
                            lab = np.array([c for _, c in lists[list_name]])
                            mask = lab == cat
                            sides[dom] = (np.load(LISTS[list_name][0][model] + "_feats.npy")[mask].astype(np.float64),
                                          reencoded[(list_name, model)][mask].astype(np.float64))
                (oc, on), (uc, un) = sides["overhead"], sides["uav"]
                ones = lambda a: np.ones(len(a))
                d_c = drift(oc, ones(oc), uc, ones(uc))
                d_n = drift(on, ones(on), un, ones(un))
                out[f"{task}/{model}/{cat}"] = {k: [round(d_c[k], 4), round(d_n[k], 4)] for k in ("l2", "cos", "ratio")}
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")
    lists = replay_crop_lists()

    report = {"settings": {"max_crops_per_class": MAX_CROPS_PER_CLASS, "random_seed": RANDOM_SEED,
                           "holdout_fraction": HOLDOUT_FRACTION, "split_seed": SPLIT_SEED,
                           "max_xview_images": MAX_XVIEW_IMAGES, "max_xview_images_for_building": MAX_XVIEW_IMAGES_FOR_BUILDING},
              "lists": {}}
    all_ok = True

    # Check 1: labels, row for row.
    for name, crops in lists.items():
        prefixes, has_labels = LISTS[name]
        replay_labels = np.array([c for _, c in crops])
        entry = {"n": len(crops), "labels": dict(Counter(replay_labels.tolist())),
                 "n_source_images": len({c.info["source"] for c, _ in crops}),
                 "n_sequences": len({c.info.get("sequence") for c, _ in crops}) if crops and "sequence" in crops[0][0].info else None,
                 "manifest": write_manifest(name, crops), "label_check": {}}
        for model, prefix in prefixes.items():
            n_cached = np.load(prefix + "_feats.npy", mmap_mode="r").shape[0]
            if has_labels:
                cached = np.load(prefix + "_labels.npy", allow_pickle=True)
                ok = len(cached) == len(replay_labels) and bool((cached == replay_labels).all())
            else:
                ok = n_cached == len(replay_labels)
            entry["label_check"][model] = ok
            all_ok &= ok
        report["lists"][name] = entry
        log(f"{name}: n={entry['n']} images={entry['n_source_images']} labels-match={entry['label_check']}")

    # Check 2: re-encode EVERY crop with every model; each re-encoded crop's nearest cached row must be
    # its own row. Pixel-exact equality is not required: JPEG sources (VisDrone, xView) decode slightly
    # differently under today's Pillow/libjpeg-turbo than in the original runs, while PNG sources
    # (DOTA, UAVid) reproduce exactly. Row identity is what the manifests must guarantee.
    log("loading models for re-encoding check...")
    clip_model, clip_processor = load_clip(device)
    rc_model, rc_preprocess = load_remoteclip(REMOTECLIP_CKPT, device)
    ft_model, _ = load_clip(device)
    ft_model.load_state_dict(torch.load(FINETUNED_CKPT, map_location=device))
    ft_model.eval()

    reencoded = {}
    for name, crops in lists.items():
        prefixes, _ = LISTS[name]
        report["lists"][name]["reencode"] = {}
        for model, prefix in prefixes.items():
            cached = np.load(prefix + "_feats.npy")
            if model == "remoteclip":
                new, _ = encode_with_remoteclip(crops, rc_model, rc_preprocess, device, batch_size=4, log=None)
            else:
                new, _ = encode_with_clip(crops, clip_model if model == "clip" else ft_model, clip_processor, device, batch_size=4, log=None)
            reencoded[(name, model)] = new
            sims = new @ cached.T
            own = np.diag(sims)
            match_rate = float((sims.argmax(axis=1) == np.arange(len(crops))).mean())
            identity_ok = match_rate >= 0.98 and float(np.median(own)) >= 0.99
            stats = {"rows_matching_own_row": int((sims.argmax(axis=1) == np.arange(len(crops))).sum()), "n": len(crops),
                     "cosine_min": round(float(own.min()), 6), "cosine_median": round(float(np.median(own)), 6),
                     "frac_pixel_exact(cos>0.9999)": round(float((own > 0.9999).mean()), 4)}
            report["lists"][name]["reencode"][model] = stats
            all_ok &= identity_ok
            log(f"  {name} / {model}: own-row match {stats['rows_matching_own_row']}/{len(crops)}, "
                f"cos median {stats['cosine_median']:.4f} min {stats['cosine_min']:.4f}, pixel-exact {stats['frac_pixel_exact(cos>0.9999)']:.2f}")
    report["reencode_drift_sensitivity"] = reencode_sensitivity(lists, reencoded)

    from huggingface_hub import HfApi
    api = HfApi()
    report["assets"] = {
        "clip_hf_model": "openai/clip-vit-base-patch32",
        "clip_hf_revision": api.model_info("openai/clip-vit-base-patch32").sha,
        "xview_hf_dataset": "HichTala/xview",
        "xview_hf_revision": api.dataset_info("HichTala/xview").sha,
        "remoteclip_checkpoint": REMOTECLIP_CKPT, "remoteclip_sha256": sha256(REMOTECLIP_CKPT),
        "finetuned_checkpoint": FINETUNED_CKPT, "finetuned_sha256": sha256(FINETUNED_CKPT),
    }
    report["all_checks_passed"] = bool(all_ok)
    with open(os.path.join(OUT_DIR, "manifest_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    log(f"ALL CHECKS PASSED: {all_ok}")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
