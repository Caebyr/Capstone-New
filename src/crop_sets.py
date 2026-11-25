"""Where every cached embedding set lives, and how the paper's evaluation tasks are assembled from them.

Row i of a cached *_feats.npy is crop i of the matching manifest in results/crop_manifests/
(built and verified by build_crop_manifests.py). All models in one list share the same rows.
"""
import csv
import os

import numpy as np

MANIFEST_DIR = "results/crop_manifests"

# list name -> {model: cached embedding prefix}, and whether a *_labels.npy exists alongside
LISTS = {
    "p1_dota": ({"clip": "embeddings/remoteclip_vs_clip/dota_clip",
                 "remoteclip": "embeddings/remoteclip_vs_clip/dota_remoteclip",
                 "finetuned": "embeddings/remoteclip_vs_clip/dota_finetuned_clip_heldout"}, True),
    "p1_visdrone_heldout": ({"clip": "embeddings/remoteclip_vs_clip/visdrone_clip",
                             "remoteclip": "embeddings/remoteclip_vs_clip/visdrone_remoteclip",
                             "finetuned": "embeddings/remoteclip_vs_clip/visdrone_finetuned_clip_heldout"}, True),
    "p2_xview": ({"clip": "embeddings/xview_vs_visdrone/xview_clip",
                  "remoteclip": "embeddings/xview_vs_visdrone/xview_remoteclip"}, True),
    "p2_visdrone": ({"clip": "embeddings/xview_vs_visdrone/visdrone_clip",
                     "remoteclip": "embeddings/xview_vs_visdrone/visdrone_remoteclip"}, True),
    "p3_xview_building": ({"clip": "embeddings/xview_vs_visdrone/xview_building_clip",
                           "remoteclip": "embeddings/xview_vs_visdrone/xview_building_remoteclip"}, False),
    "p3_uavid_building": ({"clip": "embeddings/xview_vs_visdrone/uavid_building_clip",
                           "remoteclip": "embeddings/xview_vs_visdrone/uavid_building_remoteclip"}, False),
}

# Evaluation tasks. Each domain is a list of (crop list, categories taken from it).
# P1 keeps its own two-label vocabulary; P2+P3 are pooled into one four-label vocabulary. The two are
# never merged: small-/large-vehicle overlap car/bus/truck.
TASKS = {
    "P1": {
        "categories": ["small-vehicle", "large-vehicle"],
        "models": ["clip", "remoteclip", "finetuned"],
        "overhead": [("p1_dota", ["small-vehicle", "large-vehicle"])],
        "uav": [("p1_visdrone_heldout", ["small-vehicle", "large-vehicle"])],
    },
    "P2+P3": {
        "categories": ["car", "bus", "truck", "building"],
        "models": ["clip", "remoteclip"],
        "overhead": [("p2_xview", ["car", "bus", "truck"]), ("p3_xview_building", ["building"])],
        "uav": [("p2_visdrone", ["car", "bus", "truck"]), ("p3_uavid_building", ["building"])],
    },
}

MODEL_NAMES = {"clip": "CLIP ViT-B/32", "remoteclip": "RemoteCLIP ViT-L/14", "finetuned": "CLIP fine-tuned on VisDrone"}


def read_manifest(name):
    with open(os.path.join(MANIFEST_DIR, f"{name}.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_domain(task, domain, model):
    """Concatenated crops of one task/domain/model, restricted to the task's categories.

    Returns dict with feats [N, D] (unit-norm, as cached), labels, crop_ids, image ids (cluster
    for image-level resampling) and sequence ids (cluster for sequence-level resampling; for
    overhead sources, which have no sequences, this falls back to the image id).
    """
    feats, labels, crop_ids, images, sequences = [], [], [], [], []
    for list_name, cats in TASKS[task][domain]:
        prefix = LISTS[list_name][0][model]
        f = np.load(prefix + "_feats.npy")
        rows = read_manifest(list_name)
        assert len(rows) == len(f), f"{list_name}: manifest has {len(rows)} rows, embeddings {len(f)}"
        for i, r in enumerate(rows):
            if r["label"] not in cats:
                continue
            feats.append(f[i])
            labels.append(r["label"])
            crop_ids.append(r["crop_id"])
            img = f"{r['dataset']}/{r['source_image']}"
            images.append(img)
            sequences.append(f"{r['dataset']}/seq/{r['sequence']}" if r["sequence"] else img)
    return {"feats": np.stack(feats).astype(np.float64), "labels": np.array(labels), "crop_ids": np.array(crop_ids),
            "images": np.array(images), "sequences": np.array(sequences)}
