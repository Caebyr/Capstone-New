"""
Read-only evidence collection for PROJECT_FACTS_FOR_PAPER.md. Recomputes nothing that the paper
reports; it only inspects manifests, cached files, configs and completed result files, and writes
results/paper_facts/facts.json. Does not touch embeddings, checkpoints or existing results.

Usage (hf env):  python collect_paper_facts.py
"""
import csv
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from src.crop_sets import TASKS, read_manifest

OUT_DIR = "results/paper_facts"
OLD_HF = r"D:\C drive data\Kabir Oberoi\.cache\huggingface\hub"       # restored pre-reset cache
NEW_HF = os.path.expanduser(r"~\.cache\huggingface\hub")             # current cache
CLIP_REPO = "models--openai--clip-vit-base-patch32"
facts = {}


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------- checkpoints / processor configs
def clip_cache(root):
    base = os.path.join(root, CLIP_REPO)
    out = {"refs_main": open(os.path.join(base, "refs", "main")).read().strip(), "snapshots": {}}
    for snap in sorted(os.listdir(os.path.join(base, "snapshots"))):
        d = os.path.join(base, "snapshots", snap)
        out["snapshots"][snap] = {f: {"bytes": os.path.getsize(os.path.join(d, f)),
                                      "sha256": sha256(os.path.join(d, f)) if f in ("pytorch_model.bin", "model.safetensors", "preprocessor_config.json", "config.json") else None}
                                  for f in sorted(os.listdir(d))}
    return out


facts["clip_cache_original"] = clip_cache(OLD_HF)
facts["clip_cache_current"] = clip_cache(NEW_HF)
main = facts["clip_cache_original"]["refs_main"]
facts["clip_preprocessor_config_json"] = json.load(open(os.path.join(OLD_HF, CLIP_REPO, "snapshots", main, "preprocessor_config.json")))

from transformers import CLIPProcessor
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
facts["clip_image_processor_effective"] = {k: (v if not hasattr(v, "tolist") else v.tolist()) for k, v in proc.image_processor.to_dict().items()}
facts["clip_image_processor_class"] = type(proc.image_processor).__name__

import open_clip
model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=None)
facts["remoteclip_open_clip"] = {
    "create_call": 'open_clip.create_model_and_transforms("ViT-L-14", pretrained=None) then load_state_dict(torch.load(ckpt), strict=False) with missing/unexpected keys raising',
    "visual_image_size": list(model.visual.image_size) if hasattr(model.visual.image_size, "__iter__") else model.visual.image_size,
    "preprocess_cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in model.visual.preprocess_cfg.items()},
    "transforms": [repr(t) for t in preprocess.transforms],
}
del model

from huggingface_hub import HfApi
try:
    info = HfApi().model_info("chendelong/RemoteCLIP", files_metadata=True)
    facts["remoteclip_official_release"] = {"repo": "chendelong/RemoteCLIP", "revision": info.sha,
                                           "files": {s.rfilename: {"size": s.size, "sha256": (s.lfs.sha256 if s.lfs else None)} for s in info.siblings}}
except Exception as e:  # network / repo issue: record, do not guess
    facts["remoteclip_official_release"] = {"error": repr(e)}
mr = json.load(open("results/crop_manifests/manifest_report.json"))
facts["local_checkpoints"] = {k: mr["assets"][k] for k in ("remoteclip_checkpoint", "remoteclip_sha256", "finetuned_checkpoint", "finetuned_sha256")}
facts["local_checkpoints"]["remoteclip_bytes"] = os.path.getsize(mr["assets"]["remoteclip_checkpoint"])
try:
    xi = HfApi().dataset_info("HichTala/xview")
    facts["xview_mirror"] = {"revision_now": xi.sha, "revision_in_original_cache": open(os.path.join(OLD_HF, "datasets--HichTala--xview", "refs", "main")).read().strip()}
    from datasets import load_dataset_builder
    feats = load_dataset_builder("HichTala/xview").info.features
    facts["xview_mirror"]["feature_columns"] = list(feats.keys())
    facts["xview_mirror"]["object_fields"] = list(feats["objects"].keys()) if "objects" in feats else None
except Exception as e:
    facts["xview_mirror"] = {"error": repr(e)}

# ---------------------------------------------------------------- crop sizes actually evaluated
used = {}
for task, spec in TASKS.items():
    for dom in ("overhead", "uav"):
        for list_name, cats in spec[dom]:
            used.setdefault(list_name, set()).update(cats)
sizes = {}
for list_name, cats in used.items():
    rows = [r for r in read_manifest(list_name) if r["label"] in cats]
    w = np.array([int(r["width"]) for r in rows]); h = np.array([int(r["height"]) for r in rows])
    small = (w < 16) | (h < 16)
    sizes[list_name] = {
        "evaluated_crops": len(rows), "min_width": int(w.min()), "min_height": int(h.min()),
        "crops_with_a_side_below_16": int(small.sum()),
        "below_16_detail": [{"crop_id": r["crop_id"], "label": r["label"], "source_image": r["source_image"],
                             "box": [int(r[k]) for k in ("x1", "y1", "x2", "y2")], "width": int(r["width"]), "height": int(r["height"])}
                            for r, s in zip(rows, small) if s],
        "median_min_side": float(np.median(np.minimum(w, h))),
        "per_label": {c: {"n": sum(1 for r in rows if r["label"] == c),
                          "images": len({r["source_image"] for r in rows if r["label"] == c}),
                          "sequences": len({r["sequence"] for r in rows if r["label"] == c and r["sequence"]}),
                          "max_crops_from_one_image": max(Counter(r["source_image"] for r in rows if r["label"] == c).values())} for c in sorted(cats)},
    }
facts["evaluated_crop_sizes"] = sizes

# ---------------------------------------------------------------- DOTA platform / GSD metadata of sampled images
dota_rows = read_manifest("p1_dota")
meta = {}
for img in sorted({r["source_image"] for r in dota_rows}):
    p = os.path.join("data/raw/DOTA/labelTxt", img.replace(".png", ".txt"))
    head = [l.strip() for l in open(p, encoding="utf-8", errors="replace").readlines()[:2]]
    src = next((l.split(":", 1)[1] for l in head if l.lower().startswith("imagesource")), None)
    gsd = next((l.split(":", 1)[1] for l in head if l.lower().startswith("gsd")), None)
    meta[img] = {"imagesource": src, "gsd": gsd}
by_img = Counter(m["imagesource"] for m in meta.values())
by_crop = {c: dict(Counter(meta[r["source_image"]]["imagesource"] for r in dota_rows if r["label"] == c)) for c in ("small-vehicle", "large-vehicle")}
gsds = [float(m["gsd"]) for m in meta.values() if m["gsd"] not in (None, "", "null", "None")]
all_src = Counter()
for f in os.listdir("data/raw/DOTA/images"):
    p = os.path.join("data/raw/DOTA/labelTxt", f.replace(".png", ".txt"))
    if os.path.exists(p):
        l0 = open(p, encoding="utf-8", errors="replace").readline().strip()
        all_src[l0.split(":", 1)[1] if l0.lower().startswith("imagesource") else "(no header)"] += 1
facts["dota_metadata"] = {
    "header_format": "first two lines of each labelTxt file: 'imagesource:<name>' and 'gsd:<metres per pixel or null>'",
    "sampled_images": len(meta), "sampled_images_by_imagesource": dict(by_img), "sampled_crops_by_label_and_imagesource": by_crop,
    "sampled_images_gsd_missing": sum(1 for m in meta.values() if m["gsd"] in (None, "", "null", "None")),
    "sampled_images_gsd_min_median_max": [min(gsds), float(np.median(gsds)), max(gsds)] if gsds else None,
    "all_469_local_images_by_imagesource": dict(all_src),
}

# ---------------------------------------------------------------- statistics: settings, valid replicates, image vs sequence
ds = json.load(open("results/drift_stats/drift_stats.json"))
cb = json.load(open("results/classification/classification_bootstrap.json"))
facts["drift_stats_settings"] = ds["settings"]
facts["drift_valid_replicates"] = {f"{t}/{lvl}/{m}/{c}": ds["tasks"][t]["bootstrap"][lvl][m][c]["n_valid_reps"]
                                   for t in ds["tasks"] for lvl in ("image", "sequence") for m in ds["tasks"][t]["models"] for c in ds["tasks"][t]["categories"]}
facts["drift_valid_replicates_min"] = min(facts["drift_valid_replicates"].values())
facts["cluster_counts"] = {t: {"images": ds["tasks"][t]["n_images"], "uav_sequences": ds["tasks"][t]["n_sequences_uav"]} for t in ds["tasks"]}
cmp_rows = []
for t in ds["tasks"]:
    for m in ds["tasks"][t]["models"]:
        for c in ds["tasks"][t]["categories"]:
            bi, bs = ds["tasks"][t]["bootstrap"]["image"][m][c], ds["tasks"][t]["bootstrap"]["sequence"][m][c]
            row = {"task": t, "model": m, "category": c, "point_l2": ds["tasks"][t]["point"][m][c]["l2"], "point_ratio": ds["tasks"][t]["point"][m][c]["ratio"]}
            for k in ("l2", "ratio") + (("Y",) if "Y" in bi else ()) + (("ratio_pct_vs_clip",) if "ratio_pct_vs_clip" in bi else ()):
                row[f"{k}_image_ci"], row[f"{k}_sequence_ci"] = bi[k], bs[k]
            ci, cs = cb["tasks"][t]["bootstrap"]["image"][m][c], cb["tasks"][t]["bootstrap"]["sequence"][m][c]
            for k in ("recall_gap_pp", "delta_m"):
                row[f"{k}_image_ci"], row[f"{k}_sequence_ci"] = ci[k], cs[k]
            cmp_rows.append(row)
facts["image_vs_sequence_intervals"] = cmp_rows
facts["accuracy_gaps"] = {t: {m: {"point_gap_pp": cb["tasks"][t]["point"][m]["gap"]["accuracy_pp"],
                                  "image_ci": cb["tasks"][t]["bootstrap"]["image"][m]["accuracy_gap_pp"],
                                  "sequence_ci": cb["tasks"][t]["bootstrap"]["sequence"][m]["accuracy_gap_pp"],
                                  "gap_minus_clip_gap_image_ci": cb["tasks"][t]["bootstrap"]["image"][m].get("accuracy_gap_diff_vs_clip_pp"),
                                  "gap_minus_clip_gap_sequence_ci": cb["tasks"][t]["bootstrap"]["sequence"][m].get("accuracy_gap_diff_vs_clip_pp")}
                              for m in cb["tasks"][t]["models"]} for t in cb["tasks"]}
for t in cb["tasks"]:
    for m in cb["tasks"][t]["models"]:
        if m != "clip":
            facts["accuracy_gaps"][t][m]["point_gap_minus_clip_gap_pp"] = (cb["tasks"][t]["point"][m]["gap"]["accuracy_pp"] - cb["tasks"][t]["point"]["clip"]["gap"]["accuracy_pp"])

# paired per-domain accuracy difference between encoders (image bootstrap on the saved predictions)
pred = list(csv.DictReader(open("results/classification/predictions.csv", encoding="utf-8")))
rng = np.random.default_rng(11)
enc_diff = {}
for task in ("P1", "P2+P3"):
    for other in [m for m in TASKS[task]["models"] if m != "clip"]:
        for dom in ("overhead", "uav"):
            a = [r for r in pred if r["task"] == task and r["domain"] == dom and r["model"] == "clip"]
            b = [r for r in pred if r["task"] == task and r["domain"] == dom and r["model"] == other]
            assert [r["crop_id"] for r in a] == [r["crop_id"] for r in b]
            ca, cb_ = np.array([int(r["correct"]) for r in a], float), np.array([int(r["correct"]) for r in b], float)
            uniq, inv = np.unique([r["source_image"] for r in a], return_inverse=True)
            d = []
            for _ in range(4000):
                w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[inv].astype(float)
                d.append(100 * ((w * cb_).sum() - (w * ca).sum()) / w.sum())
            enc_diff[f"{task}/{dom}/{other}_minus_clip"] = {"point_pp": float(100 * (cb_.mean() - ca.mean())),
                                                           "image_bootstrap_95ci_pp": [float(x) for x in np.quantile(d, [.025, .975])], "replicates": 4000, "seed": 11}
facts["paired_encoder_accuracy_difference"] = enc_diff

# ---------------------------------------------------------------- correlations
jt = list(csv.DictReader(open("results/joined/joined_table.csv", encoding="utf-8")))
pooled = [r for r in jt if r["task"] == "P2+P3"]
corr_rows = [{"model": r["model"], "category": r["category"], "cos_drift": float(r["cos"]), "ratio": float(r["ratio"]), "Y": float(r["Y"]),
              "recall_overhead": float(r["recall_overhead"]), "recall_uav": float(r["recall_uav"]), "recall_gap_pp_overhead_minus_uav": float(r["recall_gap_pp"]),
              "A_c": float(r["A"]), "delta_m": float(r["delta_m"])} for r in pooled]
x = [r["cos_drift"] for r in corr_rows]


def rho(a, b):
    return float(spearmanr(a, b).correlation)


facts["correlations"] = {
    "rows": corr_rows,
    "stored": json.load(open("results/joined/association.json")),
    "pooled_8_points": {
        "cos_vs_abs_recall_gap  (the reported 0.02)": rho(x, [abs(r["recall_gap_pp_overhead_minus_uav"]) for r in corr_rows]),
        "cos_vs_signed_recall_gap (overhead - UAV)": rho(x, [r["recall_gap_pp_overhead_minus_uav"] for r in corr_rows]),
        "cos_vs_overhead_recall_loss (UAV - overhead)": rho(x, [-r["recall_gap_pp_overhead_minus_uav"] for r in corr_rows]),
        "cos_vs_A_c": rho(x, [r["A_c"] for r in corr_rows]),
        "cos_vs_abs_delta_m": rho(x, [abs(r["delta_m"]) for r in corr_rows]),
    },
    "ranks_cos_drift": rankdata(x).tolist(),
    "ranks_abs_recall_gap": rankdata([abs(r["recall_gap_pp_overhead_minus_uav"]) for r in corr_rows]).tolist(),
    "ties_present": {"cos_drift": len(set(x)) < len(x),
                     "abs_recall_gap": len({round(abs(r["recall_gap_pp_overhead_minus_uav"]), 9) for r in corr_rows}) < len(corr_rows)},
    "within_encoder": {m: {"cos_vs_A_c": rho([r["cos_drift"] for r in corr_rows if r["model"] == m], [r["A_c"] for r in corr_rows if r["model"] == m]),
                           "cos_vs_abs_delta_m": rho([r["cos_drift"] for r in corr_rows if r["model"] == m], [abs(r["delta_m"]) for r in corr_rows if r["model"] == m]),
                           "cos_vs_abs_recall_gap": rho([r["cos_drift"] for r in corr_rows if r["model"] == m], [abs(r["recall_gap_pp_overhead_minus_uav"]) for r in corr_rows if r["model"] == m]),
                           "cos_vs_signed_recall_gap": rho([r["cos_drift"] for r in corr_rows if r["model"] == m], [r["recall_gap_pp_overhead_minus_uav"] for r in corr_rows if r["model"] == m])}
                       for m in ("clip", "remoteclip")},
}

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "facts.json"), "w") as f:
    json.dump(facts, f, indent=2, default=str)
print("wrote", os.path.join(OUT_DIR, "facts.json"))
