"""
Checkpoint A3: do the paper's claims hold at full scale and across sampling seeds?

Uses the full-scale cache from encode_full_scale.py (up to 5,000 crops per category and domain, drawn at
random from the full datasets) and the paper's own metric code, unchanged:
  drift / Y_c        drift_stats.task_stats, drift_stats.permutation_test (src/stats.py)
  classification     classify_crops.classify / class_metrics / text_metrics, prompt "a photo of a {}"

Two analyses:
  1. FULL   all cached crops; image-level bootstrap CIs (N_BOOT) and image-level permutation p (N_PERM).
  2. SEEDS  the paper's protocol (300 crops per category and domain) repeated for N_SEEDS independent
            seeds drawn from the cache; distribution of every number, and how often each claim holds.
Tasks: P1 (DOTA vs held-out VisDrone; CLIP, RemoteCLIP, fine-tuned), P1a (DOTA vs all VisDrone; CLIP,
RemoteCLIP), P2+P3 (xView vs VisDrone/UAVid, car/bus/truck/building), P2van (xView vs VisDrone van; drift only).

Writes results/full_scale/drift_full_scale.json and results/full_scale/drift_full_scale.md.
"""
import csv
import gzip
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_crops import class_metrics, classify, text_embeddings, text_metrics
from drift_stats import permutation_test, task_stats
from src.stats import ci, cluster_weights

EMB, OUT = "embeddings/full_scale", "results/full_scale"
N_BOOT, N_PERM, N_SEEDS, PER_CAT = 1000, 1000, 20, 300
SEED = 20261126

TASKS = {
    "P1": {"cats": ["small-vehicle", "large-vehicle"], "models": ["clip", "remoteclip", "finetuned"], "text_task": "P1",
           "overhead": [("P1_dota", None)], "uav": [("P1_visdrone_heldout", None)]},
    "P1a": {"cats": ["small-vehicle", "large-vehicle"], "models": ["clip", "remoteclip"], "text_task": "P1",
            "overhead": [("P1_dota", None)], "uav": [("P1a_visdrone_all", None)]},
    "P2+P3": {"cats": ["car", "bus", "truck", "building"], "models": ["clip", "remoteclip"], "text_task": "P2+P3",
              "overhead": [("P2_xview", ["car", "bus", "truck"]), ("P3_xview_building", None)],
              "uav": [("P2_visdrone", ["car", "bus", "truck"]), ("P3_uavid_building", None)]},
    "P2van": {"cats": ["van"], "models": ["clip", "remoteclip"], "text_task": None,
              "overhead": [("P2_xview", ["van"])], "uav": [("P2_visdrone", ["van"])]},
}
MODEL_NAMES = {"clip": "CLIP ViT-B/32", "remoteclip": "RemoteCLIP ViT-L/14", "finetuned": "CLIP fine-tuned on VisDrone"}
# The paper's values (Tables II-IV), for side-by-side comparison
PAPER = {
    "P1": {"clip": {"small-vehicle": (0.2460, 0.6498), "large-vehicle": (0.2596, 0.6235)},
           "remoteclip": {"small-vehicle": (0.1892, 0.5627), "large-vehicle": (0.2276, 0.6252)},
           "finetuned": {"small-vehicle": (0.1585, 0.8165), "large-vehicle": (0.0955, 0.4132)}},
    "P2+P3": {"clip": {"car": (0.3564, 0.9931), "bus": (0.3256, 0.8443), "truck": (0.2837, 0.7550), "building": (0.5417, 1.2219)},
              "remoteclip": {"car": (0.3024, 0.9719), "bus": (0.2802, 0.8491), "truck": (0.2366, 0.7500), "building": (0.3783, 0.9809)}},
}
PAPER_Y = {"clip": {"car": 1.95, "bus": 1.81, "truck": 1.50, "building": 2.32}, "remoteclip": {"car": 2.09, "bus": 2.30, "truck": 1.81, "building": 2.12}}
PAPER_ACC = {"clip": (32.9, 60.2), "remoteclip": (27.3, 51.8)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cache():
    with gzip.open(f"{OUT}/encode_manifest.csv.gz", "rt", newline="", encoding="utf-8") as f:
        man = {r["box_id"]: r for r in csv.DictReader(f)}
    with gzip.open(f"{OUT}/list_membership.csv.gz", "rt", newline="", encoding="utf-8") as f:
        member = list(csv.DictReader(f))
    feats = {}
    for m in ("clip", "remoteclip", "finetuned"):
        ids = open(f"{EMB}/{m}_box_ids.txt").read().split()
        feats[m] = (np.load(f"{EMB}/{m}_feats.npy").astype(np.float64), {b: i for i, b in enumerate(ids)})
    return man, member, feats


def build(task, man, member, feats):
    """data[model][domain] = {feats, labels, images, sequences}; rows aligned across models."""
    spec = TASKS[task]
    data = {m: {} for m in spec["models"]}
    for d in ("overhead", "uav"):
        sel = []
        for list_name, cats in spec[d]:
            sel += [r for r in member if r["list"] == list_name and (cats is None or r["label"] in cats) and r["label"] in spec["cats"]]
        ids = [r["box_id"] for r in sel]
        labels = np.array([r["label"] for r in sel])
        imgs = np.array([f"{man[b]['dataset']}/{man[b]['source_image']}" for b in ids])
        seqs = np.array([f"{man[b]['dataset']}/seq/{man[b]['sequence']}" if man[b]["sequence"] else f"{man[b]['dataset']}/{man[b]['source_image']}" for b in ids])
        for m in spec["models"]:
            F, index = feats[m]
            data[m][d] = {"feats": F[[index[b] for b in ids]], "labels": labels, "images": imgs, "sequences": seqs}
    return data


def classification(task, data, text, w=None):
    spec = TASKS[task]
    cats, out = spec["cats"], {}
    for m in spec["models"]:
        out[m] = {}
        for d in ("overhead", "uav"):
            x = data[m][d]
            wd = None if w is None else w[d]
            _, pred = classify(x["feats"], text[m][spec["text_task"]], cats)
            res = class_metrics(x["labels"], pred, cats, wd)
            res["text"] = text_metrics(x["feats"], x["labels"], text[m][spec["text_task"]], cats, wd)
            res["pred_counts_on_vehicles"] = {c: float(np.sum((wd if wd is not None else 1) * ((pred == c) & (x["labels"] != "building")))) for c in cats}
            out[m][d] = res
        o, u = out[m]["overhead"], out[m]["uav"]
        out[m]["gap"] = {"accuracy_pp": 100 * (o["accuracy"] - u["accuracy"]),
                         "recall_pp": {c: 100 * (o["per_category"][c]["recall"] - u["per_category"][c]["recall"]) for c in cats},
                         "delta_m": {c: o["text"][c]["margin"] - u["text"][c]["margin"] for c in cats},
                         "A": {c: abs(o["text"][c]["a"] - u["text"][c]["a"]) for c in cats}}
    return out


def seed_sample(data, cats, rng):
    """Paper protocol: PER_CAT crops per category and domain, without replacement; same rows for every model."""
    any_m = next(iter(data))
    w = {}
    for d in ("overhead", "uav"):
        lab = data[any_m][d]["labels"]
        wd = np.zeros(len(lab))
        for c in cats:
            idx = np.where(lab == c)[0]
            wd[rng.choice(idx, min(PER_CAT, len(idx)), replace=False)] = 1
        w[d] = wd
    return w


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def claims_p2p3(dr, cl):
    """The paper's qualitative statements, evaluated on one set of results."""
    cats = ["car", "bus", "truck", "building"]
    rc, cp = dr["remoteclip"], dr["clip"]
    c = {
        "RemoteCLIP raw L2 < CLIP (all 4 P2+P3 cats)": all(rc[k]["l2"] < cp[k]["l2"] for k in cats),
        "RemoteCLIP ratio lower than CLIP for building": rc["building"]["ratio"] < cp["building"]["ratio"],
        "Y_c > 1 in all 8 cells": all(dr[m][k]["Y"] > 1 for m in ("clip", "remoteclip") for k in cats),
    }
    if cl is not None:
        c.update({
            "overhead acc < UAV acc (CLIP and RemoteCLIP)": all(cl[m]["overhead"]["accuracy"] < cl[m]["uav"]["accuracy"] for m in ("clip", "remoteclip")),
            "RemoteCLIP overhead acc <= CLIP overhead acc": cl["remoteclip"]["overhead"]["accuracy"] <= cl["clip"]["overhead"]["accuracy"],
            "delta_m < 0 in all 8 cells": all(cl[m]["gap"]["delta_m"][k] < 0 for m in ("clip", "remoteclip") for k in cats),
            "CLIP labels most overhead vehicles 'car'": max(cl["clip"]["overhead"]["pred_counts_on_vehicles"], key=cl["clip"]["overhead"]["pred_counts_on_vehicles"].get) == "car",
            "RemoteCLIP labels most overhead vehicles 'building'": max(cl["remoteclip"]["overhead"]["pred_counts_on_vehicles"], key=cl["remoteclip"]["overhead"]["pred_counts_on_vehicles"].get) == "building",
        })
        pts_d = [dr[m][k]["cos"] for m in ("clip", "remoteclip") for k in cats]
        pts_g = [abs(cl[m]["gap"]["recall_pp"][k]) for m in ("clip", "remoteclip") for k in cats]
        c["pooled Spearman(cos drift, |recall gap|) < 0.5"] = spearman(pts_d, pts_g) < 0.5
    return c


def claims_p1(dr):
    rc, cp, ft = dr["remoteclip"], dr["clip"], dr.get("finetuned")
    c = {"RemoteCLIP raw L2 < CLIP (both P1 cats)": all(rc[k]["l2"] < cp[k]["l2"] for k in ("small-vehicle", "large-vehicle")),
         "RemoteCLIP ratio lower than CLIP for small-vehicle": rc["small-vehicle"]["ratio"] < cp["small-vehicle"]["ratio"]}
    if ft:
        c.update({"fine-tuned raw L2 < CLIP (both)": all(ft[k]["l2"] < cp[k]["l2"] for k in ("small-vehicle", "large-vehicle")),
                  "fine-tuned ratio > CLIP for small-vehicle": ft["small-vehicle"]["ratio"] > cp["small-vehicle"]["ratio"],
                  "fine-tuned ratio < CLIP for large-vehicle": ft["large-vehicle"]["ratio"] < cp["large-vehicle"]["ratio"]})
    return c


def main():
    t0 = time.time()
    man, member, feats = load_cache()
    tf = f"{OUT}/text_embeddings.npz"
    if os.path.exists(tf):
        z = np.load(tf)
        text = {m: {t: z[f"{m}|{t}"] for t in ("P1", "P2+P3")} for m in ("clip", "remoteclip", "finetuned")}
    else:
        text, _ = text_embeddings("cuda")
        np.savez(tf, **{f"{m}|{t}": v for m, d in text.items() for t, v in d.items()})
        torch.cuda.empty_cache()
    rng = np.random.default_rng(SEED)
    results = {"settings": {"n_boot": N_BOOT, "n_perm": N_PERM, "n_seeds": N_SEEDS, "per_cat": PER_CAT, "seed": SEED}, "tasks": {}}

    for task, spec in TASKS.items():
        data = build(task, man, member, feats)
        cats, models = spec["cats"], spec["models"]
        any_m = models[0]
        n = {d: {c: int((data[any_m][d]["labels"] == c).sum()) for c in cats} for d in ("overhead", "uav")}
        n_img = {d: {c: int(len(np.unique(data[any_m][d]["images"][data[any_m][d]["labels"] == c]))) for c in cats} for d in ("overhead", "uav")}
        log(f"{task}: crops {n}")
        ones = {d: np.ones(len(data[any_m][d]["labels"])) for d in ("overhead", "uav")}
        full = task_stats(data, cats, models, ones["overhead"], ones["uav"])
        cls = classification(task, data, text) if spec["text_task"] else None
        perm = {m: {c: permutation_test(data[m]["overhead"], data[m]["uav"], c, rng, N_PERM) for c in cats} for m in models}
        log(f"{task}: point estimates + permutations done")

        # image-level bootstrap (shared draw across models, independent per domain)
        boot = {m: {c: {"l2": [], "ratio": [], "Y": []} for c in cats} for m in models}
        boot_cls = {m: {"acc_gap": [], **{f"dm_{c}": [] for c in cats}} for m in models}
        boot_ratio_diff = {m: {c: [] for c in cats} for m in models if m != "clip"}
        for b in range(N_BOOT):
            w = {d: cluster_weights(rng, data[any_m][d]["images"]) for d in ("overhead", "uav")}
            s = task_stats(data, cats, models, w["overhead"], w["uav"])
            for m in models:
                for c in cats:
                    if s[m][c] is None:
                        continue
                    boot[m][c]["l2"].append(s[m][c]["l2"]); boot[m][c]["ratio"].append(s[m][c]["ratio"])
                    if "Y" in s[m][c]:
                        boot[m][c]["Y"].append(s[m][c]["Y"])
                    if m != "clip" and s["clip"][c] is not None:
                        boot_ratio_diff[m][c].append(100 * (s[m][c]["ratio"] / s["clip"][c]["ratio"] - 1))
            if cls is not None and b < N_BOOT // 2:  # classification bootstrap: half the replicates (it is the slow part)
                cb = classification(task, data, text, w)
                for m in models:
                    boot_cls[m]["acc_gap"].append(cb[m]["gap"]["accuracy_pp"])
                    for c in cats:
                        boot_cls[m][f"dm_{c}"].append(cb[m]["gap"]["delta_m"][c])
        boot_ci = {m: {c: {k: ci(v) for k, v in boot[m][c].items() if v} for c in cats} for m in models}
        ratio_diff_ci = {m: {c: ci(v) for c, v in d.items() if v} for m, d in boot_ratio_diff.items()}
        cls_ci = {m: {k: ci(v) for k, v in d.items() if v} for m, d in boot_cls.items()} if cls else None
        log(f"{task}: bootstrap done")

        # paper protocol over many seeds
        seeds = []
        for s_i in range(N_SEEDS):
            w = seed_sample(data, cats, rng)
            dr = task_stats(data, cats, models, w["overhead"], w["uav"])
            cl = classification(task, data, text, w) if spec["text_task"] else None
            rec = {"drift": dr, "classification": None if cl is None else
                   {m: {"acc_overhead": cl[m]["overhead"]["accuracy"], "acc_uav": cl[m]["uav"]["accuracy"],
                        "delta_m": cl[m]["gap"]["delta_m"]} for m in models}}
            if task == "P2+P3":
                rec["claims"] = claims_p2p3(dr, cl)
            elif task in ("P1", "P1a"):
                rec["claims"] = claims_p1(dr)
            seeds.append(rec)
        log(f"{task}: {N_SEEDS} seeds done")

        claims_full = claims_p2p3(full, cls) if task == "P2+P3" else (claims_p1(full) if task in ("P1", "P1a") else None)
        results["tasks"][task] = {"n_crops": n, "n_images": n_img, "full": full, "classification_full": cls,
                                  "permutation": perm, "bootstrap_ci": boot_ci, "ratio_change_vs_clip_ci_pct": ratio_diff_ci,
                                  "classification_ci": cls_ci, "seeds": seeds, "claims_full": claims_full}
    results["seconds"] = round(time.time() - t0)
    with open(f"{OUT}/drift_full_scale.json", "w") as f:
        json.dump(results, f, indent=1, default=float)
    write_md(results)
    log(f"done in {results['seconds']} s")


def rng_str(vals, nd=3):
    v = np.asarray(vals)
    return f"{v.mean():.{nd}f} [{v.min():.{nd}f}–{v.max():.{nd}f}]"


def write_md(R):
    L = ["# Full-scale and multi-seed check of the paper's drift and classification results", "",
         f"Cache: up to 5,000 crops per category and domain drawn at random from the full datasets "
         f"(DOTA 1,411 images, VisDrone 6,471, xView 20,881 chips, UAVid train+val). FULL = all cached crops, "
         f"95% image-bootstrap CI ({R['settings']['n_boot']} reps), image-level permutation p ({R['settings']['n_perm']} perms). "
         f"SEEDS = the paper's protocol (300 per category and domain) for {R['settings']['n_seeds']} independent seeds: mean [min–max].",
         "JPEG sources were re-decoded with today's Pillow (libjpeg-turbo), so numbers can differ slightly from a pure re-sampling of the August embeddings.", ""]
    for task, T in R["tasks"].items():
        L += [f"## {task}", "", "Crops (overhead / UAV): " + ", ".join(f"{c} {T['n_crops']['overhead'][c]} / {T['n_crops']['uav'][c]} "
              f"({T['n_images']['overhead'][c]} / {T['n_images']['uav'][c]} images)" for c in T["n_crops"]["overhead"]), ""]
        has_y = any("Y" in (T["full"][m][c] or {}) for m in T["full"] for c in T["full"][m])
        L += ["| Category | Model | Paper L2 / ratio | FULL L2 [CI] | FULL ratio [CI] |" + (" FULL Y_c [CI] |" if has_y else "") + " perm p | SEEDS L2 | SEEDS ratio |" + (" SEEDS Y_c |" if has_y else ""),
              "|---|---|---|---|---|" + ("---|" if has_y else "") + "---|---|---|" + ("---|" if has_y else "")]
        for m, per in T["full"].items():
            for c, v in per.items():
                if v is None:
                    continue
                b = T["bootstrap_ci"][m][c]
                pp = PAPER.get(task if task != "P1a" else "P1", {}).get(m, {}).get(c)
                ps = f"{pp[0]:.4f} / {pp[1]:.4f}" if pp and task != "P1a" else "—"
                sl = [s["drift"][m][c]["l2"] for s in T["seeds"]]
                sr = [s["drift"][m][c]["ratio"] for s in T["seeds"]]
                row = f"| {c} | {MODEL_NAMES[m]} | {ps} | {v['l2']:.4f} [{b['l2'][0]:.3f}, {b['l2'][1]:.3f}] | {v['ratio']:.4f} [{b['ratio'][0]:.3f}, {b['ratio'][1]:.3f}] |"
                if has_y:
                    row += f" {v['Y']:.2f} [{b['Y'][0]:.2f}, {b['Y'][1]:.2f}] |"
                row += f" {T['permutation'][m][c]['p_value']:.4f} | {rng_str(sl)} | {rng_str(sr)} |"
                if has_y:
                    row += f" {rng_str([s['drift'][m][c]['Y'] for s in T['seeds']], 2)} |"
                L.append(row)
        if T["ratio_change_vs_clip_ci_pct"]:
            L += ["", "Paired ratio change vs CLIP, % [95% CI]: " + "; ".join(
                f"{MODEL_NAMES[m]} {c} {100 * (T['full'][m][c]['ratio'] / T['full']['clip'][c]['ratio'] - 1):+.1f} [{v[0]:+.1f}, {v[1]:+.1f}]"
                for m, d in T["ratio_change_vs_clip_ci_pct"].items() for c, v in d.items())]
        if T["classification_full"]:
            L += ["", "| Model | Paper acc O / U (%) | FULL acc O / U (%) | FULL gap [CI] (pp) | SEEDS acc O | SEEDS acc U | FULL Δm_c per category |", "|---|---|---|---|---|---|---|"]
            for m, cl in T["classification_full"].items():
                pa = PAPER_ACC.get(m) if task == "P2+P3" else None
                g = T["classification_ci"][m]["acc_gap"]
                L.append(f"| {MODEL_NAMES[m]} | {f'{pa[0]} / {pa[1]}' if pa else '—'} | {100 * cl['overhead']['accuracy']:.1f} / {100 * cl['uav']['accuracy']:.1f} | "
                         f"{cl['gap']['accuracy_pp']:+.1f} [{g[0]:+.1f}, {g[1]:+.1f}] | "
                         f"{rng_str([100 * s['classification'][m]['acc_overhead'] for s in T['seeds']], 1)} | {rng_str([100 * s['classification'][m]['acc_uav'] for s in T['seeds']], 1)} | "
                         + ", ".join(f"{c} {v:+.4f} [{T['classification_ci'][m]['dm_' + c][0]:+.4f}, {T['classification_ci'][m]['dm_' + c][1]:+.4f}]" for c, v in cl["gap"]["delta_m"].items()) + " |")
        if T.get("claims_full"):
            L += ["", "| Paper claim | Holds at FULL scale | Holds in seeds |", "|---|---|---|"]
            for k, v in T["claims_full"].items():
                L.append(f"| {k} | {'yes' if v else '**no**'} | {sum(s['claims'][k] for s in T['seeds'])}/{len(T['seeds'])} |")
        L.append("")
    with open(f"{OUT}/drift_full_scale.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
