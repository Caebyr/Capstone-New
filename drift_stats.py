"""
Uncertainty for the paper's visual-drift numbers (Tables II-III), computed from the cached embeddings
and the verified crop manifests (build_crop_manifests.py must have been run first).

  1. Point estimates: recomputed and checked against the reported JSON results (must reproduce to
     within one unit of the 4th decimal; the originals were computed in float32, these in float64).
  2. Image-level permutation test per category/model: whole source images are shuffled between the
     two domains (keeping each domain's image count), so correlated crops from one image never split.
     H0: the category's images are exchangeable between domains. Statistic: centroid L2 displacement.
  3. Cluster bootstrap CIs (95%, percentile): images resampled with replacement, independently per
     domain, with the SAME draw applied to every model (matched), so model differences are paired.
     Run twice: image-level (primary) and sequence-level on the UAV side (VisDrone probable video ID,
     UAVid seq ID) as a sensitivity check. Overhead sources have no sequence IDs (xView chips are tiles
     of larger scenes whose scene ID the HF mirror does not expose), so they stay image/chip-level.

Usage (hf env):  python drift_stats.py
Writes results/drift_stats/drift_stats.json and drift_stats_table.md
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from src.crop_sets import MODEL_NAMES, TASKS, load_domain
from src.stats import centroid, ci, cluster_weights, drift, spacing_ratio

N_BOOT = 2000
N_PERM = 2000
SEED = 20261012
OUT_DIR = "results/drift_stats"

REPORTED = {  # (task, json path, model key map) used to check point estimates reproduce
    "P1": ("results/remoteclip_vs_clip/drift_comparison_3way_heldout.json",
           {"clip": "clip_vit_b32", "remoteclip": "remoteclip_vit_l14", "finetuned": "finetuned_clip_visdrone_heldout"}),
    "P2+P3": ("results/xview_vs_visdrone/drift_comparison_4class.json",
              {"clip": "clip_vit_b32", "remoteclip": "remoteclip_vit_l14"}),
}
YARDSTICK = ("results/category_yardstick/yardstick_table.json", "xview_visdrone_uavid",
             {"clip": "CLIP ViT-B/32", "remoteclip": "RemoteCLIP ViT-L/14"})


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def task_stats(data, cats, models, w_over, w_uav):
    """All drift statistics for one task under one pair of weight vectors."""
    out = {}
    for m in models:
        o, u = data[m]["overhead"], data[m]["uav"]
        per_cat, cents, cosd = {}, {"overhead": {}, "uav": {}}, {}
        for c in cats:
            mo, mu_ = o["labels"] == c, u["labels"] == c
            wo, wu = w_over[mo], w_uav[mu_]
            if wo.sum() == 0 or wu.sum() == 0:
                per_cat[c] = None
                continue
            per_cat[c] = drift(o["feats"][mo], wo, u["feats"][mu_], wu)
            cents["overhead"][c] = centroid(o["feats"][mo], wo)
            cents["uav"][c] = centroid(u["feats"][mu_], wu)
            cosd[c] = per_cat[c]["cos"]
        if len(cats) > 2 and all(per_cat[c] is not None for c in cats):
            for c, v in spacing_ratio(cents, cosd, cats).items():
                per_cat[c].update(B=v["B"], Y=v["Y"])
        out[m] = per_cat
    return out


def permutation_test(o, u, cat, rng, n_perm):
    """Image-level permutation p-value for the centroid L2 displacement of one category."""
    fo, fu = o["feats"][o["labels"] == cat], u["feats"][u["labels"] == cat]
    io, iu = o["images"][o["labels"] == cat], u["images"][u["labels"] == cat]
    feats = np.vstack([fo, fu])
    imgs = np.concatenate([io, iu])
    uniq, inv = np.unique(imgs, return_inverse=True)
    n_over_imgs = len(np.unique(io))
    sums = np.zeros((len(uniq), feats.shape[1]))
    np.add.at(sums, inv, feats)
    counts = np.bincount(inv, minlength=len(uniq)).astype(float)
    obs = float(np.linalg.norm(fo.mean(0) - fu.mean(0)))
    null = np.empty(n_perm)
    for i in range(n_perm):
        is_over = np.zeros(len(uniq), bool)
        is_over[rng.permutation(len(uniq))[:n_over_imgs]] = True
        a = sums[is_over].sum(0) / counts[is_over].sum()
        b = sums[~is_over].sum(0) / counts[~is_over].sum()
        null[i] = np.linalg.norm(a - b)
    return {"observed_l2": obs, "p_value": float((1 + (null >= obs).sum()) / (n_perm + 1)),
            "null_l2_q95": float(np.quantile(null, 0.95)), "null_l2_q999": float(np.quantile(null, 0.999)),
            "null_l2_max": float(null.max()), "observed_over_null_max": obs / float(null.max()),
            "n_images_overhead": int(n_over_imgs), "n_images_uav": int(len(uniq) - n_over_imgs)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = {"settings": {"n_boot": N_BOOT, "n_perm": N_PERM, "seed": SEED, "ci": "95% percentile"}, "tasks": {}}
    checks = []

    for task, spec in TASKS.items():
        cats, models = spec["categories"], spec["models"]
        data = {m: {d: load_domain(task, d, m) for d in ("overhead", "uav")} for m in models}
        # sample matching: identical crop IDs, labels and order for every model
        for d in ("overhead", "uav"):
            ref = data[models[0]][d]
            for m in models[1:]:
                same = (np.array_equal(ref["crop_ids"], data[m][d]["crop_ids"])
                        and np.array_equal(ref["labels"], data[m][d]["labels"]))
                checks.append((f"{task} {d}: crop IDs identical {models[0]} vs {m}", same))
        ref_o, ref_u = data[models[0]]["overhead"], data[models[0]]["uav"]

        # 1. point estimates + reproduction check
        point = task_stats(data, cats, models, np.ones(len(ref_o["labels"])), np.ones(len(ref_u["labels"])))
        path, keymap = REPORTED[task]
        reported = json.load(open(path))
        for m in models:
            for c in cats:
                rep = reported[keymap[m]][c]
                for ours, theirs in (("l2", "l2_drift"), ("cos", "cosine_drift"), ("ratio", "drift_ratio")):
                    checks.append((f"{task} {m} {c} {ours} reproduces {rep[theirs]}", abs(point[m][c][ours] - rep[theirs]) <= 1.5e-4))
        if len(cats) > 2:
            ys = json.load(open(YARDSTICK[0]))[YARDSTICK[1]]
            for m in models:
                for c in cats:
                    rep = ys[YARDSTICK[2][m]][c]["ratio"]
                    checks.append((f"{task} {m} {c} Y reproduces {rep}", abs(point[m][c]["Y"] - rep) <= 1.5e-4))

        # 2. image-level permutation tests
        perm = {m: {c: permutation_test(data[m]["overhead"], data[m]["uav"], c, rng, N_PERM) for c in cats} for m in models}
        log(f"{task}: permutation tests done")

        # 3. matched cluster bootstrap, image-level and sequence-level
        boot = {}
        for level, key in (("image", "images"), ("sequence", "sequences")):
            reps = []
            for _ in range(N_BOOT):
                wo, wu = cluster_weights(rng, ref_o["images"]), cluster_weights(rng, ref_u[key])
                reps.append(task_stats(data, cats, models, wo, wu))
            summary = {}
            for m in models:
                summary[m] = {}
                for c in cats:
                    rows = [r[m][c] for r in reps if r[m][c] is not None]
                    s = {k: ci([r[k] for r in rows]) for k in ("l2", "cos", "ratio", "rad") + (("Y",) if len(cats) > 2 else ())}
                    if m != "clip":  # paired change relative to CLIP, same resampled images
                        pairs = [(r[m][c], r["clip"][c]) for r in reps if r[m][c] is not None and r["clip"][c] is not None]
                        for k in ("l2", "cos", "ratio", "rad"):
                            s[f"{k}_pct_vs_clip"] = ci([100 * (a[k] / b[k] - 1) for a, b in pairs])
                        if len(cats) > 2:
                            s["Y_diff_vs_clip"] = ci([a["Y"] - b["Y"] for a, b in pairs])
                    s["n_valid_reps"] = len(rows)
                    summary[m][c] = s
            boot[level] = summary
            log(f"{task}: {level}-level bootstrap done")

        results["tasks"][task] = {
            "categories": cats, "models": models,
            "n_images": {d: int(len(np.unique(data[models[0]][d]["images"]))) for d in ("overhead", "uav")},
            "n_sequences_uav": int(len(np.unique(ref_u["sequences"]))),
            "point": point, "permutation": perm, "bootstrap": boot,
        }

    results["checks"] = [{"check": n, "passed": bool(ok)} for n, ok in checks]
    results["all_checks_passed"] = all(ok for _, ok in checks)
    with open(os.path.join(OUT_DIR, "drift_stats.json"), "w") as f:
        json.dump(results, f, indent=2)
    write_table(results)
    failed = [n for n, ok in checks if not ok]
    log(f"checks: {len(checks) - len(failed)}/{len(checks)} passed" + (f"; FAILED: {failed}" if failed else ""))
    if failed:
        sys.exit(1)


def fmt_ci(v, nd=3):
    return f"[{v[0]:.{nd}f}, {v[1]:.{nd}f}]"


def write_table(results):
    lines = ["# Visual drift with image-level uncertainty", "",
             "Point estimates reproduce Tables II-III. CIs: 95% cluster bootstrap over source images, "
             "matched across models, independent between domains. p: image-level permutation test on L2. "
             "Seq-level CIs resample UAV crops by video sequence instead (sensitivity check).", ""]
    for task, r in results["tasks"].items():
        lines += [f"## {task}  (images: overhead {r['n_images']['overhead']}, UAV {r['n_images']['uav']}; "
                  f"UAV sequences {r['n_sequences_uav']})", "",
                  "| Category | Model | L2 [95% CI] | Ratio [95% CI] | Ratio [seq-level CI] | perm p | L2 / null max |",
                  "|---|---|---|---|---|---|---|"]
        for c in r["categories"]:
            for m in r["models"]:
                p, bi, bs, pm = r["point"][m][c], r["bootstrap"]["image"][m][c], r["bootstrap"]["sequence"][m][c], r["permutation"][m][c]
                lines.append(f"| {c} | {MODEL_NAMES[m]} | {p['l2']:.4f} {fmt_ci(bi['l2'])} | {p['ratio']:.4f} {fmt_ci(bi['ratio'])} | "
                             f"{fmt_ci(bs['ratio'])} | {pm['p_value']:.4f} | {pm['observed_over_null_max']:.1f}x |")
        lines += ["", "Paired change vs CLIP (% , 95% CI, image-level):", "",
                  "| Category | Model | L2 change | Ratio change | Radius change |" + (" Y diff |" if len(r["categories"]) > 2 else ""),
                  "|---|---|---|---|---|" + ("---|" if len(r["categories"]) > 2 else "")]
        for c in r["categories"]:
            for m in r["models"]:
                if m == "clip":
                    continue
                b = r["bootstrap"]["image"][m][c]
                row = (f"| {c} | {MODEL_NAMES[m]} | {fmt_ci(b['l2_pct_vs_clip'], 1)} | {fmt_ci(b['ratio_pct_vs_clip'], 1)} | "
                       f"{fmt_ci(b['rad_pct_vs_clip'], 1)} |")
                if len(r["categories"]) > 2:
                    row += f" {fmt_ci(b['Y_diff_vs_clip'], 2)} |"
                lines.append(row)
        if len(r["categories"]) > 2:
            lines += ["", "Spacing-normalized drift Y_c (Table III) with image-level CI:", "",
                      "| Category | " + " | ".join(MODEL_NAMES[m] for m in r["models"]) + " |",
                      "|---|" + "---|" * len(r["models"])]
            for c in r["categories"]:
                lines.append(f"| {c} | " + " | ".join(f"{r['point'][m][c]['Y']:.2f} {fmt_ci(r['bootstrap']['image'][m][c]['Y'], 2)}"
                                                       for m in r["models"]) + " |")
        lines.append("")
    with open(os.path.join(OUT_DIR, "drift_stats_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
