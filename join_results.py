"""
Joins visual drift (drift_stats.py) with crop classification and text alignment (classify_crops.py)
into one per-category table, and asks -- descriptively -- whether categories with larger drift also
show larger performance gaps or margin changes.

Only 2 (P1) or 4 (P2+P3) categories per task, so rank correlations here are exploratory: with n=4
even a perfect ordering is not statistically meaningful, and no p-values are reported.

Usage (hf env):  python join_results.py    (after drift_stats.py and classify_crops.py --bootstrap)
Writes results/joined/joined_table.md and joined_table.csv
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from src.crop_sets import MODEL_NAMES

OUT_DIR = "results/joined"


def fmt(v, ci=None, nd=3, signed=False):
    s = f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"
    if ci is not None:
        s += f" [{ci[0]:+.{nd}f}, {ci[1]:+.{nd}f}]" if signed else f" [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"
    return s


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    drift = json.load(open("results/drift_stats/drift_stats.json"))["tasks"]
    cls = json.load(open("results/classification/classification_bootstrap.json"))["tasks"]

    rows = []
    for task in ("P1", "P2+P3"):
        d, c = drift[task], cls[task]
        for cat in d["categories"]:
            for m in d["models"]:
                dp, db, perm = d["point"][m][cat], d["bootstrap"]["image"][m][cat], d["permutation"][m][cat]
                cp, cb = c["point"][m], c["bootstrap"]["image"][m]
                rows.append({
                    "task": task, "category": cat, "model": m,
                    "l2": dp["l2"], "l2_ci": db["l2"], "cos": dp["cos"], "cos_ci": db["cos"],
                    "r_overhead": dp["r_overhead"], "r_uav": dp["r_uav"], "ratio": dp["ratio"], "ratio_ci": db["ratio"],
                    "Y": dp.get("Y"), "Y_ci": db.get("Y"), "perm_p": perm["p_value"],
                    "recall_overhead": cp["overhead"]["per_category"][cat]["recall"], "recall_uav": cp["uav"]["per_category"][cat]["recall"],
                    "recall_gap_pp": cp["gap"]["recall_pp"][cat], "recall_gap_ci": cb[cat]["recall_gap_pp"],
                    "a_overhead": cp["overhead"]["text"][cat]["a"], "a_uav": cp["uav"]["text"][cat]["a"],
                    "A": cp["gap"]["A"][cat], "A_ci": cb[cat]["A"],
                    "m_overhead": cp["overhead"]["text"][cat]["margin"], "m_uav": cp["uav"]["text"][cat]["margin"],
                    "delta_m": cp["gap"]["delta_m"][cat], "delta_m_ci": cb[cat]["delta_m"],
                })

    with open(os.path.join(OUT_DIR, "joined_table.csv"), "w", newline="", encoding="utf-8") as f:
        flat = [{k: (";".join(f"{x:.6f}" for x in v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows]
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)

    L = ["# Visual drift vs crop classification vs text alignment (joined, exploratory)", "",
         "All from the same crops (300 per category and domain, seed 42). Brackets: 95% image-level bootstrap CI.",
         "Recall gap = overhead - UAV in percentage points (positive = worse on UAV). delta_m = margin(overhead) - margin(UAV) "
         "(positive = smaller UAV margin). Fine-tuned CLIP: supervised on the UAV side of P1, not zero-shot.", ""]
    for task in ("P1", "P2+P3"):
        tr = [r for r in rows if r["task"] == task]
        has_y = tr[0]["Y"] is not None
        L += [f"## {task}", "",
              "| Category | Model | L2 drift | Cos drift | Ratio |" + (" Y_c |" if has_y else "") +
              " Recall overhead → UAV | Recall gap (pp) | A_c | Margin overhead → UAV | Δm_c |",
              "|---|---|---|---|---|" + ("---|" if has_y else "") + "---|---|---|---|---|"]
        for r in tr:
            L.append(f"| {r['category']} | {MODEL_NAMES[r['model']]} | {fmt(r['l2'], r['l2_ci'])} | {fmt(r['cos'], nd=4)} | {fmt(r['ratio'], r['ratio_ci'])} |"
                     + (f" {fmt(r['Y'], r['Y_ci'], nd=2)} |" if has_y else "")
                     + f" {100*r['recall_overhead']:.1f}% → {100*r['recall_uav']:.1f}% | {fmt(r['recall_gap_pp'], r['recall_gap_ci'], nd=1, signed=True)} |"
                     f" {fmt(r['A'], nd=4)} | {r['m_overhead']:+.4f} → {r['m_uav']:+.4f} | {fmt(r['delta_m'], r['delta_m_ci'], nd=4, signed=True)} |")
        L.append("")

    # Exploratory association: within each task and model, across categories.
    L += ["## Does larger drift go with a larger gap? (exploratory, within task and model, across categories)", "",
          "Spearman rank correlation across the task's categories. P1 has 2 categories (no correlation possible); "
          "P2+P3 has 4, where |rho| = 1 occurs by chance with probability 1/12 and nothing here is a statistical test.", "",
          "| Task | Model | drift measure | vs abs(recall gap) | vs A_c | vs abs(Δm_c) |", "|---|---|---|---|---|---|"]
    assoc = []
    for task in ("P2+P3",):
        for m in ("clip", "remoteclip"):
            tr = [r for r in rows if r["task"] == task and r["model"] == m]
            for measure in ("cos", "ratio", "Y"):
                x = [r[measure] for r in tr]
                vals = []
                for target in ("recall_gap_pp", "A", "delta_m"):
                    y = [abs(r[target]) for r in tr]
                    rho = spearmanr(x, y).correlation
                    vals.append(rho)
                    assoc.append({"task": task, "model": m, "drift": measure, "target": target, "rho": float(rho)})
                L.append(f"| {task} | {MODEL_NAMES[m]} | {measure} | " + " | ".join(f"{v:+.2f}" for v in vals) + " |")
    # Pooled across both encoders (8 points), still exploratory.
    tr = [r for r in rows if r["task"] == "P2+P3"]
    pooled = {t: float(spearmanr([r["cos"] for r in tr], [abs(r[t]) for r in tr]).correlation) for t in ("recall_gap_pp", "A", "delta_m")}
    L += ["", f"Pooled over both encoders (8 category-encoder points), cosine drift vs abs(recall gap) rho = {pooled['recall_gap_pp']:+.2f}, "
          f"vs A_c rho = {pooled['A']:+.2f}, vs abs(Δm_c) rho = {pooled['delta_m']:+.2f}.", ""]

    with open(os.path.join(OUT_DIR, "joined_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    with open(os.path.join(OUT_DIR, "association.json"), "w") as f:
        json.dump({"per_model": assoc, "pooled_cos": pooled}, f, indent=2)
    print(f"wrote {OUT_DIR}/joined_table.md, joined_table.csv, association.json")


if __name__ == "__main__":
    main()
