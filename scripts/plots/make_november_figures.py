"""
Presentation figures for the November 2026 results (results/NEW_EXPERIMENTS_2026-11.md).

  python scripts/plots/make_november_figures.py

Reads only committed result JSONs; writes PNGs (and the plotted numbers as CSV) to results/figures_2026-11/.
  fig1_detection_ap50.png        AP50 overhead vs UAV per category, OWLv2 and Grounding DINO        (B2)
  fig2_same_domain_baseline.png  CLIP cosine drift at three levels of difference, 95% CIs            (A4)
  fig3_resolution_control.png    cosine drift before / after shrinking UAV crops to overhead size    (A5)
  fig4_sam_found_vs_proposed.png SAM proposal coverage vs objects found after CLIP/RemoteCLIP naming (B4)
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "results", "figures_2026-11")
# Reference categorical palette, slots 1-3 in fixed order; light surface; text tokens.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CATS = ["small-vehicle", "large-vehicle", "car", "bus", "truck", "building"]
PAIRS = {"small-vehicle": ("dota_tiles", "visdrone_ho_D1"), "large-vehicle": ("dota_tiles", "visdrone_ho_D1"),
         "car": ("xview_val", "visdrone_ho_D2"), "bus": ("xview_val", "visdrone_ho_D2"),
         "truck": ("xview_val", "visdrone_ho_D2"), "building": ("xview_val", "uavid_tv")}

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False})


def load(path):
    return json.load(open(os.path.join(ROOT, path), encoding="utf-8"))


def style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name, rows, header):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    with open(os.path.join(OUT, name + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fig1():
    det = load("results/detection/detection_results.json")["detectors"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    rows, x, w = [], np.arange(len(CATS)), 0.38
    for ax, (d, title) in zip(axes, (("owlv2", "OWLv2 B/16"), ("gdino", "Grounding DINO tiny"))):
        P = det[d]["per_set"]
        o = [100 * P[PAIRS[c][0]]["class_aware"][c]["AP50"] for c in CATS]
        u = [100 * P[PAIRS[c][1]]["class_aware"][c]["AP50"] for c in CATS]
        ax.bar(x - w / 2, o, w, color=BLUE, edgecolor=SURFACE, linewidth=1.5, label="Overhead (DOTA / xView)")
        ax.bar(x + w / 2, u, w, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, label="UAV (VisDrone / UAVid)")
        for xi, (a, b) in enumerate(zip(o, u)):
            ax.text(xi - w / 2, a + 1, f"{a:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
            ax.text(xi + w / 2, b + 1, f"{b:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
            rows.append([d, CATS[xi], round(a, 1), round(b, 1)])
        ax.set_xticks(x, [c.replace("-", "\n") for c in CATS])
        ax.set_title(title, loc="left", fontsize=11)
        style(ax)
    axes[0].set_ylabel("AP50 (%)")
    axes[0].set_ylim(0, 88)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Zero-shot detection is 2-10x worse on overhead images", x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, "fig1_detection_ap50", rows, ["detector", "category", "AP50_overhead", "AP50_uav"])


def fig2():
    res = load("results/full_scale/same_domain_baseline.json")["comparisons"]
    colour = {"L0": AQUA, "L1": BLUE, "L2": ORANGE}
    name = {"L0": "Same dataset, different images", "L1": "Same platform, different dataset", "L2": "Overhead vs UAV"}
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    rows, seen = [], set()
    for i, r in enumerate(res[::-1]):
        v = r["models"]["clip"]
        p, (lo, hi) = v["point"]["cos"], v["ci"]["cos"]
        lab = name[r["level"]] if r["level"] not in seen else None
        seen.add(r["level"])
        ax.plot([lo, hi], [i, i], color=colour[r["level"]], linewidth=2, solid_capstyle="round")
        ax.scatter([p], [i], s=55, color=colour[r["level"]], edgecolor=SURFACE, linewidth=2, zorder=3, label=lab)
        rows.append([r["level"], r["comparison"], r["category"], round(p, 4), round(lo, 4), round(hi, 4)])
    ax.set_yticks(range(len(res)), [f"{r['comparison']}: {r['category']}" for r in res[::-1]], fontsize=8.5)
    ax.set_xlabel("CLIP ViT-B/32 cosine drift between centroids (95% image-bootstrap CI)")
    style(ax, "x")
    h, l = ax.get_legend_handles_labels()
    order = [l.index(name[k]) for k in ("L2", "L1", "L0")]
    ax.legend([h[k] for k in order], [l[k] for k in order], frameon=False, loc="upper right")
    ax.set_title("xView-vs-UAV drift is 2.5-5x the same-platform level; DOTA-vs-VisDrone is not", loc="left", fontsize=11, fontweight="bold")
    save(fig, "fig2_same_domain_baseline", rows[::-1], ["level", "comparison", "category", "cos", "ci_lo", "ci_hi"])


def fig3():
    A = load("results/full_scale/controls_a5.json")
    res = A["resolution"]["P2+P3"]
    cats = ["car", "bus", "truck", "building"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True, sharex=True)
    rows = []
    for ax, (m, title) in zip(axes, (("clip", "CLIP ViT-B/32"), ("remoteclip", "RemoteCLIP ViT-L/14"))):
        before = [res[m]["drift"][c]["overhead_vs_uav"]["cos"] for c in cats]
        after = [res[m]["drift"][c]["overhead_vs_uav_shrunk"]["cos"] for c in cats]
        y = np.arange(len(cats))[::-1]
        for yi, b, a in zip(y, before, after):
            ax.plot([a, b], [yi, yi], color=GRID, linewidth=2, zorder=1)
        ax.scatter(before, y, s=60, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=3, label="UAV crops as they are")
        ax.scatter(after, y, s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3, label="UAV crops shrunk to overhead size")
        for c, b, a in zip(cats, before, after):
            rows.append([m, c, round(b, 4), round(a, 4)])
        ax.set_yticks(y, cats)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("cosine drift, xView vs UAV (VisDrone; UAVid for building)", fontsize=9)
        ax.set_xlim(0, None)
        style(ax, "x")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Matching object size removes about two-thirds of the xView-vs-UAV drift", x=0.01, ha="left", fontsize=12, fontweight="bold")
    save(fig, "fig3_resolution_control", rows, ["model", "category", "cos_original", "cos_shrunk"])


def fig4():
    R = load("results/detection/sam_pipeline_results.json")
    sets = [("dota_tiles", "DOTA\n(overhead)"), ("xview_val", "xView\n(overhead)"), ("visdrone_ho_D1", "VisDrone D1\n(UAV)"),
            ("visdrone_ho_D2", "VisDrone D2\n(UAV)"), ("uavid_tv", "UAVid\n(UAV)")]
    rec = [100 * R["proposal_recall"][s]["_overall"] for s, _ in sets]
    fc = [100 * R["sets"][s]["sam_clip"]["naming"]["overall_found_rate"] for s, _ in sets]
    fr = [100 * R["sets"][s]["sam_remoteclip"]["naming"]["overall_found_rate"] for s, _ in sets]
    x, w = np.arange(len(sets)), 0.26
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for off, vals, col, lab in ((-w, rec, AQUA, "SAM proposals cover the object (IoU >= 0.5)"),
                                (0, fc, BLUE, "Still found after CLIP naming"),
                                (w, fr, ORANGE, "Still found after RemoteCLIP naming")):
        ax.bar(x + off, vals, w, color=col, edgecolor=SURFACE, linewidth=1.5, label=lab)
        for xi, v in zip(x, vals):
            ax.text(xi + off, v + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, [s[1] for s in sets])
    ax.set_ylabel("share of GT objects (%)")
    ax.set_ylim(0, 75)
    style(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5, ncol=1)
    ax.set_title("SAM finds the regions everywhere; naming discards them overhead", loc="left", fontsize=11, fontweight="bold")
    save(fig, "fig4_sam_found_vs_proposed", [[s, round(a, 1), round(b, 1), round(c, 1)] for (s, _), a, b, c in zip(sets, rec, fc, fr)],
         ["set", "proposal_recall", "found_sam_clip", "found_sam_remoteclip"])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4):
        f()
        print("done", f.__name__)
