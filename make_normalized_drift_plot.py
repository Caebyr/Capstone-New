"""
The raw L2-vs-cosine bar charts (three_way_drift_comparison_heldout.png,
four_class_drift_comparison.png) plot both metrics on one shared, literal 0-based axis.
Because CLIP embeddings are unit-normalized, L2 distance and cosine distance are related by
roughly L2 ~ sqrt(2 * cosine_distance) -- so cosine numbers are always numerically smaller than
L2 numbers, regardless of effect size. On a shared axis this visually buries the cosine bars,
which can make a non-technical viewer conclude cosine "barely shows an effect" -- when in fact
cosine distance shows a *larger* percentage reduction than L2 in every case measured here.

This plots both metrics as "% of the CLIP baseline" instead of raw units, so L2 and cosine sit on
one common, honest scale: shorter bar = less drift, and the two metrics can be compared directly
without unit confusion. Reads already-saved JSON results; nothing is recomputed.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(__file__)
OUT_DIR = os.path.join(ROOT, "results", "category_yardstick")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_LABELS = {
    "remoteclip_vit_l14": "RemoteCLIP\n(satellite-specialized)",
    "finetuned_clip_visdrone_heldout": "Fine-tuned CLIP\n(UAV-adapted)",
}
MODEL_COLORS = {
    "remoteclip_vit_l14": "#D79B00",
    "finetuned_clip_visdrone_heldout": "#82B366",
}


def pct_of_baseline(data, model, cls, metric):
    return 100 * data[model][cls][metric] / data["clip_vit_b32"][cls][metric]


d1 = json.load(open(os.path.join(ROOT, "results", "remoteclip_vs_clip", "drift_comparison_3way_heldout.json")))
d2 = json.load(open(os.path.join(ROOT, "results", "xview_vs_visdrone", "drift_comparison_4class.json")))

panels = [
    ("DOTA vs. VisDrone", d1, ["small-vehicle", "large-vehicle"],
     ["remoteclip_vit_l14", "finetuned_clip_visdrone_heldout"]),
    ("xView vs. VisDrone/UAVid", d2, ["car", "bus", "truck", "building"],
     ["remoteclip_vit_l14"]),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 4]})

for ax, (title, data, classes, models) in zip(axes, panels):
    metrics = ["l2_drift", "cosine_drift"]
    metric_labels = ["L2", "Cosine"]
    n_bars = len(models) * len(metrics)
    width = 0.8 / n_bars
    x = np.arange(len(classes))

    ax.axhline(100, color="#333333", linewidth=1.2, label="CLIP baseline (100%)")

    bar_i = 0
    for model in models:
        for metric, mlabel, hatch in zip(metrics, metric_labels, ["", "//"]):
            vals = [pct_of_baseline(data, model, c, metric) for c in classes]
            offset = (bar_i - (n_bars - 1) / 2) * width
            label = f"{MODEL_LABELS[model].splitlines()[0]} -- {mlabel}" if len(models) > 1 else mlabel
            ax.bar(x + offset, vals, width, color=MODEL_COLORS[model], hatch=hatch,
                   edgecolor="white", label=label)
            bar_i += 1

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15)
    ax.set_ylabel("Drift, as % of CLIP baseline\n(lower = less drift)")
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="lower left")

fig.suptitle("Drift reduction on a common scale: L2 and cosine both shown as % of the CLIP baseline",
             fontsize=13)
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "normalized_drift_reduction.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved", out_path)

# Print the summary table too, useful for a report table.
print()
print(f"{'comparison':<28}{'class':<15}{'metric':<8}{'clip':>8}{'other':>10}{'% of base':>12}{'% reduction':>14}")
for title, data, classes, models in panels:
    for cls in classes:
        for model in models:
            for metric, mlabel in zip(["l2_drift", "cosine_drift"], ["L2", "Cosine"]):
                base = data["clip_vit_b32"][cls][metric]
                other = data[model][cls][metric]
                pct = 100 * other / base
                print(f"{title:<28}{cls:<15}{mlabel:<8}{base:>8.4f}{other:>10.4f}{pct:>11.1f}%{100-pct:>13.1f}%")
