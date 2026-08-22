"""
Visualizes drift_ratio (already computed and saved by run_remoteclip_vs_clip.py /
run_uavid_building.py): L2 drift divided by the average intra-domain spread of that class
(l2_drift / ((sat_intra + uav_intra) / 2)). This answers "is the cross-altitude shift large
relative to how spread out a class naturally is within one domain?" -- a ratio near/above 1
means the cross-domain gap is comparable to or bigger than a class's own natural variability;
a ratio near 0 would mean the gap is negligible next to normal in-domain noise.

Reads the already-saved JSON results directly -- nothing is recomputed here.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(__file__)
OUT_DIR = os.path.join(ROOT, "results", "category_yardstick")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_LABELS = {
    "clip_vit_b32": "CLIP ViT-B/32",
    "remoteclip_vit_l14": "RemoteCLIP ViT-L/14",
    "finetuned_clip_visdrone_heldout": "Fine-tuned CLIP (UAV-adapted)",
}
MODEL_COLORS = {
    "clip_vit_b32": "#6C8EBF",
    "remoteclip_vit_l14": "#D79B00",
    "finetuned_clip_visdrone_heldout": "#82B366",
}

d1 = json.load(open(os.path.join(ROOT, "results", "remoteclip_vs_clip", "drift_comparison_3way_heldout.json")))
d2 = json.load(open(os.path.join(ROOT, "results", "xview_vs_visdrone", "drift_comparison_4class.json")))

panels = [
    ("DOTA vs. VisDrone", d1, ["small-vehicle", "large-vehicle"],
     ["clip_vit_b32", "remoteclip_vit_l14", "finetuned_clip_visdrone_heldout"]),
    ("xView vs. VisDrone/UAVid", d2, ["car", "bus", "truck", "building"],
     ["clip_vit_b32", "remoteclip_vit_l14"]),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [2, 4]})

for ax, (title, data, classes, models) in zip(axes, panels):
    n_models = len(models)
    width = 0.8 / n_models
    x = np.arange(len(classes))
    for i, model in enumerate(models):
        ratios = [data[model][c]["drift_ratio"] for c in classes]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, ratios, width, label=MODEL_LABELS[model], color=MODEL_COLORS[model])
    ax.axhline(1.0, color="#666666", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(classes) - 0.5, 1.02, "= as large as natural in-domain spread", fontsize=8, color="#666666", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Drift ratio\n(cross-domain L2 / avg. intra-domain spread)")
    ax.legend(fontsize=9)

fig.suptitle("Drift Ratio: cross-altitude shift relative to each class's natural in-domain spread", fontsize=13)
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "drift_ratio_comparison.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved", out_path)
