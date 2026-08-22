"""
PCA / t-SNE visualizations for the three *ground-truth* drift comparisons, replacing
Phase-1's flawed whole-image / zero-shot-label version with plots built on the same
crops and labels actually used for the reported drift numbers.

Unlike Phase-1 (notebooks/01_phase1_reproduction.ipynb), every point here has a real,
correct class label straight from the dataset annotations -- nothing is zero-shot-guessed.
Uses embeddings already cached under embeddings/ by the run_*.py scripts, so this only
loads .npy files and plots; no GPU / re-encoding needed.

Produces, per comparison, one PCA figure and one t-SNE figure, each with two panels
side by side (CLIP baseline | RemoteCLIP), so the reader can see directly whether the
domain split visibly tightens when you move from the generic model to the
remote-sensing-specialized one -- the geometric picture behind the drift-score bar charts.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = os.path.dirname(__file__)
E_RC = os.path.join(ROOT, "embeddings", "remoteclip_vs_clip")
E_XV = os.path.join(ROOT, "embeddings", "xview_vs_visdrone")
OUT = os.path.join(ROOT, "results", "pca_tsne")
os.makedirs(OUT, exist_ok=True)

RNG_SEED = 42
MAX_POINTS_PER_SIDE = 900  # subsample for t-SNE speed/legibility if a side is larger


def load(path):
    return np.load(path, allow_pickle=True)


def filter_classes(feats, labels, keep):
    mask = np.isin(labels, keep)
    return feats[mask], labels[mask]


def subsample(feats, labels, rng, max_n):
    if len(labels) <= max_n:
        return feats, labels
    idx = rng.choice(len(labels), size=max_n, replace=False)
    return feats[idx], labels[idx]


def make_panel(ax, feats, labels, domains, class_order, method, title):
    cmap = plt.colormaps.get_cmap("tab10")
    class_to_color = {c: cmap(i / max(len(class_order), 1)) for i, c in enumerate(class_order)}

    if method == "pca":
        reducer = PCA(n_components=2, random_state=RNG_SEED)
    else:
        perplexity = min(30, max(5, (len(labels) - 1) // 3))
        reducer = TSNE(n_components=2, perplexity=perplexity, learning_rate=200,
                        init="pca", random_state=RNG_SEED)
    coords = reducer.fit_transform(feats)

    for cls in class_order:
        for dom, marker in (("sat", "o"), ("uav", "x")):
            sel = (labels == cls) & (domains == dom)
            if not np.any(sel):
                continue
            ax.scatter(coords[sel, 0], coords[sel, 1], color=class_to_color[cls],
                       marker=marker, alpha=0.65, s=22, linewidths=0.8,
                       label=f"{cls} ({'sat' if dom=='sat' else 'uav'})")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    return class_to_color


def class_legend(fig, class_to_color, class_order):
    class_handles = [plt.Line2D([], [], marker="o", linestyle="", color=class_to_color[c], label=c)
                      for c in class_order]
    domain_handles = [
        plt.Line2D([], [], marker="o", linestyle="", color="black", label="Satellite / xView"),
        plt.Line2D([], [], marker="x", linestyle="", color="black", label="UAV (VisDrone/UAVid)"),
    ]
    leg1 = fig.legend(handles=class_handles, title="Class", loc="upper center",
                       bbox_to_anchor=(0.5, 0.02), ncol=min(len(class_order), 6), fontsize=9)
    fig.add_artist(leg1)
    fig.legend(handles=domain_handles, title="Domain", loc="upper center",
               bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9)


def make_figure(comparison_name, sat_clip, uav_clip, sat_rc, uav_rc, class_order, method, suptitle):
    rng = np.random.default_rng(RNG_SEED)

    def build_side(sat_feats, sat_labels, uav_feats, uav_labels):
        sat_feats, sat_labels = subsample(sat_feats, sat_labels, rng, MAX_POINTS_PER_SIDE)
        uav_feats, uav_labels = subsample(uav_feats, uav_labels, rng, MAX_POINTS_PER_SIDE)
        feats = np.vstack([sat_feats, uav_feats])
        labels = np.concatenate([sat_labels, uav_labels])
        domains = np.array(["sat"] * len(sat_labels) + ["uav"] * len(uav_labels))
        return feats, labels, domains

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    feats, labels, domains = build_side(*sat_clip, *uav_clip)
    make_panel(axes[0], feats, labels, domains, class_order, method, "CLIP ViT-B/32 (generic)")

    feats, labels, domains = build_side(*sat_rc, *uav_rc)
    class_to_color = make_panel(axes[1], feats, labels, domains, class_order, method,
                                 "RemoteCLIP ViT-L/14 (satellite-specialized)")

    fig.suptitle(suptitle, fontsize=13, y=1.02)
    class_legend(fig, class_to_color, class_order)
    fig.tight_layout()
    out_path = os.path.join(OUT, f"{method}_{comparison_name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)


# ---------------------------------------------------------------------------
# 1. DOTA vs. VisDrone (2 classes: small-vehicle, large-vehicle)
# ---------------------------------------------------------------------------
dota_clip_feats, dota_clip_labels = load(os.path.join(E_RC, "dota_clip_feats.npy")), load(os.path.join(E_RC, "dota_clip_labels.npy"))
vis_clip_feats, vis_clip_labels = load(os.path.join(E_RC, "visdrone_clip_feats.npy")), load(os.path.join(E_RC, "visdrone_clip_labels.npy"))
dota_rc_feats, dota_rc_labels = load(os.path.join(E_RC, "dota_remoteclip_feats.npy")), load(os.path.join(E_RC, "dota_remoteclip_labels.npy"))
vis_rc_feats, vis_rc_labels = load(os.path.join(E_RC, "visdrone_remoteclip_feats.npy")), load(os.path.join(E_RC, "visdrone_remoteclip_labels.npy"))

classes_2 = ["small-vehicle", "large-vehicle"]
for method in ("pca", "tsne"):
    make_figure(
        "dota_vs_visdrone",
        (dota_clip_feats, dota_clip_labels), (vis_clip_feats, vis_clip_labels),
        (dota_rc_feats, dota_rc_labels), (vis_rc_feats, vis_rc_labels),
        classes_2, method,
        "DOTA vs. VisDrone -- ground-truth crops, ground-truth labels (2 shared classes)",
    )

# ---------------------------------------------------------------------------
# 2. xView vs. VisDrone (3 classes: car, bus, truck -- van excluded, too few xView samples)
# ---------------------------------------------------------------------------
xv_clip_feats, xv_clip_labels = load(os.path.join(E_XV, "xview_clip_feats.npy")), load(os.path.join(E_XV, "xview_clip_labels.npy"))
vd_clip_feats, vd_clip_labels = load(os.path.join(E_XV, "visdrone_clip_feats.npy")), load(os.path.join(E_XV, "visdrone_clip_labels.npy"))
xv_rc_feats, xv_rc_labels = load(os.path.join(E_XV, "xview_remoteclip_feats.npy")), load(os.path.join(E_XV, "xview_remoteclip_labels.npy"))
vd_rc_feats, vd_rc_labels = load(os.path.join(E_XV, "visdrone_remoteclip_feats.npy")), load(os.path.join(E_XV, "visdrone_remoteclip_labels.npy"))

classes_3 = ["car", "bus", "truck"]
xv_clip_feats3, xv_clip_labels3 = filter_classes(xv_clip_feats, xv_clip_labels, classes_3)
vd_clip_feats3, vd_clip_labels3 = filter_classes(vd_clip_feats, vd_clip_labels, classes_3)
xv_rc_feats3, xv_rc_labels3 = filter_classes(xv_rc_feats, xv_rc_labels, classes_3)
vd_rc_feats3, vd_rc_labels3 = filter_classes(vd_rc_feats, vd_rc_labels, classes_3)

for method in ("pca", "tsne"):
    make_figure(
        "xview_vs_visdrone",
        (xv_clip_feats3, xv_clip_labels3), (vd_clip_feats3, vd_clip_labels3),
        (xv_rc_feats3, xv_rc_labels3), (vd_rc_feats3, vd_rc_labels3),
        classes_3, method,
        "xView vs. VisDrone -- ground-truth crops, ground-truth labels (3 shared classes)",
    )

# ---------------------------------------------------------------------------
# 3. xView+UAVid, 4 classes: car, bus, truck, building (building: xView sat side, UAVid uav side)
# ---------------------------------------------------------------------------
xvb_clip_feats = load(os.path.join(E_XV, "xview_building_clip_feats.npy"))
uab_clip_feats = load(os.path.join(E_XV, "uavid_building_clip_feats.npy"))
xvb_rc_feats = load(os.path.join(E_XV, "xview_building_remoteclip_feats.npy"))
uab_rc_feats = load(os.path.join(E_XV, "uavid_building_remoteclip_feats.npy"))

xvb_clip_labels = np.array(["building"] * len(xvb_clip_feats))
uab_clip_labels = np.array(["building"] * len(uab_clip_feats))
xvb_rc_labels = np.array(["building"] * len(xvb_rc_feats))
uab_rc_labels = np.array(["building"] * len(uab_rc_feats))

classes_4 = ["car", "bus", "truck", "building"]
sat_clip4_feats = np.vstack([xv_clip_feats3, xvb_clip_feats])
sat_clip4_labels = np.concatenate([xv_clip_labels3, xvb_clip_labels])
uav_clip4_feats = np.vstack([vd_clip_feats3, uab_clip_feats])
uav_clip4_labels = np.concatenate([vd_clip_labels3, uab_clip_labels])
sat_rc4_feats = np.vstack([xv_rc_feats3, xvb_rc_feats])
sat_rc4_labels = np.concatenate([xv_rc_labels3, xvb_rc_labels])
uav_rc4_feats = np.vstack([vd_rc_feats3, uab_rc_feats])
uav_rc4_labels = np.concatenate([vd_rc_labels3, uab_rc_labels])

for method in ("pca", "tsne"):
    make_figure(
        "xview_visdrone_uavid",
        (sat_clip4_feats, sat_clip4_labels), (uav_clip4_feats, uav_clip4_labels),
        (sat_rc4_feats, sat_rc4_labels), (uav_rc4_feats, uav_rc4_labels),
        classes_4, method,
        "xView vs. VisDrone+UAVid -- ground-truth crops, ground-truth labels (4 shared classes)",
    )

print("done")
