"""
Answers "is the cross-altitude cosine drift actually meaningful, given how small it looks in
absolute terms?" by comparing it against a natural yardstick: the cosine distance between two
genuinely DIFFERENT categories within the same domain (e.g. small-vehicle vs. large-vehicle,
both from DOTA). CLIP embeddings are unit-normalized, so cosine distance is the space's native
notion of "how different two things are" -- comparing the cross-altitude, same-category distance
against the same-altitude, different-category distance tells us whether crossing altitude moves
an object's embedding by roughly as much as swapping it for a different kind of object.

For each of the 3 comparisons, for each of CLIP/RemoteCLIP:
  cross_altitude[c]   = cosine_dist(centroid_domainA[c], centroid_domainB[c])           (drift, as reported)
  cross_category[c]   = mean over c' != c of cosine_dist to c' within domain A and domain B
  ratio[c]            = cross_altitude[c] / cross_category[c]   (~1 => altitude-crossing is as
                                                                    big a jump as a real category change)

Uses the same cached embeddings as make_pca_tsne_plots.py -- no re-encoding needed.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(__file__)
E_RC = os.path.join(ROOT, "embeddings", "remoteclip_vs_clip")
E_XV = os.path.join(ROOT, "embeddings", "xview_vs_visdrone")
OUT_DIR = os.path.join(ROOT, "results", "category_yardstick")
os.makedirs(OUT_DIR, exist_ok=True)


def load(path):
    return np.load(path, allow_pickle=True)


def cosine_dist(a, b):
    return 1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


def centroids(feats, labels, classes):
    return {c: feats[labels == c].mean(axis=0) for c in classes}


def compute_table(sat_feats, sat_labels, uav_feats, uav_labels, classes):
    sat_c = centroids(sat_feats, sat_labels, classes)
    uav_c = centroids(uav_feats, uav_labels, classes)
    rows = {}
    for c in classes:
        cross_altitude = cosine_dist(sat_c[c], uav_c[c])
        others = [c2 for c2 in classes if c2 != c]
        same_domain_dists = []
        for c2 in others:
            same_domain_dists.append(cosine_dist(sat_c[c], sat_c[c2]))
            same_domain_dists.append(cosine_dist(uav_c[c], uav_c[c2]))
        cross_category = float(np.mean(same_domain_dists)) if same_domain_dists else float("nan")
        rows[c] = {
            "cross_altitude_cosine": round(float(cross_altitude), 4),
            "cross_category_cosine": round(cross_category, 4),
            "ratio": round(float(cross_altitude) / cross_category, 4) if cross_category else float("nan"),
        }
    return rows


def plot_comparison(name, model_tables, classes, title):
    models = list(model_tables.keys())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    x = np.arange(len(classes))
    width = 0.35
    for ax, model in zip(axes, models):
        rows = model_tables[model]
        cross_alt = [rows[c]["cross_altitude_cosine"] for c in classes]
        cross_cat = [rows[c]["cross_category_cosine"] for c in classes]
        ax.bar(x - width / 2, cross_alt, width, label="Cross-altitude\n(same category)", color="#B85450")
        ax.bar(x + width / 2, cross_cat, width, label="Cross-category\n(same altitude, avg.)", color="#6C8EBF")
        for i, c in enumerate(classes):
            ax.text(i, max(cross_alt[i], cross_cat[i]) + 0.001, f"{rows[c]['ratio']:.2f}×",
                    ha="center", fontsize=9, color="#333333")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=20)
        ax.set_title(model, fontsize=12)
        ax.set_ylabel("Cosine distance")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"yardstick_{name}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)


all_results = {}

# 1. DOTA vs VisDrone
dota_clip_feats, dota_clip_labels = load(os.path.join(E_RC, "dota_clip_feats.npy")), load(os.path.join(E_RC, "dota_clip_labels.npy"))
vis_clip_feats, vis_clip_labels = load(os.path.join(E_RC, "visdrone_clip_feats.npy")), load(os.path.join(E_RC, "visdrone_clip_labels.npy"))
dota_rc_feats, dota_rc_labels = load(os.path.join(E_RC, "dota_remoteclip_feats.npy")), load(os.path.join(E_RC, "dota_remoteclip_labels.npy"))
vis_rc_feats, vis_rc_labels = load(os.path.join(E_RC, "visdrone_remoteclip_feats.npy")), load(os.path.join(E_RC, "visdrone_remoteclip_labels.npy"))

classes_2 = ["small-vehicle", "large-vehicle"]
tables_1 = {
    "CLIP ViT-B/32": compute_table(dota_clip_feats, dota_clip_labels, vis_clip_feats, vis_clip_labels, classes_2),
    "RemoteCLIP ViT-L/14": compute_table(dota_rc_feats, dota_rc_labels, vis_rc_feats, vis_rc_labels, classes_2),
}
plot_comparison("dota_vs_visdrone", tables_1, classes_2,
                 "Is cross-altitude drift as large as a real category change? (DOTA vs. VisDrone)")
all_results["dota_vs_visdrone"] = tables_1

# 2. xView vs VisDrone (3 classes)
def filt(feats, labels, keep):
    m = np.isin(labels, keep)
    return feats[m], labels[m]

xv_clip_feats, xv_clip_labels = load(os.path.join(E_XV, "xview_clip_feats.npy")), load(os.path.join(E_XV, "xview_clip_labels.npy"))
vd_clip_feats, vd_clip_labels = load(os.path.join(E_XV, "visdrone_clip_feats.npy")), load(os.path.join(E_XV, "visdrone_clip_labels.npy"))
xv_rc_feats, xv_rc_labels = load(os.path.join(E_XV, "xview_remoteclip_feats.npy")), load(os.path.join(E_XV, "xview_remoteclip_labels.npy"))
vd_rc_feats, vd_rc_labels = load(os.path.join(E_XV, "visdrone_remoteclip_feats.npy")), load(os.path.join(E_XV, "visdrone_remoteclip_labels.npy"))

classes_3 = ["car", "bus", "truck"]
xv_clip_f3, xv_clip_l3 = filt(xv_clip_feats, xv_clip_labels, classes_3)
vd_clip_f3, vd_clip_l3 = filt(vd_clip_feats, vd_clip_labels, classes_3)
xv_rc_f3, xv_rc_l3 = filt(xv_rc_feats, xv_rc_labels, classes_3)
vd_rc_f3, vd_rc_l3 = filt(vd_rc_feats, vd_rc_labels, classes_3)

tables_2 = {
    "CLIP ViT-B/32": compute_table(xv_clip_f3, xv_clip_l3, vd_clip_f3, vd_clip_l3, classes_3),
    "RemoteCLIP ViT-L/14": compute_table(xv_rc_f3, xv_rc_l3, vd_rc_f3, vd_rc_l3, classes_3),
}
plot_comparison("xview_vs_visdrone", tables_2, classes_3,
                 "Is cross-altitude drift as large as a real category change? (xView vs. VisDrone)")
all_results["xview_vs_visdrone"] = tables_2

# 3. xView+UAVid, 4 classes
xvb_clip = load(os.path.join(E_XV, "xview_building_clip_feats.npy"))
uab_clip = load(os.path.join(E_XV, "uavid_building_clip_feats.npy"))
xvb_rc = load(os.path.join(E_XV, "xview_building_remoteclip_feats.npy"))
uab_rc = load(os.path.join(E_XV, "uavid_building_remoteclip_feats.npy"))

classes_4 = ["car", "bus", "truck", "building"]
sat_clip4_f = np.vstack([xv_clip_f3, xvb_clip]); sat_clip4_l = np.concatenate([xv_clip_l3, np.array(["building"] * len(xvb_clip))])
uav_clip4_f = np.vstack([vd_clip_f3, uab_clip]); uav_clip4_l = np.concatenate([vd_clip_l3, np.array(["building"] * len(uab_clip))])
sat_rc4_f = np.vstack([xv_rc_f3, xvb_rc]); sat_rc4_l = np.concatenate([xv_rc_l3, np.array(["building"] * len(xvb_rc))])
uav_rc4_f = np.vstack([vd_rc_f3, uab_rc]); uav_rc4_l = np.concatenate([vd_rc_l3, np.array(["building"] * len(uab_rc))])

tables_3 = {
    "CLIP ViT-B/32": compute_table(sat_clip4_f, sat_clip4_l, uav_clip4_f, uav_clip4_l, classes_4),
    "RemoteCLIP ViT-L/14": compute_table(sat_rc4_f, sat_rc4_l, uav_rc4_f, uav_rc4_l, classes_4),
}
plot_comparison("xview_visdrone_uavid", tables_3, classes_4,
                 "Is cross-altitude drift as large as a real category change? (xView vs. VisDrone+UAVid)")
all_results["xview_visdrone_uavid"] = tables_3

with open(os.path.join(OUT_DIR, "yardstick_table.json"), "w") as f:
    json.dump(all_results, f, indent=2)
print("saved", os.path.join(OUT_DIR, "yardstick_table.json"))
print("done")
