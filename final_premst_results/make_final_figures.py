"""
Regenerates the four paper-ready figures (+ the absolute-values table) in this folder.

Inputs (all read from the repo; nothing is recomputed except the t-SNE embedding of Fig. 2):
  results/remoteclip_vs_clip/drift_comparison_3way_heldout.json   -> Fig. 1(a), table
  results/xview_vs_visdrone/drift_comparison_4class.json          -> Fig. 1(b), table
  results/category_yardstick/yardstick_table.json                 -> Fig. 3
  embeddings/xview_vs_visdrone/*.npy  (git-ignored, cached)       -> Fig. 2  (t-SNE)

Run with the project's `hf` environment (from anywhere):
  C:\\anaconda\\envs\\hf\\python.exe final_premst_results\\make_final_figures.py

Shared design choices: sized for a two-column paper page (7.2 in wide, so text stays legible at full
text width); one colour per encoder in every figure (CLIP blue, RemoteCLIP amber, fine-tuned green);
PDF output with embedded TrueType fonts (fonttype 42, accepted by camera-ready checkers); PNG at 300 dpi.
Text is set in Arial when installed (falls back to DejaVu Sans, which is wider, so labels may need nudging).
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from sklearn.manifold import TSNE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

C_CLIP, C_RC, C_FT = "#6C8EBF", "#D79B00", "#82B366"   # encoder colours (same hexes as the earlier repo figures)
C_GREY = "#8C8C8C"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "hatch.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_json(rel):
    with open(ROOT / rel, encoding="utf-8") as f:
        return json.load(f)


def save(fig, stem):
    fig.savefig(HERE / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("saved", stem)


# ------------------------------------------------------------------------------------------------
# Fig. 1 -- drift as % of the CLIP baseline (both comparisons, both metrics, one axis)
# ------------------------------------------------------------------------------------------------
def fig1():
    d3 = load_json("results/remoteclip_vs_clip/drift_comparison_3way_heldout.json")
    d4 = load_json("results/xview_vs_visdrone/drift_comparison_4class.json")
    panels = [
        ("(a) DOTA vs. VisDrone", d3, ["small-vehicle", "large-vehicle"],
         ["remoteclip_vit_l14", "finetuned_clip_visdrone_heldout"], {}),
        ("(b) xView vs. VisDrone / UAVid", d4, ["car", "bus", "truck", "building"],
         ["remoteclip_vit_l14"], {"building": "building*"}),
    ]
    colour = {"remoteclip_vit_l14": C_RC, "finetuned_clip_visdrone_heldout": C_FT}
    metrics = [("l2_drift", ""), ("cosine_drift", "////")]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, gridspec_kw={"width_ratios": [2, 4], "wspace": 0.06})
    for ax, (title, data, classes, models, rename) in zip(axes, panels):
        n_bars = len(models) * len(metrics)
        width = 0.8 / n_bars
        x = np.arange(len(classes))
        i = 0
        for model in models:
            for metric, hatch in metrics:
                vals = [100 * data[model][c][metric] / data["clip_vit_b32"][c][metric] for c in classes]
                off = (i - (n_bars - 1) / 2) * width
                ax.bar(x + off, vals, width, color=colour[model], hatch=hatch, edgecolor="white", linewidth=0.5)
                for xi, v in zip(x + off, vals):
                    ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
                i += 1
        ax.axhline(100, color="#333333", lw=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([rename.get(c, c) for c in classes])
        ax.set_ylim(0, 114)
        ax.set_yticks(range(0, 101, 20))
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Drift, % of CLIP ViT-B/32 baseline\n(lower = less drift)")
    axes[1].text(3.48, 101.8, "CLIP ViT-B/32 baseline = 100%", ha="right", va="bottom", fontsize=6.8, color="#333333")

    handles = [
        Patch(facecolor=C_RC, label="RemoteCLIP ViT-L/14"),
        Patch(facecolor=C_FT, label="CLIP fine-tuned on VisDrone (panel a only)"),
        Patch(facecolor="#777777", edgecolor="white", label="L2 distance"),
        Patch(facecolor="#777777", edgecolor="white", hatch="////", label="Cosine distance"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.075), columnspacing=1.4, handlelength=1.6)
    fig.text(0.5, -0.135,
             "n = 300 crops per class per domain, single seed (42). All three encoders in (a) are scored on the same held-out 20% of VisDrone images;\n"
             "the fine-tuned model was trained on the other 80%.  *building: xView vs. UAVid (VisDrone has no building class); other classes: xView vs. VisDrone.",
             ha="center", va="top", fontsize=6.3, color="#444444")
    save(fig, "fig1_drift_vs_clip_baseline")


# ------------------------------------------------------------------------------------------------
# Fig. 2 -- t-SNE of ground-truth crops: the building class separates by domain
# ------------------------------------------------------------------------------------------------
def fig2():
    E = ROOT / "embeddings" / "xview_vs_visdrone"
    ld = lambda name: np.load(E / name, allow_pickle=True)
    classes = ["car", "bus", "truck", "building"]
    class_colour = {"car": "#1f77b4", "bus": "#2ca02c", "truck": "#8c564b", "building": "#7f7f7f"}   # same as results/pca_tsne/

    def build(model):
        xf, xl = ld(f"xview_{model}_feats.npy"), ld(f"xview_{model}_labels.npy")
        vf, vl = ld(f"visdrone_{model}_feats.npy"), ld(f"visdrone_{model}_labels.npy")
        keep = classes[:3]
        m = np.isin(xl, keep); xf, xl = xf[m], xl[m]
        m = np.isin(vl, keep); vf, vl = vf[m], vl[m]
        xb, ub = ld(f"xview_building_{model}_feats.npy"), ld(f"uavid_building_{model}_feats.npy")
        sat_f, sat_l = np.vstack([xf, xb]), np.concatenate([xl, np.array(["building"] * len(xb))])
        uav_f, uav_l = np.vstack([vf, ub]), np.concatenate([vl, np.array(["building"] * len(ub))])
        feats = np.vstack([sat_f, uav_f])
        labels = np.concatenate([sat_l, uav_l])
        domain = np.array(["sat"] * len(sat_l) + ["uav"] * len(uav_l))
        return feats, labels, domain

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"wspace": 0.04})
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.12)
    for ax, (model, title) in zip(axes, [("clip", "(a) CLIP ViT-B/32"), ("remoteclip", "(b) RemoteCLIP ViT-L/14")]):
        feats, labels, domain = build(model)
        # Same procedure as make_pca_tsne_plots.py (perplexity rule, learning rate, init, seed). The only difference:
        # every point is kept (that script randomly kept 900 of 1,200 per domain, with a different subset per encoder).
        perplexity = min(30, max(5, (len(labels) - 1) // 3))
        coords = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, init="pca", random_state=42).fit_transform(feats)
        for cls in classes:
            for dom, marker in (("sat", "o"), ("uav", "x")):
                sel = (labels == cls) & (domain == dom)
                if marker == "o":
                    ax.scatter(coords[sel, 0], coords[sel, 1], s=8, marker="o", color=class_colour[cls], alpha=0.6, linewidths=0)
                else:
                    ax.scatter(coords[sel, 0], coords[sel, 1], s=9, marker="x", color=class_colour[cls], alpha=0.6, linewidths=0.6)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.8)
        ax.set_title(title, loc="left")
        print(f"  t-SNE {model}: {len(labels)} points, perplexity {perplexity}")

    handles = [Line2D([], [], marker="o", linestyle="", color=class_colour[c], label=c, markersize=5) for c in classes]
    handles += [Line2D([], [], marker="o", linestyle="", color="black", label="satellite (xView)", markersize=4.5),
                Line2D([], [], marker="x", linestyle="", color="black", label="UAV (VisDrone / UAVid)", markersize=4.5, markeredgewidth=0.8)]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.3, handletextpad=0.3)
    fig.text(0.5, -0.05,
             "Ground-truth crops and labels; 300 crops per class per domain, all points shown. t-SNE (perplexity 30, seed 42) fitted jointly on both domains for each encoder.\n"
             "building: xView vs. UAVid; car, bus, truck: xView vs. VisDrone.",
             ha="center", va="top", fontsize=6.3, color="#444444")
    save(fig, "fig2_tsne_building_domain_split")


# ------------------------------------------------------------------------------------------------
# Fig. 3 -- is cross-altitude drift as large as a real category change?
# ------------------------------------------------------------------------------------------------
def fig3():
    Y = load_json("results/category_yardstick/yardstick_table.json")["xview_visdrone_uavid"]
    classes = ["car", "bus", "truck", "building"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, gridspec_kw={"wspace": 0.05})
    for ax, (name, colour, title) in zip(axes, [("CLIP ViT-B/32", C_CLIP, "(a) CLIP ViT-B/32"), ("RemoteCLIP ViT-L/14", C_RC, "(b) RemoteCLIP ViT-L/14")]):
        rows = Y[name]
        ca = [rows[c]["cross_altitude_cosine"] for c in classes]
        cc = [rows[c]["cross_category_cosine"] for c in classes]
        x = np.arange(len(classes))
        w = 0.36
        ax.bar(x - w / 2, ca, w, color=colour, edgecolor="white", linewidth=0.5)
        ax.bar(x + w / 2, cc, w, color="#E3E3E3", edgecolor=C_GREY, hatch="////", linewidth=0.5)
        for i, c in enumerate(classes):
            ax.text(i, max(ca[i], cc[i]) + 0.004, f"{rows[c]['ratio']:.2f}\u00d7", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#222222")
        ax.set_xticks(x)
        ax.set_xticklabels([c + "*" if c == "building" else c for c in classes])
        ax.set_ylim(0, 0.255)
        ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20])
        ax.set_title(title, loc="left")
        ax.legend(handles=[Patch(facecolor=colour, edgecolor="white", label="Cross-altitude drift (same class)"),
                           Patch(facecolor="#E3E3E3", edgecolor=C_GREY, hatch="////", label="Cross-category distance (same altitude)")],
                  loc="upper left", frameon=False, fontsize=7)
    axes[0].set_ylabel("Cosine distance between centroids")
    fig.text(0.5, -0.04,
             "Bold labels: cross-altitude drift \u00f7 cross-category distance. Cross-category = mean cosine distance from a class centroid to every other class centroid within\n"
             "the same domain (both domains pooled). n = 300 crops per class per domain, single seed (42). *building: xView vs. UAVid; others: xView vs. VisDrone.",
             ha="center", va="top", fontsize=6.3, color="#444444")
    save(fig, "fig3_cross_altitude_vs_cross_category")


# ------------------------------------------------------------------------------------------------
# Fig. 4 -- pipeline (replaces diagrams/fig4_1_architecture_preview.png; adds xView/UAVid + drift formulas)
# ------------------------------------------------------------------------------------------------
def fig4():
    W, H = 7.2, 3.92
    fig, ax = plt.subplots(figsize=(W, H))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")

    SAT, UAV, EDGE, PANEL = "#DDE7F3", "#E4DCF0", "#3B3B3B", "#F7F7F7"

    def rbox(x, y, w, h, fc=PANEL, ec=EDGE, lw=0.9, r=0.06, z=1, alpha=1.0):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha)
        ax.add_patch(p)
        return p

    def label(x, y, s, size=6.4, ha="center", va="center", weight="normal", color="#111111", z=5, **kw):
        ax.text(x, y, s, fontsize=size, ha=ha, va=va, fontweight=weight, color=color, zorder=z, **kw)

    def badge(x, y, n):
        ax.add_patch(plt.Circle((x, y), 0.105, fc="#2F4B7C", ec="none", zorder=6))
        label(x, y - 0.003, str(n), size=7, weight="bold", color="white", z=7)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.1, shrinkA=0, shrinkB=0), zorder=4)

    bw, bh, gap = 2.2, 1.6, 0.275
    xs = [0.025 + i * (bw + gap) for i in range(3)]          # right edge of last box = 7.175 < 7.2
    y_bot = 0.04
    y_top = y_bot + bh + 0.62

    def panel(i, row, title, n):
        x = xs[i]
        y = y_top if row == 0 else y_bot
        rbox(x, y, bw, bh)
        badge(x + 0.17, y + bh - 0.17, n)
        label(x + 0.36, y + bh - 0.17, title, size=7.4, ha="left", weight="bold")
        return x, y

    # 1  Datasets ----------------------------------------------------------------------------------
    x, y = panel(0, 0, "Datasets", 1)
    hw = (bw - 0.3) / 2
    for j, (cap, names, fc) in enumerate([("Satellite", "DOTA\nxView\n(streamed)", SAT), ("UAV", "VisDrone\nUAVid", UAV)]):
        bx = x + 0.1 + j * (hw + 0.1)
        rbox(bx, y + 0.13, hw, bh - 0.62, fc=fc, ec="#777777", lw=0.7, z=2)
        label(bx + hw / 2, y + bh - 0.6, cap, size=7, weight="bold")
        label(bx + hw / 2, y + 0.5, names, size=6.8, linespacing=1.25)

    # 2  Ground-truth crops + shared classes ---------------------------------------------------------
    x, y = panel(1, 0, "Ground-truth crops", 2)
    lines = [("Annotation boxes (UAVid: masks)", False), ("\u2192 object crops, side \u2265 16 px", False),
             ("Shared classes per pair:", True),
             ("DOTA \u2194 VisDrone: small-, large-vehicle", False),
             ("xView \u2194 VisDrone: car, bus, truck", False),
             ("xView \u2194 UAVid: building", False),
             ("\u2264 300 crops per class per domain", True)]
    yy = y + bh - 0.44
    for s, bold in lines:
        label(x + 0.13, yy, s, size=6.3, ha="left", weight="bold" if bold else "normal")
        yy -= 0.16

    # 3  Encoders -----------------------------------------------------------------------------------
    x, y = panel(2, 0, "Encoders (frozen)", 3)
    encs = [("CLIP ViT-B/32 (512-d)", C_CLIP), ("RemoteCLIP ViT-L/14 (768-d)", C_RC), ("CLIP fine-tuned on VisDrone\u2020 (512-d)", C_FT)]
    for j, (s, c) in enumerate(encs):
        by = y + bh - 0.70 - j * 0.29
        rbox(x + 0.1, by, bw - 0.2, 0.24, fc=c, ec="#555555", lw=0.6, z=2, alpha=0.6)
        label(x + bw / 2, by + 0.12, s, size=6.3)
    label(x + 0.1, y + 0.05, "\u2020 fine-tuned on 80% of VisDrone images; used for\n   DOTA \u2194 VisDrone only (held-out 20%)", size=5.4, ha="left", va="bottom", color="#333333")

    # 4  Embeddings + centroids ---------------------------------------------------------------------
    x, y = panel(0, 1, "Embeddings & centroids", 4)
    rbox(x + 0.1, y + 0.86, bw - 0.2, 0.36, fc="#FFFFFF", ec="#777777", lw=0.6, z=2)
    label(x + bw / 2, y + 1.04, "L2-normalised image embeddings", size=6.4)
    rbox(x + 0.1, y + 0.2, bw - 0.2, 0.5, fc="#FFFFFF", ec="#777777", lw=0.6, z=2)
    label(x + bw / 2, y + 0.45, "one centroid per class and domain:\n$\\mu_c^{S}$ (satellite), $\\mu_c^{U}$ (UAV)", size=6.4, linespacing=1.35)

    # 5  Drift measures -----------------------------------------------------------------------------
    x, y = panel(1, 1, "Drift measures", 5)
    label(x + 0.16, y + 1.08, "$D_{L2}(c)=\\|\\mu_c^{S}-\\mu_c^{U}\\|_2$", size=7.2, ha="left")
    label(x + 0.16, y + 0.81, "$D_{cos}(c)=1-\\cos(\\mu_c^{S},\\mu_c^{U})$", size=7.2, ha="left")
    label(x + 0.16, y + 0.54, "drift ratio $=D_{L2}(c)\\,/\\,\\bar{r}_c$", size=7.2, ha="left")
    label(x + 0.16, y + 0.27, "$\\bar{r}_c$ = mean crop-to-centroid distance,\naveraged over both domains", size=5.6, ha="left", color="#333333", linespacing=1.3)

    # 6  Analysis -----------------------------------------------------------------------------------
    x, y = panel(2, 1, "Analysis", 6)
    for j, s in enumerate(["Encoder comparison\n(drift as % of CLIP)", "t-SNE of embeddings\n(ground-truth labels)", "Cross-category yardstick"]):
        by = y + bh - 0.74 - j * 0.36
        rbox(x + 0.1, by, bw - 0.2, 0.3, fc="#FFFFFF", ec="#777777", lw=0.6, z=2)
        label(x + bw / 2, by + 0.15, s, size=6.2, linespacing=1.15)

    # arrows -----------------------------------------------------------------------------------------
    ya = y_top + bh / 2
    yb = y_bot + bh / 2
    for row_y in (ya, yb):
        arrow(xs[0] + bw, row_y, xs[1], row_y)
        arrow(xs[1] + bw, row_y, xs[2], row_y)
    # wrap-around connector 3 -> 4
    ymid = (y_top + y_bot + bh) / 2
    x3, x4 = xs[2] + bw / 2, xs[0] + bw / 2
    ax.plot([x3, x3], [y_top, ymid], color=EDGE, lw=1.1, zorder=4, solid_capstyle="butt")
    ax.plot([x3, x4], [ymid, ymid], color=EDGE, lw=1.1, zorder=4, solid_capstyle="butt")
    arrow(x4, ymid, x4, y_bot + bh)
    save(fig, "fig4_pipeline")


# ------------------------------------------------------------------------------------------------
# Absolute-values table (companion to Fig. 1, which is normalised to CLIP)
# ------------------------------------------------------------------------------------------------
def results_table():
    d3 = load_json("results/remoteclip_vs_clip/drift_comparison_3way_heldout.json")
    d4 = load_json("results/xview_vs_visdrone/drift_comparison_4class.json")
    enc = [("clip_vit_b32", "CLIP ViT-B/32"), ("remoteclip_vit_l14", "RemoteCLIP ViT-L/14"), ("finetuned_clip_visdrone_heldout", "CLIP fine-tuned on VisDrone")]
    rows = []

    def add(comp, data, classes, sat_src, uav_src_of, nkeys):
        for c in classes:
            base = data["clip_vit_b32"][c]
            for key, name in enc:
                if key not in data:
                    continue
                r = data[key][c]
                rows.append({
                    "comparison": comp, "class": c, "satellite_source": sat_src, "uav_source": uav_src_of(c), "encoder": name,
                    "l2_drift": f"{r['l2_drift']:.4f}", "cosine_drift": f"{r['cosine_drift']:.4f}", "drift_ratio": f"{r['drift_ratio']:.4f}",
                    "l2_pct_of_clip": f"{100 * r['l2_drift'] / base['l2_drift']:.1f}", "cosine_pct_of_clip": f"{100 * r['cosine_drift'] / base['cosine_drift']:.1f}",
                    "n_satellite": r[nkeys[0]], "n_uav": r[nkeys[1]],
                })
    add("DOTA vs VisDrone (held-out)", d3, ["small-vehicle", "large-vehicle"], "DOTA", lambda c: "VisDrone (held-out 20%)", ("sat_samples", "uav_samples"))
    add("xView vs VisDrone/UAVid", d4, ["car", "bus", "truck", "building"], "xView", lambda c: "UAVid" if c == "building" else "VisDrone (all images)", ("xview_samples", "visdrone_samples"))

    with open(HERE / "results_table.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)

    md = ["| Comparison | Class | Encoder | L2 | Cosine | Drift ratio | L2, % of CLIP | Cosine, % of CLIP | n (sat / UAV) |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['comparison']} | {r['class']} | {r['encoder']} | {r['l2_drift']} | {r['cosine_drift']} | {r['drift_ratio']} | {r['l2_pct_of_clip']} | {r['cosine_pct_of_clip']} | {r['n_satellite']} / {r['n_uav']} |")
    (HERE / "results_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("saved results_table.csv / results_table.md", len(rows), "rows")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    results_table()
