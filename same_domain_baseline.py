"""
Checkpoint A4: same-domain baseline. Is overhead-vs-UAV drift larger than the drift you get from simply
changing dataset (same platform), and larger than the noise floor of two image sets from one dataset?

Three levels of comparison for the same category under the same frozen encoder (full-scale cache):
  L0 within dataset     two disjoint image sets of one dataset
                        DOTA part1 vs part2/3 (small-, large-vehicle); VisDrone train vs held-out images
                        (car); xView chips 0-10,440 vs 10,441-20,880 (car, building)
  L1 same platform      overhead vs overhead: DOTA vs xView (small-vehicle <-> xView car;
                        large-vehicle <-> xView bus + truck + van)
                        UAV vs UAV: VisDrone car vs UAVid car (moving + static masks)
  L2 cross platform     the paper's pairs: DOTA vs VisDrone (P1, all images), xView vs VisDrone (car),
                        xView vs UAVid (building, car)
Measures: cosine drift and the spread-normalized ratio (Eqs. 2-3), image-bootstrap 95% CIs, plus a
matched-n version (300 crops per side, N_SEEDS seeds) so that sample size differences cannot drive the
comparison. CLIP ViT-B/32 and RemoteCLIP ViT-L/14.

Caveats: DOTA's small-/large-vehicle and xView's car / bus+truck+van are not identical definitions; UAVid
car boxes are mask components (adjacent parked cars merge). Both are stated in the output.
Writes results/full_scale/same_domain_baseline.json and .md.
"""
import csv
import gzip
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drift_full_scale import load_cache
from src.stats import ci, cluster_weights, drift

OUT = "results/full_scale"
N_BOOT, N_SEEDS, PER_SIDE, SEED = 1000, 20, 300, 20261129
MODELS = ["clip", "remoteclip"]
XV_SPLIT = 10441  # first parquet shard of the xView train split


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rows(member, man, lists, labels, where=lambda r: True):
    out = []
    for r in member:
        if r["list"] in lists and r["label"] in labels and where(man[r["box_id"]]):
            out.append(r["box_id"])
    return list(dict.fromkeys(out))  # de-duplicate (a crop can sit in two lists)


def side(ids, man, feats, m):
    F, index = feats[m]
    return F[[index[b] for b in ids]], np.array([f"{man[b]['dataset']}/{man[b]['source_image']}" for b in ids])


def comparisons(man, member):
    dota_p1 = lambda r: r["image_ref"].replace("\\", "/").startswith("data/raw/DOTA/images/")
    xv_idx = lambda r: int(r["image_ref"].split(":")[1])
    return [
        # level, name, category, side A ids, side B ids
        ("L0", "DOTA part1 vs part2/3", "small-vehicle", rows(member, man, {"P1_dota"}, {"small-vehicle"}, dota_p1),
         rows(member, man, {"P1_dota"}, {"small-vehicle"}, lambda r: not dota_p1(r))),
        ("L0", "DOTA part1 vs part2/3", "large-vehicle", rows(member, man, {"P1_dota"}, {"large-vehicle"}, dota_p1),
         rows(member, man, {"P1_dota"}, {"large-vehicle"}, lambda r: not dota_p1(r))),
        ("L0", "VisDrone train vs held-out images", "car", rows(member, man, {"P2_visdrone"}, {"car"}, lambda r: r["visdrone_split"] == "train"),
         rows(member, man, {"P2_visdrone"}, {"car"}, lambda r: r["visdrone_split"] == "heldout")),
        ("L0", "xView first vs second half of chips", "car", rows(member, man, {"P2_xview"}, {"car"}, lambda r: xv_idx(r) < XV_SPLIT),
         rows(member, man, {"P2_xview"}, {"car"}, lambda r: xv_idx(r) >= XV_SPLIT)),
        ("L0", "xView first vs second half of chips", "building", rows(member, man, {"P3_xview_building"}, {"building"}, lambda r: xv_idx(r) < XV_SPLIT),
         rows(member, man, {"P3_xview_building"}, {"building"}, lambda r: xv_idx(r) >= XV_SPLIT)),
        ("L1", "DOTA vs xView (overhead vs overhead)", "small-vehicle / car", rows(member, man, {"P1_dota"}, {"small-vehicle"}),
         rows(member, man, {"P2_xview"}, {"car"})),
        ("L1", "DOTA vs xView (overhead vs overhead)", "large-vehicle / bus+truck+van", rows(member, man, {"P1_dota"}, {"large-vehicle"}),
         rows(member, man, {"P2_xview"}, {"bus", "truck", "van"})),
        ("L1", "VisDrone vs UAVid (UAV vs UAV)", "car", rows(member, man, {"P2_visdrone"}, {"car"}),
         rows(member, man, {"A4_uavid_car"}, {"car"})),
        ("L2", "DOTA vs VisDrone (all images)", "small-vehicle", rows(member, man, {"P1_dota"}, {"small-vehicle"}),
         rows(member, man, {"P1a_visdrone_all"}, {"small-vehicle"})),
        ("L2", "DOTA vs VisDrone (all images)", "large-vehicle", rows(member, man, {"P1_dota"}, {"large-vehicle"}),
         rows(member, man, {"P1a_visdrone_all"}, {"large-vehicle"})),
        ("L2", "xView vs VisDrone", "car", rows(member, man, {"P2_xview"}, {"car"}), rows(member, man, {"P2_visdrone"}, {"car"})),
        ("L2", "xView vs UAVid", "car", rows(member, man, {"P2_xview"}, {"car"}), rows(member, man, {"A4_uavid_car"}, {"car"})),
        ("L2", "xView vs UAVid", "building", rows(member, man, {"P3_xview_building"}, {"building"}), rows(member, man, {"P3_uavid_building"}, {"building"})),
    ]


def main():
    t0 = time.time()
    man, member, feats = load_cache()
    rng = np.random.default_rng(SEED)
    out = []
    for level, name, cat, a, b in comparisons(man, member):
        rec = {"level": level, "comparison": name, "category": cat, "n": [len(a), len(b)], "models": {}}
        for m in MODELS:
            fa, ia = side(a, man, feats, m)
            fb, ib = side(b, man, feats, m)
            pt = drift(fa, np.ones(len(fa)), fb, np.ones(len(fb)))
            bs = {"cos": [], "ratio": []}
            for _ in range(N_BOOT):
                d = drift(fa, cluster_weights(rng, ia), fb, cluster_weights(rng, ib))
                bs["cos"].append(d["cos"]); bs["ratio"].append(d["ratio"])
            matched = {"cos": [], "ratio": []}
            for _ in range(N_SEEDS):
                wa, wb = np.zeros(len(fa)), np.zeros(len(fb))
                wa[rng.choice(len(fa), min(PER_SIDE, len(fa)), replace=False)] = 1
                wb[rng.choice(len(fb), min(PER_SIDE, len(fb)), replace=False)] = 1
                d = drift(fa, wa, fb, wb)
                matched["cos"].append(d["cos"]); matched["ratio"].append(d["ratio"])
            rec["models"][m] = {"point": pt, "ci": {k: ci(v) for k, v in bs.items()},
                                "matched_300": {k: [float(np.mean(v)), float(np.min(v)), float(np.max(v))] for k, v in matched.items()}}
        rec["n_images"] = [len(np.unique(ia)), len(np.unique(ib))]
        out.append(rec)
        log(f"{level} {name} {cat}: n={rec['n']} | " + " | ".join(f"{m} cos {rec['models'][m]['point']['cos']:.4f} ratio {rec['models'][m]['point']['ratio']:.3f}" for m in MODELS))
    res = {"settings": {"n_boot": N_BOOT, "n_seeds": N_SEEDS, "per_side": PER_SIDE, "seed": SEED}, "comparisons": out}
    with open(f"{OUT}/same_domain_baseline.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    write_md(res)
    log(f"done in {time.time() - t0:.0f} s")


def write_md(res):
    names = {"clip": "CLIP", "remoteclip": "RemoteCLIP"}
    L = ["# Same-domain baseline: is platform drift larger than dataset drift?", "",
         "L0 = two image sets of one dataset (noise floor); L1 = two datasets from the same platform; L2 = overhead vs UAV "
         "(the paper's kind of comparison). Values: full cache [95% image-bootstrap CI]; matched = mean [min–max] over "
         f"{res['settings']['n_seeds']} draws of {res['settings']['per_side']} crops per side.", "",
         "Caveats: DOTA small/large-vehicle vs xView car / bus+truck+van are approximate category matches; UAVid car boxes are "
         "mask components, so adjacent parked cars can merge into one crop.", ""]
    for metric, title in (("cos", "Cosine drift"), ("ratio", "Spread-normalized ratio")):
        L += [f"## {title}", "", "| Level | Comparison | Category | n (A / B) | " + " | ".join(f"{names[m]} full [CI] | {names[m]} matched-300" for m in MODELS) + " |",
              "|---|---|---|---|" + "---|---|" * len(MODELS)]
        for r in res["comparisons"]:
            cells = []
            for m in MODELS:
                v = r["models"][m]
                p, c, mt = v["point"][metric], v["ci"][metric], v["matched_300"][metric]
                nd = 4 if metric == "cos" else 3
                cells.append(f"{p:.{nd}f} [{c[0]:.{nd}f}, {c[1]:.{nd}f}] | {mt[0]:.{nd}f} [{mt[1]:.{nd}f}–{mt[2]:.{nd}f}]")
            L.append(f"| {r['level']} | {r['comparison']} | {r['category']} | {r['n'][0]} / {r['n'][1]} | " + " | ".join(cells) + " |")
        L.append("")
    with open(f"{OUT}/same_domain_baseline.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
