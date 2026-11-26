"""
Checkpoint C3 (no-download variant): image-quality control. A5 matched object SIZE and removed about two-thirds
of the xView-vs-UAV drift but only half of the accuracy gap. Is the rest image quality (optical blur, colour /
contrast) or something a crop-level edit cannot reach (viewing geometry, context)?

  python controls_quality.py calibrate                 # CPU: pick the blur that matches overhead sharpness
  python controls_quality.py encode --model clip       # GPU, batch 4 (also remoteclip, clip_l14)
  python controls_quality.py analyze                   # CPU -> results/full_scale/controls_quality.{json,md}

Same UAV crops and overhead size donors as A5 (results/full_scale/controls_resample_plan.csv.gz), P2+P3 only
(car, bus, truck, building: xView vs VisDrone / UAVid, the comparisons where the platform effect is real per A4).
Variants of each UAV crop, applied after A5's shrink to the donor's size:
  S        shrink only (A5)
  S+B      + Gaussian blur, sigma chosen per category so the median sharpness of the shrunk-blurred UAV crops
           equals that of the overhead donors (sharpness = mean squared gradient / grey-level variance, measured at
           the crop's own pixel size; calibrated on CAL crops per category)
  S+C      + colour / contrast transfer: each RGB channel of the UAV crop is mapped to the mean and standard
           deviation of the same channel of its overhead donor crop
  S+B+C    both
  S+Cu     placebo for C: the same colour transfer, but from a random other UAV crop of the same category
           (if only overhead donors move the crop toward the overhead centroid, C is specific, not generic damage)
For each variant: cosine drift and ratio overhead vs UAV-variant, zero-shot accuracy, balanced accuracy and
margin, CLIP B/32, RemoteCLIP L/14 and CLIP L/14 (C2). Whatever gap survives S+B+C is not explained by object
size, sharpness or colour statistics of the crop.
"""
import argparse
import csv
import gzip
import io
import json
import os
import time

import numpy as np
from PIL import Image, ImageFilter

from drift_full_scale import TASKS, build, load_cache
from src.full_scale import iter_xview
from src.stats import drift

OUT, EMB = "results/full_scale", "embeddings/full_scale"
PLAN = f"{OUT}/controls_resample_plan.csv.gz"
CAL_FILE = f"{OUT}/controls_quality_calibration.json"
TASK, CAL, SEED = "P2+P3", 300, 20261202
SIGMAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
VARIANTS = ["S+B", "S+C", "S+B+C", "S+Cu"]
MODELS = ["clip", "remoteclip", "clip_l14"]
NAMES = {"clip": "CLIP ViT-B/32", "remoteclip": "RemoteCLIP ViT-L/14", "clip_l14": "CLIP ViT-L/14"}
Image.MAX_IMAGE_PIXELS = None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def plan_rows():
    with gzip.open(PLAN, "rt", newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["task"] == TASK]


class Crops:
    """Loads crops by box_id, decoding each source image once (xView chips from the local parquet)."""

    def __init__(self, man, ids):
        self.man = man
        need = {int(man[b]["image_ref"].split(":")[1]) for b in ids if man[b]["dataset"] == "xView"}
        self.x = {i: row["image"]["bytes"] for i, row in iter_xview("train", columns=["image"]) if i in need} if need else {}
        self.ref, self.img = None, None

    def get(self, b):
        r = self.man[b]
        if r["image_ref"] != self.ref:
            self.ref = r["image_ref"]
            self.img = (Image.open(io.BytesIO(self.x[int(self.ref.split(":")[1])])) if r["dataset"] == "xView" else Image.open(self.ref)).convert("RGB")
        return self.img.crop((int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"])))


def sharpness(im):
    g = np.asarray(im.convert("L"), np.float64)
    if min(g.shape) < 3 or g.var() < 1e-6:
        return np.nan
    gy, gx = np.gradient(g)
    return float(np.mean(gx ** 2 + gy ** 2) / g.var())


def colour_transfer(im, donor):
    a = np.asarray(im, np.float64); d = np.asarray(donor, np.float64)
    ma, sa = a.reshape(-1, 3).mean(0), a.reshape(-1, 3).std(0) + 1e-6
    md, sd = d.reshape(-1, 3).mean(0), d.reshape(-1, 3).std(0)
    return Image.fromarray(np.clip((a - ma) / sa * sd + md, 0, 255).astype(np.uint8))


def variant(uav, donor, r, sigma, which):
    im = uav.resize((int(r["new_w"]), int(r["new_h"])), Image.BICUBIC)  # exactly A5's shrink
    if "B" in which.split("+")[1:] and sigma > 0:
        im = im.filter(ImageFilter.GaussianBlur(sigma))
    if "C" in which.split("+")[1:] or "Cu" in which.split("+")[1:]:
        im = colour_transfer(im, donor)
    return im


def cmd_calibrate():
    man, _, _ = load_cache()
    rows = plan_rows()
    rng = np.random.default_rng(SEED)
    cats = TASKS[TASK]["cats"]
    pick = {c: [rows[i] for i in rng.choice([k for k, r in enumerate(rows) if r["label"] == c], CAL, replace=False)] for c in cats}
    src = Crops(man, [b for c in cats for r in pick[c] for b in (r["box_id"], r["donor"])])
    res = {"cal_per_cat": CAL, "sigmas": SIGMAS, "categories": {}}
    for c in cats:
        items = sorted(pick[c], key=lambda r: man[r["box_id"]]["image_ref"])
        uav = [src.get(r["box_id"]) for r in items]
        items_d = sorted(pick[c], key=lambda r: man[r["donor"]]["image_ref"])
        over = np.nanmedian([sharpness(src.get(r["donor"])) for r in items_d])
        curve = {}
        for s in SIGMAS:
            curve[s] = float(np.nanmedian([sharpness(variant(u, None, r, s, "S+B")) for u, r in zip(uav, items)]))
        orig = float(np.nanmedian([sharpness(u) for u in uav]))
        best = min(SIGMAS, key=lambda s: abs(curve[s] - over))
        res["categories"][c] = {"overhead_median_sharpness": float(over), "uav_original_median_sharpness": orig,
                                "uav_shrunk_by_sigma": curve, "sigma": best}
        log(f"{c}: overhead sharpness {over:.3f}, UAV original {orig:.3f}, shrunk {curve[0.0]:.3f} -> sigma {best} ({curve[best]:.3f})")
    with open(CAL_FILE, "w") as f:
        json.dump(res, f, indent=1)


def cmd_encode(model_name):
    import torch
    from encode_full_scale import encode_batch, load_model
    man, _, _ = load_cache()
    rows = plan_rows()
    sig = {c: v["sigma"] for c, v in json.load(open(CAL_FILE))["categories"].items()}
    todo = [v for v in (["S"] if model_name == "clip_l14" else []) + VARIANTS
            if not os.path.exists(f"{EMB}/quality_{v.replace('+', '')}_{model_name}.npy")]
    if not todo:
        log("nothing to do"); return
    model, pre = load_model(model_name, "cuda")
    src = Crops(man, [b for r in rows for b in (r["box_id"], r["donor"])])
    # donors in their own pass (sorted by source image), then UAV crops sorted by source image
    donors = {}
    if any(v != "S+Cu" for v in todo):
        for b in sorted({r["donor"] for r in rows}, key=lambda b: man[b]["image_ref"]):
            donors[b] = src.get(b)
    rng = np.random.default_rng(SEED + 1)
    by_cat = {}
    for k, r in enumerate(rows):
        by_cat.setdefault(r["label"], []).append(k)
    placebo = {}
    for c, ks in by_cat.items():
        perm = rng.permutation(ks)
        while np.any(perm == np.array(ks)):  # never a crop's own colours
            perm = rng.permutation(ks)
        placebo.update(dict(zip(ks, perm)))
    if "S+Cu" in todo:
        pd_ids = sorted({rows[placebo[k]]["box_id"] for k in range(len(rows))}, key=lambda b: man[b]["image_ref"])
        pdon = {}
        for b in pd_ids:  # only the colour statistics are used; a <=64 px thumbnail keeps memory small
            im = src.get(b)
            im.thumbnail((64, 64))
            pdon[b] = im
    order = sorted(range(len(rows)), key=lambda k: man[rows[k]["box_id"]]["image_ref"])
    out = {v: [None] * len(rows) for v in todo}
    buf = {v: [] for v in todo}; idx = []
    t0 = time.time()
    with torch.no_grad():
        for n, k in enumerate(order):
            r = rows[k]
            u = src.get(r["box_id"])
            for v in todo:
                d = pdon[rows[placebo[k]]["box_id"]] if v == "S+Cu" else donors[r["donor"]]
                buf[v].append(variant(u, d, r, sig[r["label"]], v))
            idx.append(k)
            if len(idx) == 64 or n == len(order) - 1:
                for v in todo:
                    for kk, f in zip(idx, encode_batch(model_name, model, pre, buf[v], "cuda")):
                        out[v][kk] = f
                    buf[v] = []
                idx = []
            if (n + 1) % 1000 == 0:
                log(f"  {model_name}: {n + 1}/{len(rows)} UAV crops x {len(todo)} variants, {(n + 1) / (time.time() - t0):.1f} crops/s, "
                    f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    for v in todo:
        F = np.stack(out[v]).astype(np.float32)
        F /= np.linalg.norm(F, axis=1, keepdims=True)
        np.save(f"{EMB}/quality_{v.replace('+', '')}_{model_name}.npy", F)
    log(f"wrote {len(todo)} variant files for {model_name} ({time.time() - t0:.0f} s)")


def block(F, labels, text, cats):
    from classify_crops import class_metrics, classify, text_metrics
    _, pred = classify(F, text, cats)
    cm = class_metrics(labels, pred, cats)
    tm = text_metrics(F, labels, text, cats)
    return {"acc": cm["accuracy"], "bal": float(np.mean([cm["per_category"][c]["recall"] for c in cats])),
            "recall": {c: cm["per_category"][c]["recall"] for c in cats}, "margin": {c: tm[c]["margin"] for c in cats},
            "pred_share": {c: float(np.mean(pred == c)) for c in cats}}


def cmd_analyze():
    import same_backbone
    man, member, feats = same_backbone.feats_with_l14()
    bank = same_backbone.text_banks()
    rows = plan_rows()
    cats = TASKS[TASK]["cats"]
    cal = json.load(open(CAL_FILE))
    spec = dict(TASKS[TASK], models=MODELS)
    TASKS["_q"] = spec
    data = build("_q", man, member, feats)
    labels = np.array([r["label"] for r in rows])
    a5_all = [r for r in csv.DictReader(gzip.open(PLAN, "rt", newline="", encoding="utf-8"))]
    a5_pos = [k for k, r in enumerate(a5_all) if r["task"] == TASK]
    res = {"settings": {"task": TASK, "n_uav_per_cat": {c: int((labels == c).sum()) for c in cats}, "seed": SEED,
                        "sigma": {c: cal["categories"][c]["sigma"] for c in cats}}, "calibration": cal["categories"], "models": {}}
    for m in MODELS:
        F, index = feats[m]
        V = {"original": F[[index[r["box_id"]] for r in rows]]}
        s_path = f"{EMB}/controls_shrunk_{m}.npy" if m != "clip_l14" else f"{EMB}/quality_S_{m}.npy"
        if not os.path.exists(s_path):
            continue
        V["S"] = np.load(s_path).astype(np.float64)[a5_pos] if m != "clip_l14" else np.load(s_path).astype(np.float64)
        for v in VARIANTS:
            p = f"{EMB}/quality_{v.replace('+', '')}_{m}.npy"
            if os.path.exists(p):
                V[v] = np.load(p).astype(np.float64)
        ov = data[m]["overhead"]
        text = bank[m][TASK]["a photo of a {}"]
        r = {"drift": {}, "classification": {"overhead": block(ov["feats"], ov["labels"], text, cats)}}
        for v, X in V.items():
            r["drift"][v] = {}
            for c in cats:
                mo, mu = ov["labels"] == c, labels == c
                r["drift"][v][c] = drift(ov["feats"][mo], np.ones(mo.sum()), X[mu], np.ones(mu.sum()))
            r["classification"][v] = block(X, labels, text, cats)
        res["models"][m] = r
        log(f"{m}: " + ", ".join(f"{v} bal {100 * r['classification'][v]['bal']:.1f}" for v in V) + f"; overhead {100 * r['classification']['overhead']['bal']:.1f}")
    with open(f"{OUT}/controls_quality.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    write_md(res)
    log(f"wrote {OUT}/controls_quality.json and .md")


def write_md(R):
    cats = list(R["settings"]["sigma"])
    L = ["# Image-quality control: size, blur and colour matched UAV crops (C3)", "",
         "P2+P3 (xView vs VisDrone / UAVid). The same UAV crops and overhead size donors as A5. S = shrink to the donor's size "
         "(A5); B = Gaussian blur calibrated so median crop sharpness matches the overhead crops; C = per-channel mean / std "
         "transferred from the donor crop. Prompt \"a photo of a {}\".", "",
         "## Calibration", "", "| Category | overhead sharpness | UAV original | UAV shrunk | sigma chosen | UAV shrunk + blur |", "|---|---|---|---|---|---|"]
    for c in cats:
        k = R["calibration"][c]
        L.append(f"| {c} | {k['overhead_median_sharpness']:.3f} | {k['uav_original_median_sharpness']:.3f} | {k['uav_shrunk_by_sigma']['0.0']:.3f} | "
                 f"{k['sigma']} | {k['uav_shrunk_by_sigma'][str(k['sigma'])]:.3f} |")
    for m, r in R["models"].items():
        vs = list(r["drift"])
        L += ["", f"## {NAMES[m]}", "", "Cosine drift, overhead vs UAV variant:", "", "| Category | " + " | ".join(vs) + " |", "|---|" + "---|" * len(vs)]
        for c in cats:
            L.append(f"| {c} | " + " | ".join(f"{r['drift'][v][c]['cos']:.4f}" for v in vs) + " |")
        L += ["", "Zero-shot classification on the UAV variant (overhead crops for reference):", "",
              "| | overhead | " + " | ".join(vs) + " |", "|---|---|" + "---|" * len(vs)]
        C = r["classification"]
        L.append("| balanced accuracy | " + " | ".join(f"{100 * C[v]['bal']:.1f}%" for v in ["overhead"] + vs) + " |")
        L.append("| accuracy | " + " | ".join(f"{100 * C[v]['acc']:.1f}%" for v in ["overhead"] + vs) + " |")
        for c in cats:
            L.append(f"| margin {c} | " + " | ".join(f"{C[v]['margin'][c]:+.4f}" for v in ["overhead"] + vs) + " |")
        L.append("| share labelled 'building' | " + " | ".join(f"{100 * C[v]['pred_share']['building']:.0f}%" for v in ["overhead"] + vs) + " |")
        gap0 = C["original"]["bal"] - C["overhead"]["bal"]
        L += ["", "Share of the balanced-accuracy gap (UAV original − overhead) closed by each variant: "
              + ", ".join(f"{v} {100 * (C['original']['bal'] - C[v]['bal']) / gap0:.0f}%" for v in vs if v != "original") + "."]
        if "S+Cu" in vs:
            sp = lambda v: 100 * (C["original"]["bal"] - C[v]["bal"]) / gap0
            L += ["", f"Overhead-specific colour effect (S+C minus the UAV-donor placebo S+Cu): {sp('S+C') - sp('S+Cu'):+.0f} points of the gap; "
                  "car drift " + f"{r['drift']['S+C']['car']['cos']:.4f} vs placebo {r['drift']['S+Cu']['car']['cos']:.4f}. "
                  "Whatever the placebo also achieves is generic perturbation, not a move toward the overhead look."]
    with open(f"{OUT}/controls_quality.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "encode", "analyze"])
    ap.add_argument("--model", choices=MODELS)
    a = ap.parse_args()
    {"calibrate": cmd_calibrate, "analyze": cmd_analyze}.get(a.cmd, lambda: cmd_encode(a.model))()
