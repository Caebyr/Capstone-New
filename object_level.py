"""
Checkpoint C1: object-level test. Does an object's own text margin predict whether a detector finds and names it,
after controlling for object size and category? Does it account for part of the overhead penalty?

  python object_level.py table                    # CPU: replay the B5 object sample, attach box size + detection outcomes
  python object_level.py encode --model clip      # GPU (batch 4, no_grad): crop every object, encode (also --model remoteclip)
  python object_level.py analyze                  # CPU: logistic models, image-cluster bootstrap

Why: B5 compared 6 category averages, so no correlation could reach significance (n = 6). Here the unit is the object
(~28,000 GT boxes, the B5 sample: 400 seeded images per benchmark set, non-ignored boxes, same pairing as B5).

Per object i:
  margin_i (encoder E) = cos(e_i, t_true) - max_{k != true} cos(e_i, t_k), text "a photo of a {}" over the task's classes
                         (D1: small-/large-vehicle; D2: car, bus, truck, building). E = CLIP ViT-B/32, RemoteCLIP ViT-L/14
                         (GT crop encoded independently of either detector) and OWLv2-internal (the class embedding of the
                         patch that best matches the box, from B5; near-circular for OWLv2's own outcomes, reported as a check).
  size: log2 sqrt(box area in px), log2 aspect ratio (long / short side)
  outcomes per detector (OWLv2, Grounding DINO), score >= 0.1 and IoU >= 0.5 as in eval_detection.py:
    found        some detection overlaps the box
    named|found  the best-scoring overlapping detection has the right class (found objects only)
    success      found and named
Models (logistic, category fixed effects; margin z-scored over the sample):
  M0 overhead          M1 + size          M2 + size + margin
  within-domain: success ~ category + size + margin, overhead objects only and UAV objects only
Reported: odds ratio per SD of margin; average marginal effect (AME, percentage points) of overhead in M0/M1/M2, i.e.
how much of the overhead gap remains once size, then margin, are held fixed; 95% CIs from an image-cluster bootstrap.
Writes results/detection/object_level.json, object_level.md, object_table.csv.gz.
"""
import argparse
import gzip
import csv
import json
import os
import random
import time

import numpy as np
from PIL import Image

from src.stats import ci, cluster_weights

DET_ROOT, GT, DETS, OUT = "data/processed/detection", "data/processed/detection/gt", "data/processed/detection/dets", "results/detection"
B5_EMB = "embeddings/full_scale/owlv2_gt_embeds.npz"
EMB = "embeddings/full_scale/object_level_{}.npz"
TABLE = f"{OUT}/object_table.csv.gz"
TEXT_BANK = "results/full_scale/prompt_text_embeddings.npz"
B5_SEED, N_IMG = 20261201, 400            # must match drift_detection_link.py (the replay reproduces its order)
GROUPS = [("dota_tiles",), ("visdrone_ho_D1", "visdrone_ho_D2"), ("xview_val",), ("uavid_tv",)]
PAIRS = {"small-vehicle": ("D1", "dota_tiles", "visdrone_ho_D1"), "large-vehicle": ("D1", "dota_tiles", "visdrone_ho_D1"),
         "car": ("D2", "xview_val", "visdrone_ho_D2"), "bus": ("D2", "xview_val", "visdrone_ho_D2"),
         "truck": ("D2", "xview_val", "visdrone_ho_D2"), "building": ("D2", "xview_val", "uavid_tv")}
CLASSES = {"D1": ["small-vehicle", "large-vehicle"], "D2": ["car", "bus", "truck", "building"]}
TEXT_TASK = {"D1": "P1", "D2": "P2+P3"}
CATS = list(PAIRS)
DETECTORS = {"owlv2": "OWLv2 B/16", "gdino": "Grounding DINO tiny"}
ENCODERS = {"clip": "CLIP ViT-B/32", "remoteclip": "RemoteCLIP ViT-L/14", "owl": "OWLv2-internal"}
MIN_SCORE, IOU_THR = 0.1, 0.5
N_BOOT, SEED = 500, 20261127
Image.MAX_IMAGE_PIXELS = None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def replay():
    """The B5 object list in exactly the order drift_detection_link.py cmd_owl wrote it, with annotations attached."""
    out = []
    for group in GROUPS:
        gts = {s: json.load(open(f"{GT}/{s}.json")) for s in group}
        names = {c["id"]: c["name"] for c in gts[group[0]]["categories"]}
        base = group[0].replace("_D1", "").replace("_D2", "")
        ims = sorted(gts[group[0]]["images"], key=lambda im: im["file_name"])
        ims = random.Random(f"{B5_SEED}-{base}").sample(ims, min(N_IMG, len(ims)))
        by_img = {s: {} for s in group}
        for s in group:
            id_of = {im["file_name"]: im["id"] for im in gts[s]["images"]}
            want = {id_of[im["file_name"]]: im["file_name"] for im in ims}
            for a in gts[s]["annotations"]:
                if a["image_id"] in want and not a["iscrowd"]:
                    by_img[s].setdefault(want[a["image_id"]], []).append(a)
        for im in ims:
            fn = im["file_name"]
            for s in group:
                for a in by_img[s].get(fn, []):
                    out.append({"set": s, "cat": names[a["category_id"]], "image": fn, "image_id": a["image_id"],
                                "bbox": a["bbox"], "names": names})
    return out


def iou_xywh(b, boxes):
    x, y, w, h = b
    iw = np.clip(np.minimum(x + w, boxes[:, 0] + boxes[:, 2]) - np.maximum(x, boxes[:, 0]), 0, None)
    ih = np.clip(np.minimum(y + h, boxes[:, 1] + boxes[:, 3]) - np.maximum(y, boxes[:, 1]), 0, None)
    inter = iw * ih
    return inter / (w * h + boxes[:, 2] * boxes[:, 3] - inter + 1e-9)


def cmd_table():
    objs = replay()
    z = np.load(B5_EMB)
    assert len(objs) == len(z["set"]), (len(objs), len(z["set"]))
    assert all(o["set"] == s and o["cat"] == c and o["image"] == i for o, s, c, i in zip(objs, z["set"], z["cat"], z["image"])), \
        "replay order differs from the B5 cache"
    log(f"replayed {len(objs)} objects, identical order to {B5_EMB}")
    keep = [k for k, o in enumerate(objs) if o["cat"] in PAIRS and o["set"] in PAIRS[o["cat"]][1:]]
    rows = []
    dets_cache = {}
    for k in keep:
        o = objs[k]
        task, so, su = PAIRS[o["cat"]]
        x, y, w, h = o["bbox"]
        r = {"row_b5": k, "set": o["set"], "cat": o["cat"], "task": task, "overhead": int(o["set"] == so), "image": o["image"],
             "x": x, "y": y, "w": w, "h": h, "best_patch_iou": float(z["iou"][k])}
        for d in DETECTORS:
            key = (d, o["set"])
            if key not in dets_cache:
                by = {}
                for det in json.load(open(f"{DETS}/{d}__{o['set']}.json")):
                    if det["score"] >= MIN_SCORE:
                        by.setdefault(det["image_id"], []).append(det)
                dets_cache[key] = {i: (np.array([q["bbox"] for q in v], float), np.array([q["score"] for q in v]),
                                       [q["category_id"] for q in v]) for i, v in by.items()}
            found = named = 0
            if o["image_id"] in dets_cache[key]:
                boxes, scores, cids = dets_cache[key][o["image_id"]]
                ok = np.where(iou_xywh(o["bbox"], boxes) >= IOU_THR)[0]
                if ok.size:
                    found = 1
                    named = int(o["names"].get(cids[ok[np.argmax(scores[ok])]]) == o["cat"])
            r[f"{d}_found"], r[f"{d}_named"] = found, named
        rows.append(r)
    with gzip.open(TABLE, "wt", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    log(f"wrote {TABLE}: {len(rows)} objects")
    for d in DETECTORS:
        for dom in (1, 0):
            sub = [r for r in rows if r["overhead"] == dom]
            log(f"  {d} {'overhead' if dom else 'uav'}: found {np.mean([r[f'{d}_found'] for r in sub]):.3f}, "
                f"success {np.mean([r[f'{d}_found'] * r[f'{d}_named'] for r in sub]):.3f} (n {len(sub)})")


def load_table():
    with gzip.open(TABLE, "rt", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cmd_encode(name):
    import torch
    from encode_full_scale import encode_batch, load_model
    rows = load_table()
    model, pre = load_model(name, "cuda")
    order = sorted(range(len(rows)), key=lambda i: rows[i]["image"])
    feats = np.zeros((len(rows), 0), np.float32)
    out, buf, idx, cur_fn, cur = {}, [], [], None, None
    t0 = time.time()
    with torch.no_grad():
        for n, i in enumerate(order):
            r = rows[i]
            if r["image"] != cur_fn:
                cur_fn = r["image"]
                cur = Image.open(os.path.join(DET_ROOT, cur_fn)).convert("RGB")
            x, y, w, h = (float(r[k]) for k in ("x", "y", "w", "h"))
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(cur.width, max(x1 + 1, int(x + w))), min(cur.height, max(y1 + 1, int(y + h)))
            buf.append(cur.crop((x1, y1, x2, y2))); idx.append(i)
            if len(buf) == 64:
                for j, f in zip(idx, encode_batch(name, model, pre, buf, "cuda")):
                    out[j] = f
                buf, idx = [], []
            if (n + 1) % 4000 == 0:
                log(f"  {name}: {n + 1}/{len(rows)} crops, {(n + 1) / (time.time() - t0):.1f}/s, "
                    f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        if buf:
            for j, f in zip(idx, encode_batch(name, model, pre, buf, "cuda")):
                out[j] = f
    feats = np.stack([out[i] for i in range(len(rows))]).astype(np.float32)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    np.savez_compressed(EMB.format(name), feat=feats, row_b5=np.array([int(r["row_b5"]) for r in rows]))
    log(f"wrote {EMB.format(name)} ({len(rows)} crops, {time.time() - t0:.0f} s)")


# ---------- analysis ----------

def margins(rows):
    """{encoder: per-object margin} using the task's class list and 'a photo of a {}'."""
    bank = np.load(TEXT_BANK)
    b5 = np.load(B5_EMB)
    owl_text = dict(zip(b5["text_cls"], b5["text"]))
    rb5 = np.array([int(r["row_b5"]) for r in rows])
    feats = {"owl": b5["feat"][rb5]}
    for m in ("clip", "remoteclip"):
        z = np.load(EMB.format(m))
        assert (z["row_b5"] == rb5).all()
        feats[m] = z["feat"]
    out = {}
    for m, F in feats.items():
        mg = np.zeros(len(rows))
        for task, cls in CLASSES.items():
            T = (np.stack([owl_text[c] for c in cls]) if m == "owl" else bank[f"{m}||{TEXT_TASK[task]}||a photo of a {{}}"]).astype(np.float64)
            T /= np.linalg.norm(T, axis=1, keepdims=True)
            sel = np.array([r["task"] == task for r in rows])
            cos = F[sel].astype(np.float64) @ T.T
            true = np.array([cls.index(r["cat"]) for r, s in zip(rows, sel) if s])
            own = cos[np.arange(len(true)), true]
            cos[np.arange(len(true)), true] = -np.inf
            mg[sel] = own - cos.max(1)
        out[m] = mg
    return out


def logit_fit(X, y, w, beta0=None, ridge=1e-4, iters=50):
    """Weighted logistic regression by Newton/IRLS; tiny ridge (not on the intercept, column 0) for stability."""
    b = np.zeros(X.shape[1]) if beta0 is None else beta0.copy()
    R = np.full(X.shape[1], ridge); R[0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (w * (y - p)) - R * b
        H = (X * (w * p * (1 - p))[:, None]).T @ X + np.diag(R)
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def design(rows, cols, mg_z=None, extra=("overhead",)):
    """Columns: intercept, category dummies (first category is the reference), then the requested terms."""
    n = len(rows)
    parts, names = [np.ones(n)], ["const"]
    cats_present = [c for c in CATS if any(r["cat"] == c for r in rows)]
    for c in cats_present[1:]:
        parts.append(np.array([r["cat"] == c for r in rows], float)); names.append(f"cat:{c}")
    for e in extra:
        parts.append(cols[e]); names.append(e)
    if "size" in cols and cols.get("_use_size"):
        parts += [cols["size"], cols["aspect"]]; names += ["log2_size", "log2_aspect"]
    if mg_z is not None:
        parts.append(mg_z); names.append("margin_z")
    return np.stack(parts, 1), names


def ame(X, b, j):
    """Average marginal effect of binary column j: mean p(x_j = 1) - p(x_j = 0), in probability units."""
    X1, X0 = X.copy(), X.copy()
    X1[:, j], X0[:, j] = 1, 0
    s = lambda Z: 1 / (1 + np.exp(-np.clip(Z @ b, -30, 30)))
    return float(np.mean(s(X1) - s(X0)))


def cmd_analyze():
    rows = load_table()
    n = len(rows)
    mg = margins(rows)
    ov = np.array([int(r["overhead"]) for r in rows], float)
    w_, h_ = np.array([float(r["w"]) for r in rows]), np.array([float(r["h"]) for r in rows])
    size = np.log2(np.sqrt(w_ * h_)); aspect = np.log2(np.maximum(w_, h_) / np.minimum(w_, h_))
    zs = lambda v: (v - v.mean()) / v.std()
    base_cols = {"overhead": ov, "size": zs(size), "aspect": zs(aspect)}
    mgz = {m: zs(v) for m, v in mg.items()}
    img = np.array([r["image"] for r in rows])
    Y = {}
    for d in DETECTORS:
        f = np.array([int(r[f"{d}_found"]) for r in rows], float)
        nm = np.array([int(r[f"{d}_named"]) for r in rows], float)
        Y[d] = {"found": (f, np.ones(n, bool)), "named_given_found": (nm, f == 1), "success": (f * nm, np.ones(n, bool))}

    # model specs: (key, subset mask, use size, encoder or None, extra binary terms)
    specs = []
    for d in DETECTORS:
        for o in ("found", "named_given_found", "success"):
            specs.append((d, o, "M0", None, False, ("overhead",)))
            specs.append((d, o, "M1", None, True, ("overhead",)))
            for m in ENCODERS:
                specs.append((d, o, f"M2_{m}", m, True, ("overhead",)))
        for m in ENCODERS:
            for dom in (1, 0):
                specs.append((d, "success", f"within_{'overhead' if dom else 'uav'}_{m}", m, True, ()))

    def build(spec):
        d, o, tag, m, use_size, extra = spec
        y, mask = Y[d][o]
        if tag.startswith("within_"):
            mask = mask & (ov == (1 if "_overhead_" in tag else 0))
        sub = [r for r, k in zip(rows, mask) if k]
        cols = {k: v[mask] for k, v in base_cols.items()}
        cols["_use_size"] = use_size
        X, names = design(sub, cols, None if m is None else mgz[m][mask], extra)
        return X, y[mask], mask, names

    built = [build(s) for s in specs]
    point = [logit_fit(X, y, np.ones(len(y))) for X, y, _, _ in built]

    def summarize(X, b, names):
        r = {}
        if "overhead" in names:
            r["ame_overhead_pp"] = 100 * ame(X, b, names.index("overhead"))
        if "margin_z" in names:
            j = names.index("margin_z")
            r["or_margin_per_sd"] = float(np.exp(b[j]))
        if "log2_size" in names:
            r["or_size_per_sd"] = float(np.exp(b[names.index("log2_size")]))
        return r

    pt = [summarize(X, b, names) for (X, _, _, names), b in zip(built, point)]
    rng = np.random.default_rng(SEED)
    boot = [{k: [] for k in p} for p in pt]
    t0 = time.time()
    for rep in range(N_BOOT):
        # one image-cluster draw shared by every model in this replicate
        wt_all = cluster_weights(rng, img)  # per-object weight = times its image was drawn
        for i, ((X, y, mask, names), b0) in enumerate(zip(built, point)):
            wt = wt_all[mask]
            b = logit_fit(X, y, wt, beta0=b0, iters=25)
            for k, v in summarize(X, b, names).items():
                boot[i][k].append(v)
        if (rep + 1) % 20 == 0:
            log(f"  bootstrap {rep + 1}/{N_BOOT} ({time.time() - t0:.0f} s)")

    res = {"settings": {"n_objects": n, "n_boot": N_BOOT, "seed": SEED, "min_score": MIN_SCORE, "iou": IOU_THR,
                        "prompt": "a photo of a {}", "sample": "B5 object sample (drift_detection_link.py)"},
           "counts": {}, "rates": {}, "margin_by_domain": {}, "models": {}}
    for c in CATS:
        res["counts"][c] = {dom: int(sum(1 for r in rows if r["cat"] == c and int(r["overhead"]) == k)) for dom, k in (("overhead", 1), ("uav", 0))}
    for d in DETECTORS:
        res["rates"][d] = {}
        for o, (y, mask) in Y[d].items():
            res["rates"][d][o] = {dom: float(y[mask & (ov == k)].mean()) for dom, k in (("overhead", 1), ("uav", 0))}
    for m in ENCODERS:
        res["margin_by_domain"][m] = {c: {dom: float(np.median(mg[m][np.array([r["cat"] == c for r in rows]) & (ov == k)]))
                                          for dom, k in (("overhead", 1), ("uav", 0))} for c in CATS}
    for s, p, bt in zip(specs, pt, boot):
        d, o, tag = s[0], s[1], s[2]
        res["models"].setdefault(d, {}).setdefault(o, {})[tag] = {k: {"point": v, "ci": ci(bt[k])} for k, v in p.items()}
    # descriptive: success rate by margin quintile within domain (CLIP and RemoteCLIP)
    res["quintiles"] = {}
    for m in ("clip", "remoteclip"):
        res["quintiles"][m] = {}
        for d in DETECTORS:
            res["quintiles"][m][d] = {}
            for dom, k in (("overhead", 1), ("uav", 0)):
                sel = ov == k
                q = np.quantile(mg[m][sel], [0.2, 0.4, 0.6, 0.8])
                bins = np.digitize(mg[m][sel], q)
                ys = Y[d]["success"][0][sel]
                res["quintiles"][m][d][dom] = [float(ys[bins == b].mean()) for b in range(5)]
    with open(f"{OUT}/object_level.json", "w") as f:
        json.dump(res, f, indent=1)
    write_md(res)
    log(f"wrote {OUT}/object_level.json and .md")


def write_md(res):
    M = res["models"]
    f2 = lambda e: f"{e['point']:.2f} [{e['ci'][0]:.2f}, {e['ci'][1]:.2f}]"
    f1 = lambda e: f"{e['point']:+.1f} [{e['ci'][0]:+.1f}, {e['ci'][1]:+.1f}]"
    L = ["# Object-level test: does an object's text margin predict detection? (C1)", "",
         f"Unit = one ground-truth object; {res['settings']['n_objects']:,} objects (the B5 sample: 400 seeded images per set). "
         "Margin = cos(own class text) − max cos(other class texts) for the object's crop, prompt \"a photo of a {}\", "
         "z-scored. CLIP and RemoteCLIP encode the crop independently of both detectors; OWLv2-internal uses OWLv2's own "
         "patch embedding, so for OWLv2's outcomes it is close to circular (shown as a check only). Size = log2 √area and "
         "log2 aspect ratio, z-scored. All models include category fixed effects. 95% CIs: image-cluster bootstrap, "
         f"{res['settings']['n_boot']} reps. Found / named: score ≥ {res['settings']['min_score']}, IoU ≥ {res['settings']['iou']}.", "",
         "## Objects", "", "| Category | overhead | UAV |", "|---|---|---|"]
    for c, v in res["counts"].items():
        L.append(f"| {c} | {v['overhead']} | {v['uav']} |")
    L += ["", "## Raw rates", "", "| Detector | outcome | overhead | UAV |", "|---|---|---|---|"]
    for d, oo in res["rates"].items():
        for o, v in oo.items():
            L.append(f"| {DETECTORS[d]} | {o} | {100 * v['overhead']:.1f}% | {100 * v['uav']:.1f}% |")
    L += ["", "## 1. Does margin predict detection, holding size, category and domain fixed? (model M2)", "",
          "Odds ratio per +1 SD of margin. Above 1 = objects whose crop sits nearer their own class text are more often detected.", "",
          "| Detector | outcome | CLIP | RemoteCLIP | OWLv2-internal (check) |", "|---|---|---|---|---|"]
    for d in DETECTORS:
        for o in ("found", "named_given_found", "success"):
            L.append(f"| {DETECTORS[d]} | {o} | " + " | ".join(f2(M[d][o][f"M2_{m}"]["or_margin_per_sd"]) for m in ENCODERS) + " |")
    L += ["", "OWLv2-internal for OWLv2's own naming is circular by construction (OWLv2 picks the class whose text is nearest "
          "that same embedding), hence the huge odds ratio; it only confirms the replay is aligned."]
    L += ["", "## 2. Within one domain (success ~ category + size + margin)", "",
          "Rules out margin merely standing in for 'overhead or not'.", "",
          "| Detector | domain | CLIP | RemoteCLIP | OWLv2-internal (check) |", "|---|---|---|---|---|"]
    for d in DETECTORS:
        for dom in ("overhead", "uav"):
            L.append(f"| {DETECTORS[d]} | {dom} | " + " | ".join(f2(M[d]["success"][f"within_{dom}_{m}"]["or_margin_per_sd"]) for m in ENCODERS) + " |")
    L += ["", "## 3. How much of the overhead gap is left? (average marginal effect of overhead, percentage points)", "",
          "M0: category only. M1: + size. M2: + size + margin. Closer to 0 = more of the overhead penalty is accounted for.", "",
          "| Detector | outcome | M0 | M1 (+ size) | M2 CLIP | M2 RemoteCLIP | M2 OWLv2-internal |", "|---|---|---|---|---|---|---|"]
    for d in DETECTORS:
        for o in ("found", "named_given_found", "success"):
            r = M[d][o]
            L.append(f"| {DETECTORS[d]} | {o} | {f1(r['M0']['ame_overhead_pp'])} | {f1(r['M1']['ame_overhead_pp'])} | "
                     + " | ".join(f1(r[f"M2_{m}"]["ame_overhead_pp"]) for m in ENCODERS) + " |")
    L += ["", "## Size effect (M1), odds ratio per +1 SD of log2 size", "", "| Detector | found | named given found | success |", "|---|---|---|---|"]
    for d in DETECTORS:
        L.append(f"| {DETECTORS[d]} | " + " | ".join(f2(M[d][o]["M1"]["or_size_per_sd"]) for o in ("found", "named_given_found", "success")) + " |")
    L += ["", "## Success rate by margin quintile, within domain (descriptive; categories mixed)", "",
          "| Encoder | detector | domain | Q1 (lowest margin) | Q2 | Q3 | Q4 | Q5 |", "|---|---|---|---|---|---|---|---|"]
    for m, dd in res["quintiles"].items():
        for d, doms in dd.items():
            for dom, v in doms.items():
                L.append(f"| {ENCODERS[m]} | {DETECTORS[d]} | {dom} | " + " | ".join(f"{100 * x:.0f}%" for x in v) + " |")
    L += ["", "## Median object margin by category and domain", "", "| Category | " + " | ".join(f"{ENCODERS[m]} O / U" for m in ENCODERS) + " |",
          "|---|" + "---|" * len(ENCODERS)]
    for c in CATS:
        L.append(f"| {c} | " + " | ".join(f"{res['margin_by_domain'][m][c]['overhead']:+.3f} / {res['margin_by_domain'][m][c]['uav']:+.3f}" for m in ENCODERS) + " |")
    with open(f"{OUT}/object_level.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["table", "encode", "analyze"])
    ap.add_argument("--model", choices=["clip", "remoteclip"])
    a = ap.parse_args()
    if a.cmd == "table":
        cmd_table()
    elif a.cmd == "encode":
        cmd_encode(a.model)
    else:
        cmd_analyze()
