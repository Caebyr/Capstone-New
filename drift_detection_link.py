"""
Checkpoint B5: does embedding drift explain detection loss?

  python drift_detection_link.py owl     # OWLv2's own embedding of every GT box (GPU), drift in that space
  python drift_detection_link.py link    # per-category table: crop drift (CLIP / RemoteCLIP / OWLv2) vs detection loss

owl:  For a seeded sample of N_IMG images per benchmark set, one OWLv2 forward pass per image (batch 1, no_grad).
      Every non-ignored GT box is matched to the patch whose predicted box has the highest IoU with it; that
      patch's class embedding (the vector OWLv2 compares with the text query) is L2-normalised and kept.
      Domains per category (same pairing as eval_detection.py):
        small-/large-vehicle  dota_tiles (overhead) vs visdrone_ho_D1 (UAV)
        car / bus / truck     xview_val            vs visdrone_ho_D2
        building              xview_val            vs uavid_tv
      At most CAP boxes per category per domain (seeded). Drift = Eqs. 1-3 of the paper (src/stats.drift),
      95% image-bootstrap CI; text alignment a_c and margin m_c at the centroid with OWLv2's text embeddings of
      "a photo of a {}" for the task's classes; delta_m = m_overhead - m_uav. Also reported: the share of GT
      boxes whose best patch has IoU >= 0.5 (a localisation measure in the same pass), and the same drift
      restricted to those well-matched boxes.
link: Six categories x {CLIP, RemoteCLIP, OWLv2-space} drift measures against AP50 drop, found-rate drop and
      naming-accuracy drop (UAV minus overhead) for OWLv2 and Grounding DINO. Spearman rho with an exact
      permutation p-value (6! = 720 orderings). n = 6, so these are descriptive, not a test of a law.
Writes results/detection/owlv2_space_drift.json, drift_vs_detection.json and drift_vs_detection.md.
"""
import argparse
import itertools
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image

from classify_crops import text_metrics
from src.stats import ci, cluster_weights, drift

DET_ROOT, GT = "data/processed/detection", "data/processed/detection/gt"
OUT = "results/detection"
EMB = "embeddings/full_scale/owlv2_gt_embeds.npz"
OWL, OWL_REV = "google/owlv2-base-patch16-ensemble", "cfd3195ba4ea9592eec887ded089f4c08eff231d"
N_IMG, CAP, N_BOOT, SEED = 400, 3000, 500, 20261201
GROUPS = [("dota_tiles",), ("visdrone_ho_D1", "visdrone_ho_D2"), ("xview_val",), ("uavid_tv",)]
PAIRS = {  # category -> (task, overhead set, UAV set)
    "small-vehicle": ("D1", "dota_tiles", "visdrone_ho_D1"),
    "large-vehicle": ("D1", "dota_tiles", "visdrone_ho_D1"),
    "car": ("D2", "xview_val", "visdrone_ho_D2"),
    "bus": ("D2", "xview_val", "visdrone_ho_D2"),
    "truck": ("D2", "xview_val", "visdrone_ho_D2"),
    "building": ("D2", "xview_val", "uavid_tv"),
}
CLASSES = {"D1": ["small-vehicle", "large-vehicle"], "D2": ["car", "bus", "truck", "building"]}
WORDS = {"small-vehicle": "small vehicle", "large-vehicle": "large vehicle"}
CROP_TASK = {"small-vehicle": "P1", "large-vehicle": "P1", "car": "P2+P3", "bus": "P2+P3", "truck": "P2+P3", "building": "P2+P3"}
Image.MAX_IMAGE_PIXELS = None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sample_images(gt, set_name):
    base = set_name.replace("_D1", "").replace("_D2", "")  # VisDrone D1 / D2 share images -> same sample
    ims = sorted(gt["images"], key=lambda im: im["file_name"])
    return random.Random(f"{SEED}-{base}").sample(ims, min(N_IMG, len(ims)))


def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1]); b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-9)


def cmd_owl():
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    proc = Owlv2Processor.from_pretrained(OWL, revision=OWL_REV)
    model = Owlv2ForObjectDetection.from_pretrained(OWL, revision=OWL_REV).to("cuda").eval()
    all_cls = CLASSES["D1"] + CLASSES["D2"]
    with torch.no_grad():
        t = proc(text=[f"a photo of a {WORDS.get(c, c)}" for c in all_cls], return_tensors="pt").to("cuda")
        te = model.owlv2.get_text_features(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).float()
    text = dict(zip(all_cls, (te / te.norm(dim=-1, keepdim=True)).cpu().numpy()))
    rec = {"feat": [], "set": [], "cat": [], "image": [], "iou": []}
    for group in GROUPS:  # sets in one group share images (VisDrone D1 / D2): one forward pass per image
        gts = {s: json.load(open(f"{GT}/{s}.json")) for s in group}
        names = {c["id"]: c["name"] for c in gts[group[0]]["categories"]}
        ims = sample_images(gts[group[0]], group[0])
        by_img = {s: {} for s in group}
        for s in group:
            id_of = {im["file_name"]: im["id"] for im in gts[s]["images"]}
            want = {id_of[im["file_name"]]: im["file_name"] for im in ims}
            for a in gts[s]["annotations"]:
                if a["image_id"] in want and not a["iscrowd"]:
                    by_img[s].setdefault(want[a["image_id"]], []).append(a)
        t0, n = time.time(), 0
        for k, im in enumerate(ims):
            fn = im["file_name"]
            if not any(fn in by_img[s] for s in group):
                continue
            img = Image.open(os.path.join(DET_ROOT, fn)).convert("RGB")
            S = max(img.size)  # OWLv2 pads to a square; predicted boxes are relative to it
            with torch.no_grad():
                inp = proc(text=[["a photo of a car"]], images=img, return_tensors="pt").to("cuda")
                out = model(**inp)  # class_embeds / pred_boxes do not depend on the text query
            ce = out.class_embeds[0].float()
            ce = (ce / ce.norm(dim=-1, keepdim=True)).cpu().numpy()
            cx, cy, w, h = out.pred_boxes[0].float().cpu().numpy().T
            boxes = np.stack([(cx - w / 2) * S, (cy - h / 2) * S, (cx + w / 2) * S, (cy + h / 2) * S], 1)
            for s in group:
                for a in by_img[s].get(fn, []):
                    x, y, w_, h_ = a["bbox"]
                    ov = iou(np.array([x, y, x + w_, y + h_]), boxes)
                    j = int(ov.argmax())
                    rec["feat"].append(ce[j]); rec["set"].append(s); rec["cat"].append(names[a["category_id"]])
                    rec["image"].append(fn); rec["iou"].append(float(ov[j])); n += 1
            if (k + 1) % 100 == 0:
                log(f"  {'+'.join(group)}: {k + 1}/{len(ims)} images, {n} GT boxes, {(k + 1) / (time.time() - t0):.2f} img/s, "
                    f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        log(f"{'+'.join(group)}: {n} GT boxes from {len(ims)} images ({time.time() - t0:.0f} s)")
    os.makedirs(os.path.dirname(EMB), exist_ok=True)
    np.savez_compressed(EMB, feat=np.stack(rec["feat"]).astype(np.float32), set=np.array(rec["set"]), cat=np.array(rec["cat"]),
                        image=np.array(rec["image"]), iou=np.array(rec["iou"], np.float32),
                        text=np.stack([text[c] for c in all_cls]), text_cls=np.array(all_cls))
    log(f"wrote {EMB}")
    analyze_owl()


def analyze_owl():
    z = np.load(EMB)
    F, S, C, I, U = z["feat"], z["set"], z["cat"], z["image"], z["iou"]
    text = dict(zip(z["text_cls"], z["text"]))
    rng = np.random.default_rng(SEED)
    res = {"settings": {"n_img_per_set": N_IMG, "cap_per_cat_domain": CAP, "n_boot": N_BOOT, "seed": SEED,
                        "model": OWL, "revision": OWL_REV}, "categories": {}}
    for c, (task, so, su) in PAIRS.items():
        sides = {}
        for dom, s in (("overhead", so), ("uav", su)):
            idx = np.where((S == s) & (C == c))[0]
            if len(idx) > CAP:
                idx = np.sort(rng.choice(idx, CAP, replace=False))
            sides[dom] = idx
        r = {"n": {d: int(len(v)) for d, v in sides.items()}, "n_images": {d: int(len(np.unique(I[v]))) for d, v in sides.items()},
             "best_iou_ge_0.5": {d: float((U[v] >= 0.5).mean()) for d, v in sides.items()},
             "median_best_iou": {d: float(np.median(U[v])) for d, v in sides.items()}}
        for tag, keep in (("all", None), ("iou50", 0.5)):
            o, u = sides["overhead"], sides["uav"]
            if keep is not None:
                o, u = o[U[o] >= keep], u[U[u] >= keep]
            if min(len(o), len(u)) < 20:
                r[tag] = None
                continue
            pt = drift(F[o], np.ones(len(o)), F[u], np.ones(len(u)))
            bs = {"cos": [], "ratio": []}
            for _ in range(N_BOOT):
                d = drift(F[o], cluster_weights(rng, I[o]), F[u], cluster_weights(rng, I[u]))
                bs["cos"].append(d["cos"]); bs["ratio"].append(d["ratio"])
            r[tag] = {"n": [int(len(o)), int(len(u))], "point": pt, "ci": {k: ci(v) for k, v in bs.items()}}
        # text alignment at the centroid, task's own class list
        cls = CLASSES[task]
        T = np.stack([text[k] for k in cls])
        tm = {}
        for dom, s in (("overhead", so), ("uav", su)):
            idx = sides[dom]
            tm[dom] = text_metrics(F[idx], np.array([c] * len(idx)), T, cls)[c]
        r["text"] = tm
        r["delta_m"] = tm["overhead"]["margin"] - tm["uav"]["margin"]
        res["categories"][c] = r
        log(f"OWLv2 space {c}: n {r['n']}, cos {r['all']['point']['cos']:.4f}, ratio {r['all']['point']['ratio']:.3f}, "
            f"IoU>=0.5 O/U {r['best_iou_ge_0.5']['overhead']:.2f}/{r['best_iou_ge_0.5']['uav']:.2f}, delta_m {r['delta_m']:+.4f}")
    with open(f"{OUT}/owlv2_space_drift.json", "w") as f:
        json.dump(res, f, indent=1, default=float)


def spearman_exact(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    rho = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    r0 = rho(rx, ry)
    perms = [rho(rx, np.array(p)) for p in itertools.permutations(ry)]
    return r0, float(np.mean([abs(p) >= abs(r0) - 1e-12 for p in perms]))


def cmd_link():
    a3 = json.load(open("results/full_scale/drift_full_scale.json"))["tasks"]
    det = json.load(open(f"{OUT}/detection_results.json"))["detectors"]
    owl = json.load(open(f"{OUT}/owlv2_space_drift.json"))["categories"]
    rows = {}
    for c, (task, so, su) in PAIRS.items():
        T = a3[CROP_TASK[c]]
        r = {}
        for m in ("clip", "remoteclip"):
            r[f"{m}_cos"] = T["full"][m][c]["cos"]
            r[f"{m}_ratio"] = T["full"][m][c]["ratio"]
            r[f"{m}_delta_m"] = T["classification_full"][m]["gap"]["delta_m"][c]
        r["owlv2space_cos"] = owl[c]["all"]["point"]["cos"]
        r["owlv2space_ratio"] = owl[c]["all"]["point"]["ratio"]
        r["owlv2space_delta_m"] = owl[c]["delta_m"]
        for d in ("owlv2", "gdino"):
            P = det[d]["per_set"]
            ap = lambda s: P[s]["class_aware"][c]["AP50"]
            fr = lambda s: P[s]["naming"]["found_rate"][c]
            na = lambda s: P[s]["naming"]["naming_accuracy_of_found"][c]
            r[f"{d}_AP50_drop"] = ap(su) - ap(so)
            r[f"{d}_found_drop"] = fr(su) - fr(so)
            r[f"{d}_named_drop"] = na(su) - na(so)
        rows[c] = r
    cats = list(PAIRS)
    xs = [k for k in rows[cats[0]] if not k.startswith(("owlv2_", "gdino_"))]
    ys = [k for k in rows[cats[0]] if k.startswith(("owlv2_", "gdino_"))]
    corr = {x: {y: dict(zip(("rho", "p_exact"), spearman_exact([rows[c][x] for c in cats], [rows[c][y] for c in cats]))) for y in ys} for x in xs}
    res = {"categories": cats, "rows": rows, "spearman": corr,
           "note": "n = 6 categories; exact permutation p over 720 orderings; descriptive only."}
    with open(f"{OUT}/drift_vs_detection.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    write_md(res, owl)
    for x in xs:
        log(f"{x}: " + ", ".join(f"{y} rho {corr[x][y]['rho']:+.2f} (p {corr[x][y]['p_exact']:.2f})" for y in ys))


def write_md(res, owl):
    cats, rows = res["categories"], res["rows"]
    f3, pc = (lambda v: f"{v:.3f}"), (lambda v: f"{100 * v:+.1f}")
    L = ["# Does embedding drift explain detection loss? (B5)", "",
         "Drop = UAV minus overhead (positive = worse on overhead), in percentage points. Crop drift (CLIP, RemoteCLIP) is the "
         "full-scale A3 cache; OWLv2-space drift uses OWLv2's own class embedding of the patch that best matches each GT box "
         "on the detection benchmark. Δm = text margin overhead − UAV (negative = the text query fits the overhead centroid worse).", "",
         "## OWLv2 embedding space", "",
         "| Category | n (O / U) | cos drift [CI] | ratio [CI] | ratio, IoU ≥ 0.5 only | best-patch IoU ≥ 0.5 (O / U) | margin O / U | Δm |",
         "|---|---|---|---|---|---|---|---|"]
    for c in cats:
        o = owl[c]; a = o["all"]; b = o["iou50"]
        L.append(f"| {c} | {o['n']['overhead']} / {o['n']['uav']} | {a['point']['cos']:.4f} [{a['ci']['cos'][0]:.4f}, {a['ci']['cos'][1]:.4f}] | "
                 f"{a['point']['ratio']:.3f} [{a['ci']['ratio'][0]:.3f}, {a['ci']['ratio'][1]:.3f}] | "
                 f"{'—' if b is None else format(b['point']['ratio'], '.3f')} | "
                 f"{100 * o['best_iou_ge_0.5']['overhead']:.0f}% / {100 * o['best_iou_ge_0.5']['uav']:.0f}% | "
                 f"{o['text']['overhead']['margin']:+.4f} / {o['text']['uav']['margin']:+.4f} | {o['delta_m']:+.4f} |")
    L += ["", "## Per category: drift vs detection loss", "",
          "| Category | CLIP ratio | RemoteCLIP ratio | OWLv2-space ratio | CLIP Δm | OWLv2-space Δm | OWLv2 AP50 drop | OWLv2 found drop | OWLv2 named drop | G-DINO AP50 drop | G-DINO found drop | G-DINO named drop |",
          "|---|" + "---|" * 11]
    for c in cats:
        r = rows[c]
        L.append(f"| {c} | {f3(r['clip_ratio'])} | {f3(r['remoteclip_ratio'])} | {f3(r['owlv2space_ratio'])} | {r['clip_delta_m']:+.4f} | "
                 f"{r['owlv2space_delta_m']:+.4f} | " + " | ".join(pc(r[f'{d}_{k}_drop']) for d in ("owlv2", "gdino") for k in ("AP50", "found", "named")) + " |")
    ys = list(next(iter(res["spearman"].values())))
    L += ["", "## Spearman rank correlation across the 6 categories (exact p)", "",
          "| Drift measure | " + " | ".join(y.replace("_", " ") for y in ys) + " |", "|---|" + "---|" * len(ys)]
    for x, d in res["spearman"].items():
        L.append(f"| {x.replace('_', ' ')} | " + " | ".join(f"{d[y]['rho']:+.2f} ({d[y]['p_exact']:.2f})" for y in ys) + " |")
    L += ["", "With six categories the smallest attainable two-sided p is 2/720 ≈ 0.003, and |rho| ≥ 0.89 is needed for p < 0.05; "
          "treat these as descriptive."]
    with open(f"{OUT}/drift_vs_detection.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["owl", "analyze", "link"])
    a = ap.parse_args()
    {"owl": cmd_owl, "analyze": analyze_owl, "link": cmd_link}[a.cmd]()
