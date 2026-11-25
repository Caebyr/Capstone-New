"""
Checkpoint B4: SAM proposals + zero-shot naming by CLIP / RemoteCLIP (a detector built from our encoders).

  python sam_pipeline.py propose                      # SAM ViT-B automatic masks -> boxes (GPU)
  python sam_pipeline.py name --model clip            # name every proposal with CLIP (GPU), then remoteclip
  python sam_pipeline.py evaluate                     # COCO metrics on the subset, vs OWLv2 on the same images

Proposals: facebook/sam-vit-base, transformers "mask-generation" pipeline, 32x32 point grid
(points_per_batch 64), pred_iou_thresh 0.88, stability_score_thresh 0.92; mask -> tight box; boxes with a side
< 8 px dropped; at most MAX_PROPOSALS per image (highest SAM IoU score first).
Naming: each proposal crop is embedded with the paper's preprocessing and compared with unit-norm text
embeddings of "a photo of a {}" for the task's classes plus BACKGROUND prompts. Score = softmax(100 * cos)
probability of the best class; proposals whose best match is a background prompt are dropped. No training,
no tuning; the same prompts in both domains.
Subset: SUBSET images per benchmark set (seeded). SAM is ~1-3 s per image on the RTX 3050.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image

from encode_full_scale import encode_batch, load_model
from src.detection_eval import agnostic_metrics, coco_metrics, naming

DET_ROOT, GT = "data/processed/detection", "data/processed/detection/gt"
PROP = f"{DET_ROOT}/sam_proposals"
SUBSET = {"dota_tiles": 120, "visdrone_ho_D1": 150, "visdrone_ho_D2": 150, "xview_val": 300, "uavid_tv": 150}
TASK_OF = {"dota_tiles": "D1", "visdrone_ho_D1": "D1", "xview_val": "D2", "visdrone_ho_D2": "D2", "uavid_tv": "D2"}
CLASSES = {"D1": ["small-vehicle", "large-vehicle"], "D2": ["car", "bus", "truck", "building"]}
BACKGROUND = ["background", "road", "tree", "grass", "ground"]
WORDS = {"small-vehicle": "small vehicle", "large-vehicle": "large vehicle"}
CAT_ID = {"small-vehicle": 1, "large-vehicle": 2, "car": 3, "bus": 4, "truck": 5, "building": 6}
MAX_PROPOSALS, MIN_SIDE, SEED = 300, 8, 20261131
Image.MAX_IMAGE_PIXELS = None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def subset_images(set_name):
    gt = json.load(open(f"{GT}/{set_name}.json"))
    base = set_name.replace("_D1", "").replace("_D2", "")  # VisDrone D1 and D2 use the same images
    ims = sorted(gt["images"], key=lambda im: im["file_name"])
    rng = random.Random(f"{SEED}-{base}")
    keep = sorted(rng.sample(ims, min(SUBSET[set_name], len(ims))), key=lambda im: im["id"])
    return gt, keep


def cmd_propose():
    from transformers import pipeline
    gen = pipeline("mask-generation", model="facebook/sam-vit-base", revision="70c1a07f894ebb5b307fd9eaaee97b9dfc16068f", device=0, points_per_batch=64)
    os.makedirs(PROP, exist_ok=True)
    done_files = set()
    for set_name in SUBSET:
        _, ims = subset_images(set_name)
        t0 = time.time()
        for k, im in enumerate(ims):
            out = f"{PROP}/{im['file_name'].replace('/', '__').replace('..', '')}.json"
            if os.path.exists(out) or out in done_files:
                continue
            img = Image.open(os.path.join(DET_ROOT, im["file_name"])).convert("RGB")
            with torch.no_grad():
                r = gen(img, pred_iou_thresh=0.88, stability_score_thresh=0.92, output_bboxes_mask=False)
            props = []
            for mask, s in zip(r["masks"], r["scores"]):
                ys, xs = np.where(np.asarray(mask))
                if ys.size == 0:
                    continue
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
                if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
                    continue
                props.append([x1, y1, x2, y2, float(s)])
            props = sorted(props, key=lambda p: -p[4])[:MAX_PROPOSALS]
            json.dump(props, open(out, "w"))
            done_files.add(out)
            if (k + 1) % 25 == 0:
                log(f"  {set_name}: {k + 1}/{len(ims)} images, {(k + 1) / (time.time() - t0):.2f} img/s, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        log(f"{set_name}: proposals done ({time.time() - t0:.0f} s)")


def text_embed(model_name, model, pre, words):
    import open_clip
    prompts = [f"a photo of a {w}" for w in words]
    with torch.no_grad():
        if model_name == "remoteclip":
            e = model.encode_text(open_clip.get_tokenizer("ViT-L-14")(prompts).to("cuda")).float()
        else:
            e = model.get_text_features(**pre(text=prompts, return_tensors="pt", padding=True).to("cuda")).float()
    return (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()


def cmd_name(model_name):
    model, pre = load_model(model_name, "cuda")
    for set_name in SUBSET:
        out_path = f"{DET_ROOT}/dets/sam_{model_name}__{set_name}.json"
        if os.path.exists(out_path):
            continue
        _, ims = subset_images(set_name)
        cls = CLASSES[TASK_OF[set_name]]
        text = text_embed(model_name, model, pre, [WORDS.get(c, c) for c in cls] + BACKGROUND)
        dets, t0 = [], time.time()
        for im in ims:
            props = json.load(open(f"{PROP}/{im['file_name'].replace('/', '__').replace('..', '')}.json"))
            if not props:
                continue
            img = Image.open(os.path.join(DET_ROOT, im["file_name"])).convert("RGB")
            crops = [img.crop(tuple(p[:4])) for p in props]
            f = np.concatenate([encode_batch(model_name, model, pre, crops[i:i + 64], "cuda") for i in range(0, len(crops), 64)])
            logits = 100 * f @ text.T
            prob = np.exp(logits - logits.max(1, keepdims=True))
            prob /= prob.sum(1, keepdims=True)
            best = prob.argmax(1)
            for p, b, pr in zip(props, best, prob):
                if b >= len(cls):
                    continue  # background wins
                dets.append({"image_id": im["id"], "category_id": CAT_ID[cls[b]],
                             "bbox": [p[0], p[1], p[2] - p[0], p[3] - p[1]], "score": round(float(pr[b]), 4)})
        json.dump(dets, open(out_path, "w"))
        log(f"sam_{model_name} {set_name}: {len(ims)} images, {len(dets)} detections ({time.time() - t0:.0f} s)")
    del model
    torch.cuda.empty_cache()


def subset_gt(gt, ims):
    ids = {im["id"] for im in ims}
    return dict(gt, images=ims, annotations=[a for a in gt["annotations"] if a["image_id"] in ids])


def proposal_recall(g, ims):
    """Share of (non-ignored) GT boxes covered at IoU >= 0.5 by any SAM proposal, before naming: the ceiling on 'found'."""
    names = {c["id"]: c["name"] for c in g["categories"]}
    props = {im["id"]: np.array(json.load(open(f"{PROP}/{im['file_name'].replace('/', '__').replace('..', '')}.json")) or np.zeros((0, 5)))
             for im in ims}
    hit, tot, sizes = {}, {}, {}
    for a in g["annotations"]:
        if a["iscrowd"]:
            continue
        c = names[a["category_id"]]
        x, y, w, h = a["bbox"]
        P = props[a["image_id"]]
        ok = False
        if len(P):
            ix = np.clip(np.minimum(x + w, P[:, 2]) - np.maximum(x, P[:, 0]), 0, None)
            iy = np.clip(np.minimum(y + h, P[:, 3]) - np.maximum(y, P[:, 1]), 0, None)
            inter = ix * iy
            ok = bool(((inter / (w * h + (P[:, 2] - P[:, 0]) * (P[:, 3] - P[:, 1]) - inter)) >= 0.5).any())
        hit[c] = hit.get(c, 0) + ok; tot[c] = tot.get(c, 0) + 1
        sizes.setdefault(c, []).append(min(w, h))
    out = {c: {"recall": hit[c] / tot[c], "n_gt": tot[c], "median_short_side_px": float(np.median(sizes[c]))} for c in tot}
    out["_proposals_per_image"] = float(np.mean([len(p) for p in props.values()]))
    out["_overall"] = sum(hit.values()) / max(1, sum(tot.values()))
    return out


def cmd_evaluate():
    res = {"subset_sizes": SUBSET, "background_prompts": BACKGROUND, "sets": {}, "proposal_recall": {}}
    for set_name in SUBSET:
        gt, ims = subset_images(set_name)
        g = subset_gt(gt, ims)
        ids = {im["id"] for im in ims}
        res["proposal_recall"][set_name] = pr = proposal_recall(g, ims)
        log(f"{set_name} SAM proposal recall@0.5 {pr['_overall']:.3f}, {pr['_proposals_per_image']:.0f} proposals/image")
        res["sets"][set_name] = {}
        for det in ("sam_clip", "sam_remoteclip", "owlv2", "gdino"):
            p = f"{DET_ROOT}/dets/{det}__{set_name}.json"
            if not os.path.exists(p):
                continue
            d = [x for x in json.load(open(p)) if x["image_id"] in ids]
            res["sets"][set_name][det] = {"class_aware": coco_metrics(g, d), "agnostic": agnostic_metrics(g, d), "naming": naming(g, d, min_score=0.1)}
            log(f"{set_name} {det}: mean AP50 {res['sets'][set_name][det]['class_aware']['mean']['AP50']:.3f}, "
                f"agnostic AP50 {res['sets'][set_name][det]['agnostic']['AP50']:.3f}, found {res['sets'][set_name][det]['naming']['overall_found_rate']:.3f}, "
                f"named {res['sets'][set_name][det]['naming']['overall_naming_accuracy']:.3f}")
    os.makedirs("results/detection", exist_ok=True)
    json.dump(res, open("results/detection/sam_pipeline_results.json", "w"), indent=1, default=float)
    names = {"sam_clip": "SAM + CLIP", "sam_remoteclip": "SAM + RemoteCLIP", "owlv2": "OWLv2", "gdino": "Grounding DINO"}
    pc = lambda v: "—" if v != v else f"{100 * v:.1f}"
    L = ["# SAM proposals + zero-shot naming (CLIP / RemoteCLIP) vs open-vocabulary detectors", "",
         f"Same image subset for every pipeline (sizes: {SUBSET}). AP50 per category; class-agnostic AP50; found = GT overlapped "
         "(IoU >= 0.5) by a detection with score >= 0.1; named = of found, share named correctly.", ""]
    for set_name, R in res["sets"].items():
        cats = [c for c in next(iter(R.values()))["class_aware"] if c != "mean"] if R else []
        L += [f"## {set_name}", "", "| Pipeline | " + " | ".join(f"AP50 {c}" for c in cats) + " | agnostic AP50 | found | named |", "|---|" + "---|" * (len(cats) + 3)]
        for det, r in R.items():
            L.append(f"| {names[det]} | " + " | ".join(pc(r["class_aware"][c]["AP50"]) for c in cats) +
                     f" | {pc(r['agnostic']['AP50'])} | {pc(r['naming']['overall_found_rate'])} | {pc(r['naming']['overall_naming_accuracy'])} |")
        pr = res["proposal_recall"][set_name]
        L += ["", f"SAM proposal recall (IoU ≥ 0.5, before naming; {pr['_proposals_per_image']:.0f} proposals/image): " +
              ", ".join(f"{c} {pc(v['recall'])}% (median short side {v['median_short_side_px']:.0f} px)" for c, v in pr.items() if not c.startswith("_")), ""]
    open("results/detection/sam_pipeline_results.md", "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["propose", "name", "evaluate"])
    ap.add_argument("--model", choices=["clip", "remoteclip"])
    a = ap.parse_args()
    {"propose": cmd_propose, "evaluate": cmd_evaluate}.get(a.cmd, lambda: cmd_name(a.model))()
