"""
Checkpoint A5: two controls for the paper's interpretation.

(a) Resolution control  -- is the overhead/UAV gap mostly a resolution effect?
    Each UAV crop is shrunk so that its longer side equals the longer side of a randomly drawn overhead crop of
    the same category (aspect ratio kept; PIL bicubic with antialiasing), then encoded with the paper's
    preprocessing (which upsamples it back to 224 like any small overhead crop). Compared on the same crops:
      overhead vs UAV-original   and   overhead vs UAV-shrunk
    for drift (cosine, ratio) and zero-shot crop classification ("a photo of a {}").
    python controls_a5.py resample --model clip|remoteclip|finetuned     (GPU)
(b) Prompt control -- does the classification result depend on the one prompt the paper used?
    Templates: the paper's "a photo of a {}", "a satellite photo of a {}", "an aerial photo of a {}",
    "a drone photo of a {}", "{}" and the mean of all five (prompt ensemble). Same crops, same rule.
    python controls_a5.py prompts                                         (GPU, text towers only)
Then: python controls_a5.py analyze  ->  results/full_scale/controls_a5.json / .md
"""
import argparse
import csv
import gzip
import io
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from classify_crops import class_metrics, classify, text_metrics
from drift_full_scale import TASKS, build, load_cache
from encode_full_scale import encode_batch, load_model
from src.dataset import MIN_CROP_SIZE
from src.full_scale import iter_xview
from src.stats import drift

OUT, EMB = "results/full_scale", "embeddings/full_scale"
PER_CAT, SEED = 2000, 20261130
TEMPLATES = ["a photo of a {}", "a satellite photo of a {}", "an aerial photo of a {}", "a drone photo of a {}", "{}"]
WORDS = {"small-vehicle": "small vehicle", "large-vehicle": "large vehicle"}
Image.MAX_IMAGE_PIXELS = None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def plan(man, member):
    """For P1 and P2+P3: pick PER_CAT UAV crops per category and an overhead size donor for each."""
    rng = np.random.default_rng(SEED)
    rows = []
    for task in ("P1", "P2+P3"):
        spec = TASKS[task]
        for c in spec["cats"]:
            def ids(domain):
                out = []
                for list_name, cats in spec[domain]:
                    out += [r["box_id"] for r in member if r["list"] == list_name and r["label"] == c and (cats is None or c in cats)]
                return list(dict.fromkeys(out))
            uav, over = ids("uav"), ids("overhead")
            pick = rng.choice(len(uav), min(PER_CAT, len(uav)), replace=False)
            donors = rng.choice(len(over), len(pick), replace=True)
            for i, j in zip(pick, donors):
                u, o = man[uav[i]], man[over[j]]
                uw, uh = int(u["width"]), int(u["height"])
                target = max(int(o["width"]), int(o["height"]))
                # never upsample; keep the short side >= MIN_CROP_SIZE (16 px), the floor every real crop meets
                s = min(1.0, max(target / max(uw, uh), MIN_CROP_SIZE / min(uw, uh)))
                rows.append({"task": task, "label": c, "box_id": uav[i], "donor": over[j],
                             "new_w": max(1, round(uw * s)), "new_h": max(1, round(uh * s)), "scale": s})
    return rows


def cmd_resample(model_name):
    man, member, _ = load_cache()
    p = f"{OUT}/controls_resample_plan.csv.gz"
    if not os.path.exists(p):
        rows = plan(man, member)
        with gzip.open(p, "wt", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    with gzip.open(p, "rt", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if model_name == "finetuned":
        rows = [r for r in rows if r["task"] == "P1"]
    out = f"{EMB}/controls_shrunk_{model_name}.npy"
    if os.path.exists(out):
        log(f"{out} exists"); return
    model, pre = load_model(model_name, "cuda")
    need_x = {int(man[r["box_id"]]["image_ref"].split(":")[1]) for r in rows if man[r["box_id"]]["dataset"] == "xView"}
    xcache = {i: row["image"]["bytes"] for i, row in iter_xview("train", columns=["image"]) if i in need_x} if need_x else {}
    order = sorted(range(len(rows)), key=lambda k: man[rows[k]["box_id"]]["image_ref"])
    feats = np.zeros((len(rows), 0), np.float32)
    res, buf, idx, cur_ref, cur = [None] * len(rows), [], [], None, None
    t0 = time.time()
    for n, k in enumerate(order):
        r = rows[k]; b = man[r["box_id"]]
        if b["image_ref"] != cur_ref:
            cur_ref = b["image_ref"]
            cur = Image.open(io.BytesIO(xcache[int(cur_ref.split(":")[1])])).convert("RGB") if b["dataset"] == "xView" else Image.open(cur_ref).convert("RGB")
        crop = cur.crop((int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"]))).resize((int(r["new_w"]), int(r["new_h"])), Image.BICUBIC)
        buf.append(crop); idx.append(k)
        if len(buf) == 64 or n == len(order) - 1:
            f = encode_batch(model_name, model, pre, buf, "cuda")
            for kk, ff in zip(idx, f):
                res[kk] = ff
            buf, idx = [], []
        if (n + 1) % 2000 == 0:
            log(f"  {model_name}: {n + 1}/{len(rows)} shrunk crops, {(n + 1) / (time.time() - t0):.1f} crops/s")
    np.save(out, np.stack(res))
    log(f"wrote {out}")


def text_bank(device="cuda"):
    """{model: {task: {template_or_'ensemble': [K, D]}}} unit-norm text embeddings."""
    path = f"{OUT}/prompt_text_embeddings.npz"
    if os.path.exists(path):
        z = np.load(path)
        bank = {}
        for k in z.files:
            m, t, tp = k.split("||")
            bank.setdefault(m, {}).setdefault(t, {})[tp] = z[k]
        return bank
    import open_clip
    from src.embeddings import load_clip, load_remoteclip
    bank = {}
    for m in ("clip", "finetuned", "remoteclip"):
        if m == "remoteclip":
            model, _ = load_remoteclip("models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt", device)
            tok = open_clip.get_tokenizer("ViT-L-14")
            enc = lambda texts: model.encode_text(tok(texts).to(device)).float()
        else:
            model, proc = load_clip(device)
            if m == "finetuned":
                model.load_state_dict(torch.load("models/finetuned_clip_visdrone_heldout/finetuned_clip_visdrone_heldout_epoch3.pt", map_location=device))
            enc = lambda texts: model.get_text_features(**proc(text=texts, return_tensors="pt", padding=True).to(device)).float()
        bank[m] = {}
        for t in ("P1", "P2+P3"):
            bank[m][t] = {}
            with torch.no_grad():
                for tp in TEMPLATES:
                    e = enc([tp.format(WORDS.get(c, c)) for c in TASKS[t]["cats"]])
                    bank[m][t][tp] = (e / e.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float64)
            ens = np.mean([bank[m][t][tp] for tp in TEMPLATES], 0)
            bank[m][t]["ensemble"] = ens / np.linalg.norm(ens, axis=1, keepdims=True)
        del model
        torch.cuda.empty_cache()
    np.savez(path, **{f"{m}||{t}||{tp}": v for m, d in bank.items() for t, dd in d.items() for tp, v in dd.items()})
    return bank


def accuracy_block(x, text, cats):
    _, pred = classify(x["feats"], text, cats)
    cm = class_metrics(x["labels"], pred, cats)
    tm = text_metrics(x["feats"], x["labels"], text, cats)
    return {"accuracy": cm["accuracy"], "recall": {c: cm["per_category"][c]["recall"] for c in cats},
            "pred_share": {c: float(np.mean(pred == c)) for c in cats}, "margin": {c: tm[c]["margin"] for c in cats}}


def cmd_analyze():
    man, member, feats = load_cache()
    bank = text_bank()
    with gzip.open(f"{OUT}/controls_resample_plan.csv.gz", "rt", newline="", encoding="utf-8") as f:
        plan_rows = list(csv.DictReader(f))
    res = {"settings": {"per_cat": PER_CAT, "seed": SEED, "templates": TEMPLATES}, "resolution": {}, "prompts": {}}
    for task in ("P1", "P2+P3"):
        spec, cats = TASKS[task], TASKS[task]["cats"]
        data = build(task, man, member, feats)
        res["resolution"][task], res["prompts"][task] = {}, {}
        rows = [r for r in plan_rows if r["task"] == task]
        res["resolution"][task]["median_scale"] = {c: float(np.median([float(r["scale"]) for r in rows if r["label"] == c])) for c in cats}
        for m in spec["models"]:
            path = f"{EMB}/controls_shrunk_{m}.npy"
            if not os.path.exists(path):
                continue
            allrows = [r for r in plan_rows if m != "finetuned" or r["task"] == "P1"]
            shr = np.load(path).astype(np.float64)
            pos = [k for k, r in enumerate(allrows) if r["task"] == task]
            F, index = feats[m]
            uo = {"feats": F[[index[rows_i["box_id"]] for rows_i in rows]], "labels": np.array([r["label"] for r in rows])}
            us = {"feats": shr[pos], "labels": uo["labels"]}
            ov = data[m]["overhead"]
            text = bank[m][task]["a photo of a {}"]
            r = {"drift": {}, "classification": {}}
            for c in cats:
                mo, mu = ov["labels"] == c, uo["labels"] == c
                d_orig = drift(ov["feats"][mo], np.ones(mo.sum()), uo["feats"][mu], np.ones(mu.sum()))
                d_shr = drift(ov["feats"][mo], np.ones(mo.sum()), us["feats"][mu], np.ones(mu.sum()))
                d_self = drift(uo["feats"][mu], np.ones(mu.sum()), us["feats"][mu], np.ones(mu.sum()))
                r["drift"][c] = {"overhead_vs_uav": d_orig, "overhead_vs_uav_shrunk": d_shr, "uav_vs_uav_shrunk": d_self}
            r["classification"] = {"overhead": accuracy_block(ov, text, cats), "uav_original": accuracy_block(uo, text, cats),
                                   "uav_shrunk": accuracy_block(us, text, cats)}
            res["resolution"][task][m] = r
            log(f"resolution {task} {m}: acc overhead {r['classification']['overhead']['accuracy']:.3f}, UAV {r['classification']['uav_original']['accuracy']:.3f}, "
                f"UAV shrunk {r['classification']['uav_shrunk']['accuracy']:.3f}")
        for m in spec["models"]:
            res["prompts"][task][m] = {}
            for tp, text in bank[m][task].items():
                res["prompts"][task][m][tp] = {d: accuracy_block(data[m][d], text, cats) for d in ("overhead", "uav")}
    with open(f"{OUT}/controls_a5.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    write_md(res)
    log("analysis written")


def write_md(res):
    names = {"clip": "CLIP", "remoteclip": "RemoteCLIP", "finetuned": "fine-tuned CLIP"}
    pc = lambda v: f"{100 * v:.1f}"
    L = ["# Controls: resolution matching and prompt sensitivity", "",
         f"## (a) Resolution control", "",
         f"{res['settings']['per_cat']} UAV crops per category, each shrunk to the size of a random overhead crop of the same category, then encoded "
         "with the same preprocessing. Drift = cosine drift; ratio = spread-normalized ratio. Accuracy = zero-shot crop classification, \"a photo of a {}\".", ""]
    for task, T in res["resolution"].items():
        L += [f"### {task} (median shrink factor: " + ", ".join(f"{c} {v:.2f}" for c, v in T["median_scale"].items()) + ")", "",
              "| Model | Category | cos O vs U | cos O vs U-shrunk | ratio O vs U | ratio O vs U-shrunk | cos U vs U-shrunk |", "|---|---|---|---|---|---|---|"]
        for m, r in T.items():
            if m == "median_scale":
                continue
            for c, d in r["drift"].items():
                L.append(f"| {names[m]} | {c} | {d['overhead_vs_uav']['cos']:.4f} | {d['overhead_vs_uav_shrunk']['cos']:.4f} | {d['overhead_vs_uav']['ratio']:.3f} | "
                         f"{d['overhead_vs_uav_shrunk']['ratio']:.3f} | {d['uav_vs_uav_shrunk']['cos']:.4f} |")
        L += ["", "| Model | Acc overhead | Acc UAV original | Acc UAV shrunk | UAV-shrunk predictions |", "|---|---|---|---|---|"]
        for m, r in T.items():
            if m == "median_scale":
                continue
            c = r["classification"]
            L.append(f"| {names[m]} | {pc(c['overhead']['accuracy'])} | {pc(c['uav_original']['accuracy'])} | {pc(c['uav_shrunk']['accuracy'])} | "
                     + ", ".join(f"{k} {pc(v)}%" for k, v in c["uav_shrunk"]["pred_share"].items()) + " |")
        L.append("")
    L += ["## (b) Prompt sensitivity", "", "Accuracy overhead / UAV (%), full cache, per prompt template; last column = share of overhead *vehicle* crops predicted as each class for the ensemble.", ""]
    for task, T in res["prompts"].items():
        L += [f"### {task}", "", "| Model | " + " | ".join(f"`{tp}`" for tp in res["settings"]["templates"]) + " | ensemble |", "|---|" + "---|" * (len(res["settings"]["templates"]) + 1)]
        for m, P in T.items():
            L.append(f"| {names[m]} | " + " | ".join(f"{pc(P[tp]['overhead']['accuracy'])} / {pc(P[tp]['uav']['accuracy'])}" for tp in res["settings"]["templates"] + ["ensemble"]) + " |")
        L.append("")
    with open(f"{OUT}/controls_a5.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["resample", "prompts", "analyze"])
    ap.add_argument("--model", choices=["clip", "remoteclip", "finetuned"])
    a = ap.parse_args()
    if a.cmd == "resample":
        cmd_resample(a.model)
    elif a.cmd == "prompts":
        text_bank()
    else:
        cmd_analyze()
