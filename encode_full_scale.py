"""
Checkpoint A2 of the full-scale work: sample crops from the full box pools and embed them.

  python encode_full_scale.py sample                 # writes results/full_scale/encode_manifest.csv.gz
  python encode_full_scale.py encode --model clip    # then remoteclip, then finetuned (one GPU model at a time)

Sampling: up to CAP crops per (list, category), uniformly at random from the full pool (seed SEED).
A random 300-subset of these is a random 300-subset of the population, so the paper's protocol can be
repeated over many seeds from this one cache (drift_full_scale.py).

Lists (overhead | UAV), labels after mapping:
  P1   DOTA all images            | VisDrone held-out 20% images (paper split)   small-/large-vehicle   clip, remoteclip, finetuned
  P1a  -                          | VisDrone all images                          small-/large-vehicle   clip, remoteclip
  P2   xView all train chips      | VisDrone all images                          car, bus, truck, van   clip, remoteclip
  P3   xView all train chips      | UAVid train+val                              building               clip, remoteclip
  A4   (same-domain baseline)     | UAVid train+val                              car                    clip, remoteclip
       DOTA vs xView reuses the P1/P2 overhead crops.

Encoding: same loaders and preprocessing as the paper (src/embeddings.py: CLIPProcessor / OpenCLIP
preprocess, unit-norm image embeddings, fp32, batch 4, no_grad). Each source image is decoded once
per model. Output is written in shards of SHARD rows, so an interrupted run resumes where it stopped.
JPEG sources decode with the installed Pillow (libjpeg-turbo), not the August build (IJG libjpeg 9):
the paper's cached embeddings are not used or overwritten here.
"""
import argparse
import csv
import glob
import gzip
import io
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

from src.dataset import XVIEW_TO_EXPANDED
from src.full_scale import iter_xview, read_pool, sample_rows

POOLS = "results/full_scale/pools"
MANIFEST = "results/full_scale/encode_manifest.csv.gz"
EMB = "embeddings/full_scale"
CAP, SEED, SHARD = 5000, 20261125, 2000
REMOTECLIP_CKPT = "models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt"
FINETUNED_CKPT = "models/finetuned_clip_visdrone_heldout/finetuned_clip_visdrone_heldout_epoch3.pt"
P1_MAP = {"small-vehicle": "small-vehicle", "large-vehicle": "large-vehicle",
          "car": "small-vehicle", "van": "large-vehicle", "truck": "large-vehicle", "bus": "large-vehicle"}

Image.MAX_IMAGE_PIXELS = None  # local, trusted DOTA scenes can exceed Pillow's default bomb limit


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def define_lists(pools):
    """list name -> (rows, label function, models)"""
    d, v, x, u = pools["dota"], pools["visdrone"], pools["xview"], pools["uavid"]
    vh = [r for r in v if r["visdrone_split"] == "heldout"]
    xv_label = lambda r: XVIEW_TO_EXPANDED.get(r["raw_class"])
    return {
        "P1_dota": (d, lambda r: r["raw_class"], ["clip", "remoteclip", "finetuned"]),
        "P1_visdrone_heldout": (vh, lambda r: P1_MAP[r["raw_class"]], ["clip", "remoteclip", "finetuned"]),
        "P1a_visdrone_all": (v, lambda r: P1_MAP[r["raw_class"]], ["clip", "remoteclip"]),
        "P2_xview": (x, lambda r: xv_label(r) if xv_label(r) != "building" else None, ["clip", "remoteclip"]),
        "P2_visdrone": (v, lambda r: r["raw_class"], ["clip", "remoteclip"]),
        "P3_xview_building": (x, lambda r: "building" if r["raw_class"] == "Building" else None, ["clip", "remoteclip"]),
        "P3_uavid_building": (u, lambda r: "building" if r["raw_class"] == "building" else None, ["clip", "remoteclip"]),
        "A4_uavid_car": (u, lambda r: "car" if r["raw_class"] == "car" else None, ["clip", "remoteclip"]),
    }


def cmd_sample():
    pools = {k: read_pool(f"{POOLS}/{k}_boxes.csv.gz") for k in ("dota", "visdrone", "xview", "uavid")}
    order = {}  # box_id -> encode row
    rows_out, membership = [], []
    for li, (name, (rows, lab, models)) in enumerate(define_lists(pools).items()):
        picked = sample_rows(rows, lab, CAP, SEED + li)
        for r in picked:
            bid = r["box_id"]
            if bid not in order:
                order[bid] = len(rows_out)
                rows_out.append(dict(r, models=set()))
            rows_out[order[bid]]["models"].update(models)
            membership.append({"list": name, "label": lab(r), "box_id": bid})
        log(f"{name}: {len(picked)} crops {dict(sorted(__import__('collections').Counter(lab(r) for r in picked).items()))}")
    # encode order: dataset, then source image in pool order, so each image is decoded once
    ds_rank = {"DOTA": 0, "VisDrone": 1, "xView": 2, "UAVid": 3}
    def img_key(r):
        ref = r["image_ref"]
        return (ds_rank[r["dataset"]], int(ref.split(":")[1]) if r["dataset"] == "xView" else ref, int(r["box_id"].split(":")[1]))
    rows_out.sort(key=img_key)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    fields = list(rows_out[0].keys())
    with gzip.open(MANIFEST, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["enc_row"] + fields)
        w.writeheader()
        for i, r in enumerate(rows_out):
            w.writerow(dict(r, enc_row=i, models="|".join(sorted(r["models"]))))
    with gzip.open("results/full_scale/list_membership.csv.gz", "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["list", "label", "box_id"])
        w.writeheader()
        w.writerows(membership)
    log(f"encode manifest: {len(rows_out)} unique crops; per model: "
        + ", ".join(f"{m}={sum(m in r['models'] for r in rows_out)}" for m in ("clip", "remoteclip", "finetuned")))


def load_model(name, device):
    from src.embeddings import load_clip, load_remoteclip
    if name == "remoteclip":
        return load_remoteclip(REMOTECLIP_CKPT, device)
    model, proc = load_clip(device)
    if name == "finetuned":
        model.load_state_dict(torch.load(FINETUNED_CKPT, map_location=device))
        model.eval()
    return model, proc


def encode_batch(name, model, pre, crops, device):
    from src.embeddings import encode_with_clip, encode_with_remoteclip
    pairs = [(c, "") for c in crops]
    if name == "remoteclip":
        f, _ = encode_with_remoteclip(pairs, model, pre, device, batch_size=4, log=None)
    else:
        f, _ = encode_with_clip(pairs, model, pre, device, batch_size=4, log=None)
    return f.astype(np.float32)


def cmd_encode(name):
    device = "cuda"
    with gzip.open(MANIFEST, "rt", newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if name in r["models"].split("|")]
    out_dir = f"{EMB}/{name}"
    os.makedirs(out_dir, exist_ok=True)
    n_shards = (len(rows) + SHARD - 1) // SHARD
    todo = [s for s in range(n_shards) if not os.path.exists(f"{out_dir}/shard_{s:04d}.npy")]
    log(f"{name}: {len(rows)} crops in {n_shards} shards; {len(todo)} to do")
    if not todo:
        return finalize(name, rows, out_dir, n_shards)
    model, pre = load_model(name, device)
    xview_cache = {}
    if any(r["dataset"] == "xView" for s in todo for r in rows[s * SHARD:(s + 1) * SHARD]):
        need = {int(r["image_ref"].split(":")[1]) for s in todo for r in rows[s * SHARD:(s + 1) * SHARD] if r["dataset"] == "xView"}
        for i, row in iter_xview("train", columns=["image"]):
            if i in need:
                xview_cache[i] = row["image"]["bytes"]
        log(f"  loaded {len(xview_cache)} xView chips")
    t0, done = time.time(), 0
    cur_ref, cur_img = None, None
    for s in todo:
        chunk = rows[s * SHARD:(s + 1) * SHARD]
        feats, buf = [], []
        for r in chunk:
            if r["image_ref"] != cur_ref:
                cur_ref = r["image_ref"]
                if r["dataset"] == "xView":
                    cur_img = Image.open(io.BytesIO(xview_cache[int(cur_ref.split(":")[1])])).convert("RGB")
                else:
                    cur_img = Image.open(cur_ref).convert("RGB")
            buf.append(cur_img.crop((int(r["x1"]), int(r["y1"]), int(r["x2"]), int(r["y2"]))))
            if len(buf) == 64:
                feats.append(encode_batch(name, model, pre, buf, device)); buf = []
        if buf:
            feats.append(encode_batch(name, model, pre, buf, device))
        np.save(f"{out_dir}/shard_{s:04d}.npy", np.concatenate(feats))
        done += len(chunk)
        el = time.time() - t0
        log(f"  {name} shard {s + 1}/{n_shards} | {done} crops this run, {done / el:.1f} crops/s, "
            f"ETA {(sum(len(rows[t * SHARD:(t + 1) * SHARD]) for t in todo) - done) / max(done / el, 1e-9) / 60:.0f} min | "
            f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    del model
    torch.cuda.empty_cache()
    finalize(name, rows, out_dir, n_shards)


def finalize(name, rows, out_dir, n_shards):
    feats = np.concatenate([np.load(f"{out_dir}/shard_{s:04d}.npy") for s in range(n_shards)])
    assert len(feats) == len(rows), (len(feats), len(rows))
    norms = np.linalg.norm(feats, axis=1)
    assert np.allclose(norms, 1, atol=1e-3), f"non-unit embeddings: {norms.min()}..{norms.max()}"
    np.save(f"{EMB}/{name}_feats.npy", feats)
    with open(f"{EMB}/{name}_box_ids.txt", "w") as f:
        f.write("\n".join(r["box_id"] for r in rows) + "\n")
    log(f"{name}: wrote {EMB}/{name}_feats.npy {feats.shape} (+ box_ids)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "encode"])
    ap.add_argument("--model", choices=["clip", "remoteclip", "finetuned"])
    a = ap.parse_args()
    if a.cmd == "sample":
        cmd_sample()
    else:
        if not a.model:
            sys.exit("--model is required for encode")
        cmd_encode(a.model)
