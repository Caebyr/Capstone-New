"""Full-dataset box pools for the November 2026 full-scale experiments.

The paper's runs (src/dataset.py) cut every crop into memory and then sampled 300 per class. At full
scale that does not fit in RAM, so these functions list *boxes only* (no pixels): one record per
ground-truth object that passes exactly the same filters as src/dataset.py. Pixels are cut later,
one source image at a time, by the encoder (encode_full_scale.py).

Filters reproduced from src/dataset.py (do not change without discussion):
  DOTA      oriented box -> axis-aligned (int min/max), clip to image, keep if both sides >= 16
  VisDrone  skip class 0 (ignored region) and unknown ids; keep if annotated w, h >= 16 (BEFORE
            clipping, as in get_visdrone_crops); then clip to image
  xView     clip to chip, keep if both sides >= 16
  UAVid     connected components of the class colour, area >= 800 px, (x2 - x1) and (y2 - y1) >= 16
            with x2/y2 the max pixel index (crop box excludes that last row/column, as before)

Records carry the raw dataset class, so each experiment applies its own label mapping.
"""
import csv
import gzip
import os
import random

import numpy as np

from src.dataset import MIN_CROP_SIZE, VISDRONE_CLASSES, parse_dota_label, split_visdrone_images

DOTA_IMG_DIRS = ["data/raw/DOTA/images", "data/raw/DOTA/images_part2_part3"]  # part1 kept separate: the paper's replay scans only images/
DOTA_LABEL_DIR = "data/raw/DOTA/labelTxt"
VISDRONE_BASE = "data/raw/VisDrone/VisDrone2019-DET-train"
UAVID_BASE = "data/raw/UAVid/modified_uavid_dataset"
XVIEW_REPO, XVIEW_REVISION = "HichTala/xview", "00a1b977dc16f6eda77cce0236ca2c2a13ac840b"
HOLDOUT_FRACTION, SPLIT_SEED = 0.2, 1234  # identical to the paper's VisDrone split

UAVID_COLOURS = {"building": [(128, 0, 0)], "car": [(64, 0, 128), (192, 0, 192)]}  # moving + static car merged

FIELDS = ["box_id", "dataset", "source_image", "image_ref", "sequence", "raw_class", "x1", "y1", "x2", "y2",
          "width", "height", "img_w", "img_h", "visdrone_split", "extra"]


def _record(**kw):
    r = {k: "" for k in FIELDS}
    r.update(kw)
    r["width"], r["height"] = r["x2"] - r["x1"], r["y2"] - r["y1"]
    return r


def list_dota_boxes(classes=("small-vehicle", "large-vehicle")):
    from PIL import Image
    out = []
    for d in DOTA_IMG_DIRS:
        for img_name in sorted(os.listdir(d)):
            label_path = os.path.join(DOTA_LABEL_DIR, img_name.replace(".png", ".txt"))
            if not os.path.exists(label_path):
                continue
            with Image.open(os.path.join(d, img_name)) as im:  # header only, no pixel decode
                W, H = im.size
            meta = {}
            with open(label_path) as f:
                for line in f:
                    if ":" in line and len(line.split()) == 1:
                        k, v = line.strip().split(":", 1)
                        meta[k] = v
            difficult = _dota_difficult(label_path)
            for k, (x1, y1, x2, y2, cls) in enumerate(parse_dota_label(label_path)):
                if cls not in classes:
                    continue
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                    continue
                out.append(_record(dataset="DOTA", source_image=img_name, image_ref=os.path.join(d, img_name),
                                   raw_class=cls, x1=x1, y1=y1, x2=x2, y2=y2, img_w=W, img_h=H,
                                   extra=f"gsd={meta.get('gsd', '')};source={meta.get('imagesource', '')};difficult={difficult[k]}"))
    return out


def _dota_difficult(label_path):
    """Difficult flag per parsed box, aligned with parse_dota_label's output order."""
    flags = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            try:
                list(map(float, parts[:8]))
            except ValueError:
                continue
            flags.append(parts[9] if len(parts) > 9 else "")
    return flags


def list_visdrone_boxes(classes=("car", "van", "truck", "bus")):
    from PIL import Image
    img_dir, ann_dir = os.path.join(VISDRONE_BASE, "images"), os.path.join(VISDRONE_BASE, "annotations")
    _, holdout = split_visdrone_images(ann_dir, HOLDOUT_FRACTION, SPLIT_SEED)
    out = []
    for ann_file in sorted(os.listdir(ann_dir)):
        img_name = ann_file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue
        with Image.open(img_path) as im:
            W, H = im.size
        split = "heldout" if ann_file in holdout else "train"
        with open(os.path.join(ann_dir, ann_file)) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                try:
                    x, y, w, h, cls_id = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[5])
                except ValueError:
                    continue
                if cls_id == 0 or cls_id not in VISDRONE_CLASSES or w < MIN_CROP_SIZE or h < MIN_CROP_SIZE:
                    continue
                cls = VISDRONE_CLASSES[cls_id]
                if cls not in classes:
                    continue
                x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
                out.append(_record(dataset="VisDrone", source_image=img_name, image_ref=img_path,
                                   sequence=img_name.split("_")[0], raw_class=cls, x1=x1, y1=y1, x2=x2, y2=y2,
                                   img_w=W, img_h=H, visdrone_split=split))
    return out


def xview_parquet_files(split="train"):
    from huggingface_hub import snapshot_download
    root = snapshot_download(XVIEW_REPO, repo_type="dataset", revision=XVIEW_REVISION, allow_patterns=["data/*"])
    files = sorted(os.path.join(root, "data", f) for f in os.listdir(os.path.join(root, "data")) if f.startswith(split + "-"))
    return files


def iter_xview(split="train", columns=None):
    """Yields (row_index, record dict) over the local, revision-pinned parquet files, in file order."""
    import pyarrow.parquet as pq
    i = 0
    for f in xview_parquet_files(split):
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=256, columns=columns):
            for row in b.to_pylist():
                yield i, row
                i += 1


def xview_category_names():
    import json
    import pyarrow.parquet as pq
    meta = pq.ParquetFile(xview_parquet_files("train")[0]).schema_arrow.metadata
    feats = json.loads(meta[b"huggingface"])["info"]["features"]
    return feats["objects"]["feature"]["category"]["names"]


def list_xview_boxes(class_names, split="train"):
    """class_names: raw xView category names to keep (e.g. keys of XVIEW_TO_EXPANDED)."""
    import io
    from PIL import Image
    names = xview_category_names()
    keep = set(class_names)
    out = []
    for i, row in iter_xview(split, columns=["image_id", "image", "objects"]):
        with Image.open(io.BytesIO(row["image"]["bytes"])) as im:
            W, H = im.size
        objs = row["objects"]
        for cat_id, (x, y, w, h) in zip(objs["category"], objs["bbox"]):
            cls = names[cat_id]
            if cls not in keep:
                continue
            x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
            if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                continue
            out.append(_record(dataset="xView", source_image=f"xview_chip_{row['image_id']}", image_ref=f"{split}:{i}",
                               raw_class=cls, x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2), img_w=W, img_h=H))
    return out


def list_uavid_boxes(classes=("building", "car"), splits=("train_data", "val_data"), min_area=800):
    from PIL import Image
    from scipy import ndimage
    out = []
    for split in splits:
        label_dir, img_dir = os.path.join(UAVID_BASE, split, "Labels"), os.path.join(UAVID_BASE, split, "Images")
        for label_name in sorted(os.listdir(label_dir)):
            img_path = os.path.join(img_dir, label_name)
            if not os.path.exists(img_path):
                continue
            arr = np.array(Image.open(os.path.join(label_dir, label_name)).convert("RGB"))
            H, W = arr.shape[:2]
            for cls in classes:
                mask = np.zeros((H, W), bool)
                for colour in UAVID_COLOURS[cls]:
                    mask |= np.all(arr == np.array(colour), axis=-1)
                if not mask.any():
                    continue
                labeled, n = ndimage.label(mask)
                for sl, comp in zip(ndimage.find_objects(labeled), range(1, n + 1)):
                    area = int((labeled[sl] == comp).sum())
                    if area < min_area:
                        continue
                    y1, y2 = sl[0].start, sl[0].stop - 1  # max pixel index, as in get_uavid_crops
                    x1, x2 = sl[1].start, sl[1].stop - 1
                    if (x2 - x1) < MIN_CROP_SIZE or (y2 - y1) < MIN_CROP_SIZE:
                        continue
                    out.append(_record(dataset="UAVid", source_image=f"{split}/{label_name}", image_ref=img_path,
                                       sequence=label_name.split("_")[0], raw_class=cls, x1=x1, y1=y1, x2=x2, y2=y2,
                                       img_w=W, img_h=H, extra=f"area={area}"))
    return out


def write_pool(path, records, prefix):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for i, r in enumerate(records):
            r["box_id"] = f"{prefix}:{i}"
            w.writerow(r)


def read_pool(path):
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sample_rows(rows, key, cap, seed):
    """Up to `cap` rows per value of key(row), uniformly at random; returns rows in a stable order."""
    rng = random.Random(seed)
    groups = {}
    for r in rows:
        k = key(r)
        if k is not None:
            groups.setdefault(k, []).append(r)
    out = []
    for k in sorted(groups):
        g = groups[k]
        out.extend(g if len(g) <= cap else rng.sample(g, cap))
    return out
