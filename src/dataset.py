import os
import random

DOTA_TO_COMMON = {
    "small-vehicle": "small-vehicle",
    "large-vehicle": "large-vehicle",
}

VISDRONE_CLASSES = {
    1: "pedestrian", 2: "person", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor",
}

# Only classes with real DOTA/VisDrone counterparts (PROJECT_CONTEXT.md Section 5 / 14a) --
# DOTA has no person/motorcycle equivalent, so those are intentionally excluded here.
# NOTE: bicycle is deliberately excluded from small-vehicle (2026-08-19 fix) -- DOTA's
# "small-vehicle" label is satellite-altitude, which can only resolve cars, not bicycles.
# Including bicycle mixed two visually distinct object types into one bucket (~5.8% of the
# bucket was bicycles), adding noise to the small-vehicle drift measurement.
VISDRONE_TO_COMMON = {
    "car": "small-vehicle",
    "van": "large-vehicle",
    "truck": "large-vehicle",
    "bus": "large-vehicle",
}

COMMON_VOCAB = ["small-vehicle", "large-vehicle"]

MIN_CROP_SIZE = 16

# --- Expanded vocabulary: xView <-> VisDrone (separate from COMMON_VOCAB above) ---
# xView's fine-grained vehicle taxonomy lets car/bus/truck/van map 1:1 onto VisDrone's own
# categories instead of being lumped into small-vehicle/large-vehicle. DOTA cannot use this
# scheme -- it only has the coarse small-vehicle/large-vehicle split, no finer subtypes -- so
# this is for xView<->VisDrone comparisons specifically, not a replacement for COMMON_VOCAB.
# xView has no motorcycle or person/pedestrian class (satellite altitude can't resolve them),
# so those VisDrone categories are excluded here too, same reasoning as COMMON_VOCAB above.
# "building" added 2026-08-20: xView has a real "Building" class, and UAVid (a UAV semantic
# segmentation dataset) has one too -- pairing them adds a genuinely new shared category beyond
# vehicles, and buildings are far more common per-image than any single vehicle type (unlike
# "van", which is bottlenecked by how rare xView's closest match, Pickup Truck, actually is).
EXPANDED_VOCAB = ["car", "bus", "truck", "van", "building"]

XVIEW_TO_EXPANDED = {
    "Small Car": "car",
    "Passenger Vehicle": "car",
    "Bus": "bus",
    "Pickup Truck": "van",
    "Truck": "truck",
    "Cargo Truck": "truck",
    "Truck w/Box": "truck",
    "Truck w/Flatbed": "truck",
    "Truck w/Liquid": "truck",
    "Truck Tractor": "truck",
    "Utility Truck": "truck",
    "Dump Truck": "truck",
    "Haul Truck": "truck",
    "Building": "building",
}

VISDRONE_TO_EXPANDED = {
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "van": "van",
}

# UAVid ground-truth mask RGB colors -> expanded vocabulary. UAVid provides per-pixel semantic
# masks, not bounding boxes, so crops are derived via connected-component analysis on the mask
# (see get_uavid_crops). Only "building" is mapped here -- UAVid's car classes are redundant
# with what VisDrone already gives us, and UAVid has no bus/truck/van class at all.
UAVID_TO_EXPANDED = {
    (128, 0, 0): "building",  # Building
}


def parse_dota_label(label_path):
    """Returns list of (x1, y1, x2, y2, classname) from a DOTA OBB label file."""
    boxes = []
    with open(label_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        try:
            coords = list(map(float, parts[:8]))
        except ValueError:
            continue
        cls = parts[8]
        xs, ys = coords[0::2], coords[1::2]
        boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)), cls))
    return boxes


def get_dota_crops(img_dir, label_dir, class_map=DOTA_TO_COMMON, min_crop_size=MIN_CROP_SIZE):
    """Ground-truth object crops from DOTA, filtered/mapped to the shared vocabulary."""
    from PIL import Image

    crops = []
    for img_name in sorted(os.listdir(img_dir)):
        label_path = os.path.join(label_dir, img_name.replace(".png", ".txt"))
        if not os.path.exists(label_path):
            continue
        image = Image.open(os.path.join(img_dir, img_name)).convert("RGB")
        W, H = image.size
        for (x1, y1, x2, y2, cls) in parse_dota_label(label_path):
            mapped = class_map.get(cls)
            if mapped is None:
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
                continue
            crops.append((image.crop((x1, y1, x2, y2)), mapped))
    return crops


def get_visdrone_crops(img_dir, ann_dir, class_map=VISDRONE_TO_COMMON, min_crop_size=MIN_CROP_SIZE, allowed_files=None):
    """Ground-truth object crops from VisDrone, filtered/mapped to the shared vocabulary.

    allowed_files: optional set of annotation filenames to restrict extraction to (e.g. a
    train-only or held-out-only image split, to prevent train/eval leakage when a model is
    fine-tuned on VisDrone and then evaluated on it too). Default None = no restriction.
    """
    from PIL import Image

    crops = []
    ann_files = sorted(os.listdir(ann_dir))
    if allowed_files is not None:
        ann_files = [f for f in ann_files if f in allowed_files]
    for ann_file in ann_files:
        img_name = ann_file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        with open(os.path.join(ann_dir, ann_file), "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                cls_id = int(parts[5])
            except ValueError:
                continue
            if cls_id == 0 or cls_id not in VISDRONE_CLASSES or w < min_crop_size or h < min_crop_size:
                continue
            mapped = class_map.get(VISDRONE_CLASSES[cls_id])
            if mapped is None:
                continue
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            crops.append((image.crop((x1, y1, x2, y2)), mapped))
    return crops


def get_uavid_crops(img_dir, label_dir, class_map=None, min_crop_size=MIN_CROP_SIZE, min_area=800, log=print):
    """Ground-truth object crops from UAVid, derived from pixel-level segmentation masks via
    connected-component analysis -- UAVid provides per-pixel RGB-coded masks, not bounding boxes
    like DOTA/VisDrone/xView, so each contiguous same-color blob above min_area becomes one crop.

    class_map defaults to UAVID_TO_EXPANDED ({(128,0,0): "building"}). img_dir and label_dir
    must contain identically-named PNG files (UAVid's own train_data/Images + train_data/Labels
    layout already satisfies this).
    """
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    if class_map is None:
        class_map = UAVID_TO_EXPANDED

    crops = []
    n_images = 0
    for label_name in sorted(os.listdir(label_dir)):
        img_path = os.path.join(img_dir, label_name)
        if not os.path.exists(img_path):
            continue
        n_images += 1
        image = Image.open(img_path).convert("RGB")
        label_arr = np.array(Image.open(os.path.join(label_dir, label_name)).convert("RGB"))
        for color, cls in class_map.items():
            mask = np.all(label_arr == np.array(color), axis=-1)
            if not mask.any():
                continue
            labeled, num_components = ndimage.label(mask)
            for i in range(1, num_components + 1):
                ys, xs = np.where(labeled == i)
                if len(ys) < min_area:
                    continue
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
                    continue
                crops.append((image.crop((x1, y1, x2, y2)), cls))
        if log is not None and n_images % 50 == 0:
            log(f"  [uavid] scanned {n_images} images, {len(crops)} mapped crops so far", flush=True)

    if log is not None:
        log(f"  [uavid] done: scanned {n_images} images, {len(crops)} mapped crops", flush=True)
    return crops


def get_xview_crops(class_map=None, split="train", min_crop_size=MIN_CROP_SIZE, max_images=None, log=print):
    """Ground-truth object crops from xView, streamed from the HichTala/xview HF mirror --
    no local download needed. Images are already-decoded 512x512 chips with parsed bbox/category
    lists, so unlike get_dota_crops/get_visdrone_crops there's no raw label-file parsing step.

    class_map defaults to XVIEW_TO_EXPANDED (car/bus/truck/van). NOTE: this expanded vocabulary
    is for xView<->VisDrone comparisons specifically -- DOTA only distinguishes small-vehicle/
    large-vehicle, not car/bus/truck/van, so it can't participate in this finer scheme.
    """
    from datasets import load_dataset

    if class_map is None:
        class_map = XVIEW_TO_EXPANDED

    ds = load_dataset("HichTala/xview", split=split, streaming=True)
    category_names = ds.features["objects"]["category"].feature.names

    crops = []
    n_images = 0
    for example in ds:
        if max_images is not None and n_images >= max_images:
            break
        n_images += 1
        image = example["image"]
        W, H = image.size
        objects = example["objects"]
        for cat_id, bbox in zip(objects["category"], objects["bbox"]):
            mapped = class_map.get(category_names[cat_id])
            if mapped is None:
                continue
            x, y, w, h = bbox
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
                continue
            crops.append((image.crop((x1, y1, x2, y2)), mapped))
        if log is not None and n_images % 200 == 0:
            log(f"  [xview] scanned {n_images} images, {len(crops)} mapped crops so far", flush=True)

    if log is not None:
        log(f"  [xview] done: scanned {n_images} images, {len(crops)} mapped crops", flush=True)
    return crops


def split_visdrone_images(ann_dir, holdout_fraction, seed):
    """Split VisDrone annotation filenames into a training pool and a held-out pool, at the
    image level (so no image contributes crops to both). Use this whenever a model is (or might
    be) fine-tuned on VisDrone and then also evaluated on it, to prevent train/eval leakage --
    fine-tuning should only ever draw from the training pool, evaluation only from the held-out
    pool. Returns (train_files, holdout_files), both sets of annotation filenames.
    """
    all_ann_files = sorted(os.listdir(ann_dir))
    rng = random.Random(seed)
    shuffled = all_ann_files[:]
    rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * holdout_fraction)
    holdout_files = set(shuffled[:n_holdout])
    train_files = set(shuffled[n_holdout:])
    return train_files, holdout_files


def subsample_balanced(crops, max_per_class, seed):
    """Random, class-balanced subsample -- representative across the whole scan, not a filename prefix."""
    rng = random.Random(seed)
    by_class = {}
    for crop, cls in crops:
        by_class.setdefault(cls, []).append((crop, cls))
    out = []
    for cls, items in by_class.items():
        if len(items) > max_per_class:
            items = rng.sample(items, max_per_class)
        out.extend(items)
    rng.shuffle(out)
    return out
