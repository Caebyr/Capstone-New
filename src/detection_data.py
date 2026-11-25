"""Detection benchmark (checkpoint B1): COCO-format ground truth for overhead and UAV images.

Sets (images are written once to data/processed/detection/<set>/, git-ignored):
  dota_tiles      DOTA scenes cut into 1024x1024 tiles (stride 1024), from a fixed random set of images
  visdrone_ho     VisDrone held-out 20% images (the paper split; never seen by the fine-tuned model), as is
  xview_val       xView *validation* chips (512x512) from the pinned mirror; not used anywhere in the drift work
  uavid_tv        UAVid train + val images (3840x2160) cut into a 2x2 grid of 1920x1080 tiles (detectors are
                  zero-shot, so the train split is fair to use; val alone has only ~250 buildings)

Label schemes (COCO category ids are global):
  D1  small-vehicle (1), large-vehicle (2)              DOTA tiles vs VisDrone held-out (P1 mapping)
  D2  car (3), bus (4), truck (5), building (6)         xView val vs VisDrone held-out (car/bus/truck) and UAVid train+val (building, car)

Ignore regions (iscrowd=1, so a detection there is neither a TP nor an FP):
  - any GT box whose clipped width or height is < 16 px (the crop analysis used the same cut-off)
  - boxes cut by a tile border: less than 70% of their area inside the tile
  - DOTA difficult=1; VisDrone ignored regions (class 0)
  - D2 only: vans (VisDrone van, xView Pickup Truck) -- calling a van a car or truck is not a clear error
Each set lists the categories it is evaluated on ("eval_categories"); a dataset is never scored on a
class it does not annotate (VisDrone has no buildings; DOTA has no car/bus/truck split).
UAVid boxes come from connected mask components (area >= 800 px), as in the crop analysis: adjacent
buildings or parked cars can merge into one box, which makes UAVid AP approximate.
"""
import io
import json
import os
import random

import numpy as np
from PIL import Image

from src.dataset import MIN_CROP_SIZE, VISDRONE_CLASSES, parse_dota_label, split_visdrone_images
from src.full_scale import DOTA_IMG_DIRS, DOTA_LABEL_DIR, UAVID_BASE, UAVID_COLOURS, VISDRONE_BASE, iter_xview, xview_category_names

OUT = "data/processed/detection"
CATS = [{"id": 1, "name": "small-vehicle"}, {"id": 2, "name": "large-vehicle"}, {"id": 3, "name": "car"},
        {"id": 4, "name": "bus"}, {"id": 5, "name": "truck"}, {"id": 6, "name": "building"}]
CAT_ID = {c["name"]: c["id"] for c in CATS}
KEEP_FRAC = 0.7
Image.MAX_IMAGE_PIXELS = None

XVIEW_D2 = {"Small Car": "car", "Passenger Vehicle": "car", "Bus": "bus", "Truck": "truck", "Cargo Truck": "truck",
            "Truck w/Box": "truck", "Truck w/Flatbed": "truck", "Truck w/Liquid": "truck", "Truck Tractor": "truck",
            "Utility Truck": "truck", "Dump Truck": "truck", "Haul Truck": "truck", "Building": "building"}
XVIEW_IGNORE = {"Pickup Truck"}
VISDRONE_D1 = {"car": "small-vehicle", "van": "large-vehicle", "truck": "large-vehicle", "bus": "large-vehicle"}
VISDRONE_D2 = {"car": "car", "bus": "bus", "truck": "truck"}
VISDRONE_IGNORE_D2 = {"van"}


class CocoBuilder:
    def __init__(self, name, eval_categories, note=""):
        self.name, self.eval_categories, self.note = name, eval_categories, note
        self.images, self.anns = [], []

    def add_image(self, file_name, w, h, **meta):
        iid = len(self.images) + 1
        self.images.append({"id": iid, "file_name": file_name, "width": w, "height": h, **meta})
        return iid

    def add_box(self, iid, cat, x1, y1, x2, y2, ignore=False):
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return
        if not ignore and (w < MIN_CROP_SIZE or h < MIN_CROP_SIZE):
            ignore = True
        self.anns.append({"id": len(self.anns) + 1, "image_id": iid, "category_id": CAT_ID[cat] if cat in CAT_ID else 0,
                          "bbox": [float(x1), float(y1), float(w), float(h)], "area": float(w * h), "iscrowd": int(ignore)})

    def save(self, path):
        # ignore regions need a real category id for COCOeval; give them every evaluated id (duplicated) so they
        # suppress FPs for any class
        anns = []
        for a in self.anns:
            if a["iscrowd"]:
                for c in self.eval_categories:
                    anns.append(dict(a, id=len(anns) + 1, category_id=CAT_ID[c]))
            elif a["category_id"] in [CAT_ID[c] for c in self.eval_categories]:
                anns.append(dict(a, id=len(anns) + 1))
        js = {"info": {"set": self.name, "eval_categories": self.eval_categories, "note": self.note},
              "images": self.images, "annotations": anns, "categories": CATS}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(js, f)
        n_gt = sum(1 for a in anns if not a["iscrowd"])
        return {"images": len(self.images), "gt_boxes": n_gt,
                "gt_per_category": {c: sum(1 for a in anns if not a["iscrowd"] and a["category_id"] == CAT_ID[c]) for c in self.eval_categories}}


def _clip_to_tile(box, tx, ty, tw, th):
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = max(x1, tx), max(y1, ty), min(x2, tx + tw), min(y2, ty + th)
    if cx2 <= cx1 or cy2 <= cy1:
        return None, 0.0
    frac = (cx2 - cx1) * (cy2 - cy1) / max((x2 - x1) * (y2 - y1), 1)
    return (cx1 - tx, cy1 - ty, cx2 - tx, cy2 - ty), frac


def build_dota_tiles(n_images=80, tile=1024, seed=20261127):
    """Tiles from a random set of DOTA scenes that contain at least one vehicle; all tiles of each scene are kept
    (including empty ones, so false positives on background count)."""
    paths = []
    for d in DOTA_IMG_DIRS:
        paths += [os.path.join(d, f) for f in sorted(os.listdir(d))]
    with_vehicles = []
    for p in paths:
        lp = os.path.join(DOTA_LABEL_DIR, os.path.basename(p).replace(".png", ".txt"))
        if os.path.exists(lp) and any(b[4] in ("small-vehicle", "large-vehicle") for b in parse_dota_label(lp)):
            with_vehicles.append(p)
    chosen = sorted(random.Random(seed).sample(with_vehicles, min(n_images, len(with_vehicles))))
    cb = CocoBuilder("dota_tiles", ["small-vehicle", "large-vehicle"], "1024 px tiles, stride 1024")
    out_dir = f"{OUT}/dota_tiles"
    os.makedirs(out_dir, exist_ok=True)
    for p in chosen:
        name = os.path.basename(p)[:-4]
        lp = os.path.join(DOTA_LABEL_DIR, name + ".txt")
        boxes, flags = [], []
        with open(lp) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                try:
                    c = list(map(float, parts[:8]))
                except ValueError:
                    continue
                xs, ys = c[0::2], c[1::2]
                boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)), parts[8]))
                flags.append(parts[9] if len(parts) > 9 else "0")
        img = Image.open(p).convert("RGB")
        W, H = img.size
        for ty in range(0, H, tile):
            for tx in range(0, W, tile):
                tw, th = min(tile, W - tx), min(tile, H - ty)
                if tw < 256 or th < 256:  # skip slivers at the scene edge
                    continue
                fn = f"dota_tiles/{name}_{tx}_{ty}.png"
                if not os.path.exists(f"{OUT}/{fn}"):
                    img.crop((tx, ty, tx + tw, ty + th)).save(f"{OUT}/{fn}")
                iid = cb.add_image(fn, tw, th, source=os.path.basename(p), offset=[tx, ty])
                for (x1, y1, x2, y2, cls), diff in zip(boxes, flags):
                    if cls not in ("small-vehicle", "large-vehicle"):
                        continue
                    cbox, frac = _clip_to_tile((max(0, x1), max(0, y1), min(W, x2), min(H, y2)), tx, ty, tw, th)
                    if cbox is None:
                        continue
                    cb.add_box(iid, cls, *cbox, ignore=(frac < KEEP_FRAC or diff == "1"))
    return cb


def build_visdrone_heldout():
    img_dir, ann_dir = os.path.join(VISDRONE_BASE, "images"), os.path.join(VISDRONE_BASE, "annotations")
    _, holdout = split_visdrone_images(ann_dir, 0.2, 1234)
    d1 = CocoBuilder("visdrone_ho_D1", ["small-vehicle", "large-vehicle"], "held-out 20% images, P1 mapping")
    d2 = CocoBuilder("visdrone_ho_D2", ["car", "bus", "truck"], "held-out 20% images; van = ignore")
    for ann_file in sorted(holdout):
        img_name = ann_file.replace(".txt", ".jpg")
        path = os.path.join(img_dir, img_name)
        with Image.open(path) as im:
            W, H = im.size
        rel = os.path.relpath(path, OUT).replace("\\", "/")
        i1 = d1.add_image(rel, W, H, source=img_name)
        i2 = d2.add_image(rel, W, H, source=img_name)
        with open(os.path.join(ann_dir, ann_file)) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                x, y, w, h, cid = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[5])
                box = (max(0, x), max(0, y), min(W, x + w), min(H, y + h))
                if cid == 0:  # annotated ignore region
                    d1.add_box(i1, None, *box, ignore=True)
                    d2.add_box(i2, None, *box, ignore=True)
                    continue
                cls = VISDRONE_CLASSES.get(cid)
                small = w < MIN_CROP_SIZE or h < MIN_CROP_SIZE  # same pre-clip size rule as the crop pools
                if cls in VISDRONE_D1:
                    d1.add_box(i1, VISDRONE_D1[cls], *box, ignore=small)
                if cls in VISDRONE_D2:
                    d2.add_box(i2, VISDRONE_D2[cls], *box, ignore=small)
                elif cls in VISDRONE_IGNORE_D2:
                    d2.add_box(i2, None, *box, ignore=True)
    return d1, d2


def build_xview_val(max_chips=None):
    names = xview_category_names()
    cb = CocoBuilder("xview_val", ["car", "bus", "truck", "building"], "pinned mirror validation split; Pickup Truck = ignore")
    os.makedirs(f"{OUT}/xview_val", exist_ok=True)
    for i, row in iter_xview("validation", columns=["image_id", "image", "objects"]):
        if max_chips is not None and i >= max_chips:
            break
        img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        W, H = img.size
        fn = f"xview_val/chip_{row['image_id']}.png"
        if not os.path.exists(f"{OUT}/{fn}"):
            img.save(f"{OUT}/{fn}")
        iid = cb.add_image(fn, W, H, source=f"xview_val_{row['image_id']}", row=i)
        for cat_id, (x, y, w, h) in zip(row["objects"]["category"], row["objects"]["bbox"]):
            cls = names[cat_id]
            box = (max(0, x), max(0, y), min(W, x + w), min(H, y + h))
            if cls in XVIEW_D2:
                cb.add_box(iid, XVIEW_D2[cls], *box)
            elif cls in XVIEW_IGNORE:
                cb.add_box(iid, None, *box, ignore=True)
    return cb


def build_uavid(min_area=800, splits=("train_data", "val_data")):
    from scipy import ndimage
    cb = CocoBuilder("uavid_tv", ["car", "building"], "train+val images as 2x2 tiles of 1920x1080; boxes = mask components")
    os.makedirs(f"{OUT}/uavid_tv", exist_ok=True)
    for split in splits:
      img_dir, lab_dir = f"{UAVID_BASE}/{split}/Images", f"{UAVID_BASE}/{split}/Labels"
      for name in sorted(os.listdir(lab_dir)):
          img = Image.open(os.path.join(img_dir, name)).convert("RGB")
          arr = np.array(Image.open(os.path.join(lab_dir, name)).convert("RGB"))
          H, W = arr.shape[:2]
          comps = []
          for cls in ("building", "car"):
              mask = np.zeros((H, W), bool)
              for col in UAVID_COLOURS[cls]:
                  mask |= np.all(arr == np.array(col), axis=-1)
              lab, n = ndimage.label(mask)
              for k, sl in enumerate(ndimage.find_objects(lab), start=1):
                  area = int((lab[sl] == k).sum())
                  box = (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)
                  comps.append((cls, box, area >= min_area))
          tw, th = W // 2, H // 2
          for ty in (0, th):
              for tx in (0, tw):
                  fn = f"uavid_tv/{split[:-5]}_{name[:-4]}_{tx}_{ty}.png"
                  if not os.path.exists(f"{OUT}/{fn}"):
                      img.crop((tx, ty, tx + tw, ty + th)).save(f"{OUT}/{fn}")
                  iid = cb.add_image(fn, tw, th, source=f"{split}/{name}", offset=[tx, ty])
                  for cls, box, big in comps:
                      cbox, frac = _clip_to_tile(box, tx, ty, tw, th)
                      if cbox is None:
                          continue
                      cb.add_box(iid, cls, *cbox, ignore=(frac < KEEP_FRAC or not big))
    return cb
