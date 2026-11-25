"""Scoring for the detection benchmark (checkpoints B1-B3).

detections: list of {"image_id", "category_id", "bbox": [x, y, w, h], "score"} (COCO results format).

  coco_metrics      COCO AP@[.5:.95], AP50, AR per evaluated category (pycocotools; maxDets 500, since
                    aerial tiles hold hundreds of vehicles)
  agnostic_metrics  the same with every class merged into one "object" class: did the detector FIND the
                    objects, whatever it called them
  naming            for every non-ignored GT box that some detection overlaps at IoU >= 0.5, the class of the
                    highest-scoring such detection: accuracy and confusion. Separates "found but misnamed"
                    from "not found".
"""
import contextlib
import copy
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

MAX_DETS = [10, 100, 500]


def _coco(gt_json):
    c = COCO()
    c.dataset = copy.deepcopy(gt_json)
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c


def _run(gt_json, dets, cat_ids):
    gt = _coco(gt_json)
    if not dets:
        return None
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(copy.deepcopy(dets))
        ev = COCOeval(gt, dt, "bbox")
        ev.params.maxDets = MAX_DETS
        ev.params.catIds = cat_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


def _ap_from_eval(ev, k, iou=None):
    """AP for category index k from accumulated precision [T, R, K, A, M]; area 'all', maxDets 500."""
    p = ev.eval["precision"]
    t = slice(None) if iou is None else [int(np.argmin(np.abs(ev.params.iouThrs - iou)))]
    v = p[t, :, k, 0, -1]
    v = v[v > -1]
    return float(v.mean()) if v.size else float("nan")


def _ar_from_eval(ev, k):
    r = ev.eval["recall"][:, k, 0, -1]
    r = r[r > -1]
    return float(r.mean()) if r.size else float("nan")


def coco_metrics(gt_json, dets):
    names = gt_json["info"]["eval_categories"]
    ids = [c["id"] for c in gt_json["categories"] if c["name"] in names]
    id2name = {c["id"]: c["name"] for c in gt_json["categories"]}
    dets = [d for d in dets if d["category_id"] in ids]
    ev = _run(gt_json, dets, ids)
    out = {}
    for k, cid in enumerate(ids):
        n_gt = sum(1 for a in gt_json["annotations"] if a["category_id"] == cid and not a["iscrowd"])
        if ev is None:
            out[id2name[cid]] = {"AP": 0.0, "AP50": 0.0, "AR": 0.0, "n_gt": n_gt}
        else:
            out[id2name[cid]] = {"AP": _ap_from_eval(ev, k), "AP50": _ap_from_eval(ev, k, 0.5), "AR": _ar_from_eval(ev, k), "n_gt": n_gt}
    vals = [v for v in out.values() if v["n_gt"] > 0]
    out["mean"] = {m: float(np.nanmean([v[m] for v in vals])) for m in ("AP", "AP50", "AR")}
    return out


def agnostic_metrics(gt_json, dets):
    g = copy.deepcopy(gt_json)
    keep = {c["id"] for c in g["categories"] if c["name"] in g["info"]["eval_categories"]}
    anns, seen_crowd = [], set()
    for a in g["annotations"]:
        if a["category_id"] not in keep:
            continue
        if a["iscrowd"]:  # ignore regions are duplicated per category in the GT file; keep one copy
            key = (a["image_id"], tuple(a["bbox"]))
            if key in seen_crowd:
                continue
            seen_crowd.add(key)
        anns.append(dict(a, category_id=1, id=len(anns) + 1))
    g["annotations"] = anns
    g["categories"] = [{"id": 1, "name": "object"}]
    g["info"] = dict(g["info"], eval_categories=["object"])
    d = [dict(x, category_id=1) for x in dets]
    return coco_metrics(g, d)["object"]


def _iou(box, boxes):
    x, y, w, h = box
    bx, by, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    iw = np.clip(np.minimum(x + w, bx + bw) - np.maximum(x, bx), 0, None)
    ih = np.clip(np.minimum(y + h, by + bh) - np.maximum(y, by), 0, None)
    inter = iw * ih
    return inter / (w * h + bw * bh - inter + 1e-9)


def naming(gt_json, dets, iou_thr=0.5, min_score=0.0):
    names = gt_json["info"]["eval_categories"]
    id2name = {c["id"]: c["name"] for c in gt_json["categories"]}
    by_img = {}
    for d in dets:
        if d["score"] >= min_score:
            by_img.setdefault(d["image_id"], []).append(d)
    conf = {g: {p: 0 for p in names + ["not found"]} for g in names}
    for a in gt_json["annotations"]:
        if a["iscrowd"] or id2name[a["category_id"]] not in names:
            continue
        g = id2name[a["category_id"]]
        cand = by_img.get(a["image_id"], [])
        if not cand:
            conf[g]["not found"] += 1
            continue
        boxes = np.array([c["bbox"] for c in cand], float)
        ious = _iou(np.array(a["bbox"], float), boxes)
        ok = np.where(ious >= iou_thr)[0]
        if not ok.size:
            conf[g]["not found"] += 1
            continue
        best = ok[np.argmax([cand[i]["score"] for i in ok])]
        p = id2name.get(cand[best]["category_id"])
        conf[g][p if p in names else "not found"] += 1
    found = {g: sum(v for k, v in conf[g].items() if k != "not found") for g in names}
    total = {g: sum(conf[g].values()) for g in names}
    return {"found_rate": {g: found[g] / total[g] if total[g] else float("nan") for g in names},
            "naming_accuracy_of_found": {g: conf[g][g] / found[g] if found[g] else float("nan") for g in names},
            "overall_found_rate": sum(found.values()) / max(sum(total.values()), 1),
            "overall_naming_accuracy": sum(conf[g][g] for g in names) / max(sum(found.values()), 1),
            "confusion": conf}


def self_test():
    """Synthetic checks with known answers; raises on failure."""
    cats = [{"id": 3, "name": "car"}, {"id": 4, "name": "bus"}]
    gt = {"info": {"eval_categories": ["car", "bus"]}, "categories": cats,
          "images": [{"id": 1, "width": 200, "height": 200}, {"id": 2, "width": 200, "height": 200}],
          "annotations": [
              {"id": 1, "image_id": 1, "category_id": 3, "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0},
              {"id": 2, "image_id": 1, "category_id": 4, "bbox": [100, 100, 60, 40], "area": 2400, "iscrowd": 0},
              {"id": 3, "image_id": 2, "category_id": 3, "bbox": [20, 20, 40, 40], "area": 1600, "iscrowd": 0},
              {"id": 4, "image_id": 2, "category_id": 3, "bbox": [150, 150, 30, 30], "area": 900, "iscrowd": 1},
              {"id": 5, "image_id": 2, "category_id": 4, "bbox": [150, 150, 30, 30], "area": 900, "iscrowd": 1}]}
    perfect = [{"image_id": a["image_id"], "category_id": a["category_id"], "bbox": a["bbox"], "score": 0.9}
               for a in gt["annotations"] if not a["iscrowd"]]
    m = coco_metrics(gt, perfect)
    assert abs(m["car"]["AP"] - 1) < 1e-6 and abs(m["bus"]["AP"] - 1) < 1e-6, m
    # a detection inside an ignore region must not lower precision
    m2 = coco_metrics(gt, perfect + [{"image_id": 2, "category_id": 3, "bbox": [152, 152, 25, 25], "score": 0.95}])
    assert abs(m2["car"]["AP"] - 1) < 1e-6, m2
    # a real false positive with a higher score than the TPs must lower car AP
    m3 = coco_metrics(gt, perfect + [{"image_id": 1, "category_id": 3, "bbox": [60, 150, 30, 30], "score": 0.99}])
    assert m3["car"]["AP"] < 0.9, m3
    # swapped classes: class-aware AP 0 for bus, class-agnostic AP 1, naming accuracy 0 for bus
    swapped = [dict(d, category_id=3 if d["category_id"] == 4 else d["category_id"]) for d in perfect]
    assert coco_metrics(gt, swapped)["bus"]["AP"] == 0
    assert abs(agnostic_metrics(gt, swapped)["AP"] - 1) < 1e-6
    nm = naming(gt, swapped)
    assert nm["naming_accuracy_of_found"]["bus"] == 0 and nm["naming_accuracy_of_found"]["car"] == 1 and nm["overall_found_rate"] == 1
    # only one of two cars detected: car AR = 0.5, found rate 0.5
    half = [d for d in perfect if not (d["image_id"] == 2)]
    m4 = coco_metrics(gt, half)
    assert abs(m4["car"]["AR"] - 0.5) < 1e-6, m4
    assert abs(naming(gt, half)["found_rate"]["car"] - 0.5) < 1e-6
    # IoU just below 0.5 does not count at AP50
    shifted = [{"image_id": 1, "category_id": 3, "bbox": [30, 30, 40, 40], "score": 0.9}]
    assert coco_metrics(gt, shifted)["car"]["AP50"] == 0
    return "detection_eval self-test passed (perfect, ignore region, FP, class swap, half recall, IoU threshold)"


if __name__ == "__main__":
    print(self_test())
