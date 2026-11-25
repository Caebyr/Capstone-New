"""
Checkpoints B2/B3: score the detector outputs, overhead vs UAV, and split errors into finding vs naming.

  python eval_detection.py

For every detector and benchmark set (src/detection_eval.py):
  class-aware COCO AP, AP50, AR per category      (B2: does detection degrade on overhead imagery?)
  class-agnostic AP50 / AR, found rate             (B3: are the objects found at all?)
  naming accuracy of found objects, confusion      (B3: when found, are they named correctly?)
"found" and "naming" use detections with score >= NAMING_MIN_SCORE and IoU >= 0.5.
Image-bootstrap CIs (N_BOOT reps over images) for found rate and naming accuracy.

Writes results/detection/detection_results.json and detection_results.md.
"""
import json
import os
import time

import numpy as np

from src.detection_eval import agnostic_metrics, coco_metrics, naming

DETS, GT, OUT = "data/processed/detection/dets", "data/processed/detection/gt", "results/detection"
NAMING_MIN_SCORE, N_BOOT, SEED = 0.1, 500, 20261128
PAIRS = {  # task -> list of (category, overhead set, UAV set)
    "D1": [("small-vehicle", "dota_tiles", "visdrone_ho_D1"), ("large-vehicle", "dota_tiles", "visdrone_ho_D1")],
    "D2": [("car", "xview_val", "visdrone_ho_D2"), ("bus", "xview_val", "visdrone_ho_D2"),
           ("truck", "xview_val", "visdrone_ho_D2"), ("building", "xview_val", "uavid_tv")],
}
DETECTORS = {"owlv2": "OWLv2 B/16 (ensemble)", "gdino": "Grounding DINO tiny"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def boot_naming(gt, dets, rng):
    """Image-bootstrap CI of found rate and naming accuracy per category (resample images with replacement)."""
    names = gt["info"]["eval_categories"]
    img_ids = [im["id"] for im in gt["images"]]
    # per-image per-category counts: found, named-correctly, total
    per = {i: {c: [0, 0, 0] for c in names} for i in img_ids}
    sub = {"info": gt["info"], "categories": gt["categories"], "images": gt["images"]}
    by_img_ann, by_img_det = {}, {}
    for a in gt["annotations"]:
        by_img_ann.setdefault(a["image_id"], []).append(a)
    for d in dets:
        by_img_det.setdefault(d["image_id"], []).append(d)
    for i in img_ids:
        r = naming(dict(sub, annotations=by_img_ann.get(i, [])), by_img_det.get(i, []), min_score=NAMING_MIN_SCORE)
        for c in names:
            row = r["confusion"][c]
            tot = sum(row.values())
            per[i][c] = [tot - row["not found"], row[c], tot]
    arr = np.array([[per[i][c] for c in names] for i in img_ids], float)  # [images, cats, 3]
    found, named = {c: [] for c in names}, {c: [] for c in names}
    for _ in range(N_BOOT):
        s = arr[rng.integers(0, len(img_ids), len(img_ids))].sum(0)
        for k, c in enumerate(names):
            if s[k, 2]:
                found[c].append(s[k, 0] / s[k, 2])
            if s[k, 0]:
                named[c].append(s[k, 1] / s[k, 0])
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if v else [float("nan")] * 2
    return {c: {"found_rate_ci": q(found[c]), "naming_accuracy_ci": q(named[c])} for c in names}


def main():
    rng = np.random.default_rng(SEED)
    res = {"settings": {"naming_min_score": NAMING_MIN_SCORE, "n_boot": N_BOOT}, "detectors": {}}
    sets = sorted({s for p in PAIRS.values() for _, o, u in p for s in (o, u)} | {"uavid_tv"})
    for det in DETECTORS:
        if not all(os.path.exists(f"{DETS}/{det}__{s}.json") for s in sets):
            log(f"{det}: detections missing, skipped")
            continue
        R = {}
        for s in sets:
            gt = json.load(open(f"{GT}/{s}.json"))
            dets = json.load(open(f"{DETS}/{det}__{s}.json"))
            R[s] = {"class_aware": coco_metrics(gt, dets), "agnostic": agnostic_metrics(gt, dets),
                    "naming": naming(gt, dets, min_score=NAMING_MIN_SCORE), "naming_ci": boot_naming(gt, dets, rng),
                    "n_images": len(gt["images"]), "n_dets": len(dets)}
            log(f"{det} {s}: mean AP50 {R[s]['class_aware']['mean']['AP50']:.3f}, agnostic AP50 {R[s]['agnostic']['AP50']:.3f}, "
                f"found {R[s]['naming']['overall_found_rate']:.3f}, naming acc {R[s]['naming']['overall_naming_accuracy']:.3f}")
        comp = {}
        for task, pairs in PAIRS.items():
            comp[task] = {}
            for c, o, u in pairs:
                co, cu = R[o]["class_aware"][c], R[u]["class_aware"][c]
                no, nu = R[o]["naming"], R[u]["naming"]
                comp[task][c] = {"overhead_set": o, "uav_set": u,
                                 "AP50": [co["AP50"], cu["AP50"]], "AP": [co["AP"], cu["AP"]], "AR": [co["AR"], cu["AR"]],
                                 "found_rate": [no["found_rate"][c], nu["found_rate"][c]],
                                 "found_rate_ci": [R[o]["naming_ci"][c]["found_rate_ci"], R[u]["naming_ci"][c]["found_rate_ci"]],
                                 "naming_acc": [no["naming_accuracy_of_found"][c], nu["naming_accuracy_of_found"][c]],
                                 "naming_acc_ci": [R[o]["naming_ci"][c]["naming_accuracy_ci"], R[u]["naming_ci"][c]["naming_accuracy_ci"]],
                                 "confusion": [no["confusion"][c], nu["confusion"][c]]}
        res["detectors"][det] = {"per_set": R, "overhead_vs_uav": comp}
    with open(f"{OUT}/detection_results.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    write_md(res)
    log("done")


def write_md(res):
    pct = lambda v: "—" if v is None or v != v else f"{100 * v:.1f}"
    L = ["# Zero-shot open-vocabulary detection: overhead vs UAV", "",
         "AP50 / AP: COCO, class-aware. Found: share of GT boxes overlapped (IoU ≥ 0.5) by any detection with score ≥ "
         f"{res['settings']['naming_min_score']}. Named: of found boxes, share whose best detection has the right class. "
         f"Brackets: 95% image-bootstrap CI ({res['settings']['n_boot']} reps). O = overhead, U = UAV.", ""]
    for det, D in res["detectors"].items():
        L += [f"## {DETECTORS[det]}", "", "| Task | Category | Sets (O / U) | AP50 O / U | AP O / U | Found O [CI] | Found U [CI] | Named O [CI] | Named U [CI] |",
              "|---|---|---|---|---|---|---|---|---|"]
        for task, cats in D["overhead_vs_uav"].items():
            for c, v in cats.items():
                ci = lambda x: f"[{pct(x[0])}, {pct(x[1])}]"
                L.append(f"| {task} | {c} | {v['overhead_set']} / {v['uav_set']} | {pct(v['AP50'][0])} / {pct(v['AP50'][1])} | {pct(v['AP'][0])} / {pct(v['AP'][1])} | "
                         f"{pct(v['found_rate'][0])} {ci(v['found_rate_ci'][0])} | {pct(v['found_rate'][1])} {ci(v['found_rate_ci'][1])} | "
                         f"{pct(v['naming_acc'][0])} {ci(v['naming_acc_ci'][0])} | {pct(v['naming_acc'][1])} {ci(v['naming_acc_ci'][1])} |")
        L += ["", "Per set: " + "; ".join(f"{s} mean AP50 {pct(r['class_aware']['mean']['AP50'])}, class-agnostic AP50 {pct(r['agnostic']['AP50'])}"
                                         for s, r in D["per_set"].items()), ""]
    with open(f"{OUT}/detection_results.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
