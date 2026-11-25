"""
Checkpoint A1 of the full-scale work: list every eligible ground-truth box in the full datasets.

  DOTA      all 1,413 training images (part1 + part2 + part3), small-vehicle / large-vehicle
  VisDrone  all 6,471 training images, car / van / truck / bus, tagged train vs held-out (paper split)
  xView     every chip of the revision-pinned HF mirror's train split, vehicle classes + Building
  UAVid     train + val, building and car (moving + static merged) from the masks

Writes results/full_scale/pools/<dataset>_boxes.csv.gz and results/full_scale/pools/pool_report.json.
Also checks that the boxes the paper drew from are reproduced: restricted to part1 DOTA / the first
5,000 (500 for buildings) xView chips, the per-class counts must equal the counts printed by the August
runs (results/paper_facts/original_environment_records.txt). Exits non-zero if a check fails.

Usage (hf env, repo root; set HF_HOME to the project's HF cache to reuse the downloaded xView):
  python build_full_pools.py
"""
import json
import os
import sys
import time
from collections import Counter

from src.dataset import XVIEW_TO_EXPANDED
from src.full_scale import (list_dota_boxes, list_uavid_boxes, list_visdrone_boxes, list_xview_boxes,
                            write_pool)

OUT = "results/full_scale/pools"
# Per-class counts printed by the August runs (also the replay checks in verify_crop_identity.py)
PAPER_COUNTS = {
    "dota_part1": {"large-vehicle": 3033, "small-vehicle": 6537},
    "visdrone_heldout_p1": {"small-vehicle": 22368, "large-vehicle": 7341},
    "visdrone_all": {"car": 109567, "van": 20534, "truck": 11297, "bus": 5233},
    "xview_first5000": {"car": 1269, "truck": 3694, "bus": 903, "van": 25},
    "xview_first500_building": {"building": 3711},
    "uavid_building": {"building": 1199},
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    report, checks = {}, []

    def check(name, got, want):
        ok = got == want
        checks.append({"check": name, "ok": ok, "got": got, "expected": want})
        log(f"  check {'OK  ' if ok else 'FAIL'} {name}: {got}" + ("" if ok else f" (expected {want})"))

    log("DOTA ...")
    dota = list_dota_boxes()
    write_pool(f"{OUT}/dota_boxes.csv.gz", dota, "dota")
    p1 = [r for r in dota if r["image_ref"].replace("\\", "/").startswith("data/raw/DOTA/images/")]
    check("DOTA part1 counts = paper", dict(Counter(r["raw_class"] for r in p1)), PAPER_COUNTS["dota_part1"])
    report["dota"] = {"images": len({r["source_image"] for r in dota}), "boxes": dict(Counter(r["raw_class"] for r in dota))}
    log(f"  {report['dota']}")

    log("VisDrone ...")
    vis = list_visdrone_boxes()
    write_pool(f"{OUT}/visdrone_boxes.csv.gz", vis, "visdrone")
    p1map = {"car": "small-vehicle", "van": "large-vehicle", "truck": "large-vehicle", "bus": "large-vehicle"}
    check("VisDrone held-out P1 counts = paper",
          dict(Counter(p1map[r["raw_class"]] for r in vis if r["visdrone_split"] == "heldout")), PAPER_COUNTS["visdrone_heldout_p1"])
    check("VisDrone all-image counts = paper", dict(Counter(r["raw_class"] for r in vis)), PAPER_COUNTS["visdrone_all"])
    report["visdrone"] = {"images": len({r["source_image"] for r in vis}),
                          "boxes": dict(Counter(r["raw_class"] for r in vis)),
                          "boxes_heldout": dict(Counter(r["raw_class"] for r in vis if r["visdrone_split"] == "heldout"))}
    log(f"  {report['visdrone']}")

    log("xView (local parquet, pinned revision) ...")
    xv = list_xview_boxes(list(XVIEW_TO_EXPANDED))
    write_pool(f"{OUT}/xview_boxes.csv.gz", xv, "xview")
    idx = lambda r: int(r["image_ref"].split(":")[1])
    first5000 = Counter(XVIEW_TO_EXPANDED[r["raw_class"]] for r in xv if idx(r) < 5000 and r["raw_class"] != "Building")
    check("xView first 5,000 chips counts = paper", dict(first5000), PAPER_COUNTS["xview_first5000"])
    check("xView first 500 chips building = paper",
          {"building": sum(1 for r in xv if idx(r) < 500 and r["raw_class"] == "Building")}, PAPER_COUNTS["xview_first500_building"])
    n_chips = max(idx(r) for r in xv) + 1 if xv else 0
    report["xview"] = {"chips_with_boxes": len({r["image_ref"] for r in xv}), "last_chip_index_with_box": n_chips - 1,
                       "boxes_raw": dict(Counter(r["raw_class"] for r in xv)),
                       "boxes_mapped": dict(Counter(XVIEW_TO_EXPANDED[r["raw_class"]] for r in xv))}
    log(f"  {report['xview']['boxes_mapped']} over {report['xview']['chips_with_boxes']} chips")

    log("UAVid ...")
    ua = list_uavid_boxes()
    write_pool(f"{OUT}/uavid_boxes.csv.gz", ua, "uavid")
    check("UAVid building counts = paper", {"building": sum(1 for r in ua if r["raw_class"] == "building")}, PAPER_COUNTS["uavid_building"])
    report["uavid"] = {"images": len({r["source_image"] for r in ua}), "boxes": dict(Counter(r["raw_class"] for r in ua))}
    log(f"  {report['uavid']}")

    report["checks"] = checks
    report["seconds"] = round(time.time() - t0)
    with open(f"{OUT}/pool_report.json", "w") as f:
        json.dump(report, f, indent=2)
    n_fail = sum(not c["ok"] for c in checks)
    log(f"done in {report['seconds']} s; {len(checks) - n_fail}/{len(checks)} checks passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
