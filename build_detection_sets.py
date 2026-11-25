"""
Checkpoint B1: build the detection benchmark (COCO ground truth + image tiles) and self-test the scorer.

  python build_detection_sets.py

Writes images to data/processed/detection/ (git-ignored), ground truth to data/processed/detection/gt/*.json (git-ignored; rebuilt identically by this script) and a
summary to results/detection/benchmark_report.json. See src/detection_data.py for the sets, label schemes
and ignore rules. Exits non-zero if the scorer self-test fails.
"""
import json
import os
import sys
import time

from src.detection_data import (build_dota_tiles, build_uavid, build_visdrone_heldout, build_xview_val)
from src.detection_eval import self_test

GT = "data/processed/detection/gt"  # regenerable, git-ignored


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log(self_test())
    os.makedirs(GT, exist_ok=True)
    report = {}
    for name, fn in [("dota_tiles", build_dota_tiles), ("visdrone_ho", build_visdrone_heldout),
                     ("xview_val", build_xview_val), ("uavid_tv", build_uavid)]:
        t = time.time()
        built = fn()
        for cb in (built if isinstance(built, tuple) else (built,)):
            report[cb.name] = cb.save(f"{GT}/{cb.name}.json")
            log(f"{cb.name}: {report[cb.name]} ({time.time() - t:.0f} s)")
    report["tasks"] = {
        "D1": {"prompts": ["small-vehicle", "large-vehicle"], "overhead": "dota_tiles", "uav": "visdrone_ho_D1"},
        "D2": {"prompts": ["car", "bus", "truck", "building"], "overhead": "xview_val", "uav": ["visdrone_ho_D2", "uavid_tv"]},
    }
    with open("results/detection/benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)
    log("done")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("SELF-TEST FAILED:", e)
        sys.exit(1)
