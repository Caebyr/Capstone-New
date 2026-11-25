"""
Checkpoint B2: zero-shot open-vocabulary detectors on the overhead and UAV benchmark sets.

  python run_detectors.py --detector owlv2      # google/owlv2-base-patch16-ensemble
  python run_detectors.py --detector gdino      # IDEA-Research/grounding-dino-tiny
  python run_detectors.py --detector owlv2 --limit 20   # quick smoke run

Tasks (results/detection/benchmark_report.json): D1 prompts {small vehicle, large vehicle} on DOTA tiles and
VisDrone held-out; D2 prompts {car, bus, truck, building} on xView val, VisDrone held-out and UAVid val.
Same prompt words in both domains; no tuning. OWLv2 uses the template "a photo of a {}" (as in the crop
classification); Grounding DINO takes the bare class words ("car . bus . truck . building .").
Each box keeps its highest-scoring class only (one label per box for both detectors), score >= MIN_SCORE,
at most MAX_PER_IMAGE boxes per image. Batch 1, fp32, torch.no_grad().

Detections: data/processed/detection/dets/<detector>__<set>.json (COCO results format, git-ignored).
Metrics are computed by eval_detection.py.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from PIL import Image

DET_ROOT, GT = "data/processed/detection", "data/processed/detection/gt"
MIN_SCORE, MAX_PER_IMAGE = 0.05, 300
MODELS = {"owlv2": "google/owlv2-base-patch16-ensemble", "gdino": "IDEA-Research/grounding-dino-tiny"}
REV = {"owlv2": "cfd3195ba4ea9592eec887ded089f4c08eff231d", "gdino": "a2bb814dd30d776dcf7e30523b00659f4f141c71"}  # results/detection/model_revisions.json
RUNS = [("dota_tiles", "D1"), ("visdrone_ho_D1", "D1"), ("xview_val", "D2"), ("visdrone_ho_D2", "D2"), ("uavid_tv", "D2")]
PROMPTS = {"D1": ["small-vehicle", "large-vehicle"], "D2": ["car", "bus", "truck", "building"]}
WORDS = {"small-vehicle": "small vehicle", "large-vehicle": "large vehicle"}
CAT_ID = {"small-vehicle": 1, "large-vehicle": 2, "car": 3, "bus": 4, "truck": 5, "building": 6}
Image.MAX_IMAGE_PIXELS = None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Owl:
    def __init__(self, device):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        self.proc = Owlv2Processor.from_pretrained(MODELS["owlv2"], revision=REV["owlv2"])
        self.model = Owlv2ForObjectDetection.from_pretrained(MODELS["owlv2"], revision=REV["owlv2"]).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, img, classes):
        texts = [[f"a photo of a {WORDS.get(c, c)}" for c in classes]]
        inp = self.proc(text=texts, images=img, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        W, H = img.size
        S = max(W, H)  # OWLv2 pads to a square before resizing; boxes are relative to that square
        r = self.proc.post_process_object_detection(out, threshold=MIN_SCORE, target_sizes=torch.tensor([[S, S]], device=self.device))[0]
        return r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), r["labels"].cpu().numpy()


class GDino:
    def __init__(self, device):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(MODELS["gdino"], revision=REV["gdino"])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(MODELS["gdino"], revision=REV["gdino"]).to(device).eval()
        self.device = device
        self._spans = {}

    def spans(self, classes, input_ids):
        key = tuple(classes)
        if key not in self._spans:
            ids = input_ids[0].tolist()
            out = []
            for c in classes:
                sub = self.proc.tokenizer(WORDS.get(c, c), add_special_tokens=False)["input_ids"]
                pos = next(i for i in range(len(ids) - len(sub) + 1) if ids[i:i + len(sub)] == sub)
                out.append(list(range(pos, pos + len(sub))))
            self._spans[key] = out
        return self._spans[key]

    @torch.no_grad()
    def __call__(self, img, classes):
        text = " . ".join(WORDS.get(c, c) for c in classes) + " ."
        inp = self.proc(images=img, text=text, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        prob = out.logits[0].sigmoid()  # [queries, text tokens]
        sc = torch.stack([prob[:, s].max(-1).values for s in self.spans(classes, inp["input_ids"])], -1)  # [queries, classes]
        scores, labels = sc.max(-1)
        W, H = img.size
        cx, cy, w, h = out.pred_boxes[0].unbind(-1)  # normalised to the (unpadded) image
        boxes = torch.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], -1)
        keep = scores >= MIN_SCORE
        return boxes[keep].cpu().numpy(), scores[keep].cpu().numpy(), labels[keep].cpu().numpy()


def run(detector, limit=None):
    device = "cuda"
    model = Owl(device) if detector == "owlv2" else GDino(device)
    os.makedirs(f"{DET_ROOT}/dets", exist_ok=True)
    for set_name, task in RUNS:
        out_path = f"{DET_ROOT}/dets/{detector}__{set_name}{'__limit' + str(limit) if limit else ''}.json"
        if os.path.exists(out_path):
            log(f"{set_name}: exists, skipping")
            continue
        gt = json.load(open(f"{GT}/{set_name}.json"))
        classes = PROMPTS[task]
        images = gt["images"][:limit] if limit else gt["images"]
        dets, t0 = [], time.time()
        for k, im in enumerate(images):
            img = Image.open(os.path.join(DET_ROOT, im["file_name"])).convert("RGB")
            boxes, scores, labels = model(img, classes)
            W, H = img.size
            order = np.argsort(-scores)[:MAX_PER_IMAGE]
            for j in order:
                x1, y1, x2, y2 = np.clip(boxes[j], 0, [W, H, W, H])
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                dets.append({"image_id": im["id"], "category_id": CAT_ID[classes[int(labels[j])]],
                             "bbox": [round(float(x1), 1), round(float(y1), 1), round(float(x2 - x1), 1), round(float(y2 - y1), 1)],
                             "score": round(float(scores[j]), 4)})
            if (k + 1) % 200 == 0:
                el = time.time() - t0
                log(f"  {detector} {set_name}: {k + 1}/{len(images)} images, {(k + 1) / el:.2f} img/s, "
                    f"ETA {(len(images) - k - 1) / ((k + 1) / el) / 60:.0f} min, peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
        with open(out_path, "w") as f:
            json.dump(dets, f)
        log(f"{detector} {set_name}: {len(images)} images, {len(dets)} detections, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=list(MODELS), required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.detector, a.limit)
