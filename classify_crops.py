"""
Common zero-shot classifier on ground-truth crops (NOT detection), plus the paper's text-anchored
metrics A_c and delta_m_c (Section III-D), on exactly the crops behind Tables II-III.

Same rule for every model and domain: unit-norm image embedding (cached, verified against the
manifests), unit-norm text embeddings of "a photo of a {label}" from that model's own text encoder,
predict the candidate with the highest cosine. No training, no prompt tuning, no other classifier.

Tasks (src/crop_sets.py): P1 = {small vehicle, large vehicle} on DOTA vs held-out VisDrone (CLIP,
RemoteCLIP, fine-tuned CLIP); P2+P3 pooled = {car, bus, truck, building} on xView vs VisDrone/UAVid
(CLIP, RemoteCLIP). The vocabularies are never merged, and a one-candidate task is refused.

NOTE on the fine-tuned model: it was trained on VisDrone crops labelled small-/large-vehicle with
"a drone photo of a {}", i.e. nearly this exact P1 task on the UAV side (different images: held-out
split). Its UAV accuracy is supervised in-domain performance, not zero-shot transfer.

Usage (hf env):
  python classify_crops.py               # point estimates (step 1)
  python classify_crops.py --bootstrap   # + image-level and sequence-level bootstrap CIs (step 2)
Writes results/classification/
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

sys.path.insert(0, os.path.dirname(__file__))
from src.crop_sets import MODEL_NAMES, TASKS, load_domain
from src.embeddings import load_clip, load_remoteclip
from src.stats import ci, cluster_weights

PROMPT = "a photo of a {}"
# label as stored in the crop lists -> words used in the prompt
PROMPT_WORDS = {"small-vehicle": "small vehicle", "large-vehicle": "large vehicle",
                "car": "car", "bus": "bus", "truck": "truck", "building": "building"}
REMOTECLIP_CKPT = "models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt"
FINETUNED_CKPT = "models/finetuned_clip_visdrone_heldout/finetuned_clip_visdrone_heldout_epoch3.pt"
OUT_DIR = "results/classification"
N_BOOT = 2000
SEED = 20261013


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- text encoders

@torch.no_grad()
def text_embeddings(device):
    """{model: {task: [K, D] unit-norm text embeddings in TASKS[task]['categories'] order}}"""
    out = {}
    prompts = {t: [PROMPT.format(PROMPT_WORDS[c]) for c in s["categories"]] for t, s in TASKS.items()}

    clip_model, processor = load_clip(device)
    ft_model, _ = load_clip(device)
    ft_model.load_state_dict(torch.load(FINETUNED_CKPT, map_location=device))
    ft_model.eval()
    for name, model in (("clip", clip_model), ("finetuned", ft_model)):
        out[name] = {}
        for t, p in prompts.items():
            inputs = processor(text=p, return_tensors="pt", padding=True).to(device)
            e = model.get_text_features(**inputs).float()
            out[name][t] = (e / e.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float64)
    del clip_model, ft_model

    import open_clip
    rc_model, _ = load_remoteclip(REMOTECLIP_CKPT, device)
    tok = open_clip.get_tokenizer("ViT-L-14")
    out["remoteclip"] = {}
    for t, p in prompts.items():
        e = rc_model.encode_text(tok(p).to(device)).float()
        out["remoteclip"][t] = (e / e.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float64)
    del rc_model
    torch.cuda.empty_cache()
    return out, prompts


# ---------------------------------------------------------------- classifier + metrics

def classify(feats, text, cats):
    """Cosine scores [N, K] and predicted labels. Refuses a one-candidate vocabulary (would be 100%)."""
    if len(cats) < 2:
        raise ValueError(f"refusing one-class evaluation with candidates {cats}")
    scores = feats @ text.T
    return scores, np.array(cats)[scores.argmax(1)]


def class_metrics(y_true, y_pred, cats, w=None):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=cats, sample_weight=w, zero_division=0)
    acc = float(np.average(y_true == y_pred, weights=w))
    return {"accuracy": acc, "macro_f1": float(f1_score(y_true, y_pred, labels=cats, average="macro", sample_weight=w, zero_division=0)),
            "per_category": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "n": float(s[i])} for i, c in enumerate(cats)},
            "confusion": confusion_matrix(y_true, y_pred, labels=cats, sample_weight=w).tolist()}


def text_metrics(feats, labels, text, cats, w=None):
    """a_c = cos(mu_c, t_c) and m_c = a_c - max_{k!=c} cos(mu_c, t_k) at the arithmetic centroid mu_c."""
    w = np.ones(len(labels)) if w is None else w
    out = {}
    for i, c in enumerate(cats):
        m = labels == c
        if w[m].sum() == 0:
            out[c] = None
            continue
        mu = (w[m][:, None] * feats[m]).sum(0) / w[m].sum()
        cos = (text @ mu) / (np.linalg.norm(text, axis=1) * np.linalg.norm(mu))
        out[c] = {"a": float(cos[i]), "margin": float(cos[i] - np.delete(cos, i).max()),
                  "nearest_other": cats[int(np.argmax(np.where(np.arange(len(cats)) == i, -np.inf, cos)))]}
    return out


def evaluate(task, data, text, w=None):
    """Everything for one task under optional per-domain weights {domain: w}."""
    cats = TASKS[task]["categories"]
    res = {}
    for m in TASKS[task]["models"]:
        res[m] = {}
        for d in ("overhead", "uav"):
            x = data[m][d]
            wd = None if w is None else w[d]
            _, pred = classify(x["feats"], text[m][task], cats)
            res[m][d] = class_metrics(x["labels"], pred, cats, wd)
            res[m][d]["text"] = text_metrics(x["feats"], x["labels"], text[m][task], cats, wd)
        o, u = res[m]["overhead"], res[m]["uav"]
        res[m]["gap"] = {
            "accuracy_pp": 100 * (o["accuracy"] - u["accuracy"]),
            "macro_f1_pp": 100 * (o["macro_f1"] - u["macro_f1"]),
            "recall_pp": {c: 100 * (o["per_category"][c]["recall"] - u["per_category"][c]["recall"]) for c in cats},
            "A": {c: abs(o["text"][c]["a"] - u["text"][c]["a"]) if o["text"][c] and u["text"][c] else None for c in cats},
            "delta_m": {c: o["text"][c]["margin"] - u["text"][c]["margin"] if o["text"][c] and u["text"][c] else None for c in cats},
        }
    return res


# ---------------------------------------------------------------- correctness checks

def unit_checks():
    """Small synthetic checks of signs, refusal and argmax; raise on failure."""
    cats = ["a", "b"]
    t = np.eye(2)
    try:
        classify(np.eye(2), t[:1], ["a"])
        raise AssertionError("one-class evaluation was not refused")
    except ValueError:
        pass
    _, pred = classify(np.array([[0.9, 0.1], [0.2, 0.8]]), t, cats)
    assert pred.tolist() == ["a", "b"], "argmax prediction wrong"
    # overhead: both 'a' crops right; UAV: one of two wrong -> recall gap for 'a' must be +50 pp
    yo, po = np.array(["a", "a", "b"]), np.array(["a", "a", "b"])
    yu, pu = np.array(["a", "a", "b"]), np.array(["a", "b", "b"])
    gap = 100 * (class_metrics(yo, po, cats)["per_category"]["a"]["recall"] - class_metrics(yu, pu, cats)["per_category"]["a"]["recall"])
    assert abs(gap - 50) < 1e-9, f"recall gap sign/value wrong: {gap}"
    # margin: centroid exactly on t_a -> margin 1; centroid between t_a and t_b leaning to b -> negative
    fe = np.array([[1.0, 0.0]])
    assert abs(text_metrics(fe, np.array(["a"]), t, cats)["a"]["margin"] - 1) < 1e-9
    fe2 = np.array([[0.6, 0.8]])
    assert text_metrics(fe2, np.array(["a"]), t, cats)["a"]["margin"] < 0
    # delta_m sign: overhead margin larger than UAV margin -> positive (smaller UAV margin)
    mo = text_metrics(np.array([[1.0, 0.0]]), np.array(["a"]), t, cats)["a"]["margin"]
    mu = text_metrics(np.array([[0.8, 0.6]]), np.array(["a"]), t, cats)["a"]["margin"]
    assert mo - mu > 0
    log("unit checks passed (refusal, argmax, recall-gap sign, margin sign, delta_m sign)")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    unit_checks()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device: {device}")

    text, prompts = text_embeddings(device)
    checks = []
    for t in TASKS:
        # the fine-tuned model's text tower was frozen during fine-tuning -> must equal CLIP's
        if "finetuned" in TASKS[t]["models"]:
            checks.append((f"{t}: fine-tuned text embeddings equal CLIP's (frozen text tower)",
                           bool(np.allclose(text["finetuned"][t], text["clip"][t], atol=1e-5))))
        for m in TASKS[t]["models"]:
            checks.append((f"{t} {m}: text embeddings unit-norm", bool(np.allclose(np.linalg.norm(text[m][t], axis=1), 1, atol=1e-6))))

    results = {"prompt": PROMPT, "prompts": prompts, "note_finetuned": "supervised on the UAV side of P1 (held-out images); not zero-shot",
               "tasks": {}}
    pred_rows = []
    for task, spec in TASKS.items():
        cats = spec["categories"]
        data = {m: {d: load_domain(task, d, m) for d in ("overhead", "uav")} for m in spec["models"]}
        for d in ("overhead", "uav"):
            ref = data[spec["models"][0]][d]
            for m in spec["models"]:
                x = data[m][d]
                checks.append((f"{task} {d} {m}: crop IDs match {spec['models'][0]}", bool(np.array_equal(ref["crop_ids"], x["crop_ids"]))))
                checks.append((f"{task} {d} {m}: image embeddings unit-norm", bool(np.allclose(np.linalg.norm(x["feats"], axis=1), 1, atol=1e-4))))
                checks.append((f"{task} {d} {m}: image/text dims match", x["feats"].shape[1] == text[m][task].shape[1]))
                checks.append((f"{task} {d}: every candidate category present", set(np.unique(x["labels"])) == set(cats)))

        point = evaluate(task, data, text)
        results["tasks"][task] = {"categories": cats, "models": spec["models"],
                                  "n": {d: {c: int((data[spec['models'][0]][d]["labels"] == c).sum()) for c in cats} for d in ("overhead", "uav")},
                                  "point": point}
        # crop-level predictions with every candidate score
        for m in spec["models"]:
            for d in ("overhead", "uav"):
                x = data[m][d]
                scores, pred = classify(x["feats"], text[m][task], cats)
                for i in range(len(pred)):
                    pred_rows.append([task, m, d, x["crop_ids"][i], x["images"][i], x["sequences"][i], x["labels"][i], pred[i],
                                      int(pred[i] == x["labels"][i])] + [f"{s:.6f}" for s in scores[i]] + [""] * (4 - len(cats)))
                acc_csv = float(np.mean(pred == x["labels"]))
                checks.append((f"{task} {m} {d}: accuracy from predictions matches metric", abs(acc_csv - point[m][d]["accuracy"]) < 1e-12))
        log(f"{task}: point estimates done")

        if args.bootstrap:
            rng = np.random.default_rng(SEED)
            ref = data[spec["models"][0]]
            boot = {}
            for level, key in (("image", "images"), ("sequence", "sequences")):
                reps = []
                for _ in range(N_BOOT):
                    w = {"overhead": cluster_weights(rng, ref["overhead"]["images"]), "uav": cluster_weights(rng, ref["uav"][key])}
                    reps.append(evaluate(task, data, text, w))
                boot[level] = summarize_boot(reps, spec["models"], cats)
                log(f"{task}: {level}-level bootstrap done")
            results["tasks"][task]["bootstrap"] = boot

    with open(os.path.join(OUT_DIR, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["task", "model", "domain", "crop_id", "source_image", "cluster_sequence", "true_label", "predicted_label",
                    "correct", "score_1", "score_2", "score_3", "score_4"])
        w.writerows(pred_rows)
    results["score_columns"] = {t: TASKS[t]["categories"] for t in TASKS}
    results["checks"] = [{"check": n, "passed": ok} for n, ok in checks]
    results["all_checks_passed"] = all(ok for _, ok in checks)
    name = "classification_bootstrap.json" if args.bootstrap else "classification.json"
    with open(os.path.join(OUT_DIR, name), "w") as f:
        json.dump(results, f, indent=2)
    write_table(results, args.bootstrap)
    failed = [n for n, ok in checks if not ok]
    log(f"checks: {len(checks) - len(failed)}/{len(checks)} passed" + (f"; FAILED: {failed}" if failed else ""))
    if failed:
        sys.exit(1)


def summarize_boot(reps, models, cats):
    s = {}
    for m in models:
        s[m] = {"accuracy_gap_pp": ci([r[m]["gap"]["accuracy_pp"] for r in reps]),
                "macro_f1_gap_pp": ci([r[m]["gap"]["macro_f1_pp"] for r in reps])}
        for d in ("overhead", "uav"):
            s[m][f"accuracy_{d}"] = ci([r[m][d]["accuracy"] for r in reps])
        for c in cats:
            s[m][c] = {"recall_gap_pp": ci([r[m]["gap"]["recall_pp"][c] for r in reps]),
                       "A": ci([r[m]["gap"]["A"][c] for r in reps if r[m]["gap"]["A"][c] is not None]),
                       "delta_m": ci([r[m]["gap"]["delta_m"][c] for r in reps if r[m]["gap"]["delta_m"][c] is not None])}
            for d in ("overhead", "uav"):
                s[m][c][f"recall_{d}"] = ci([r[m][d]["per_category"][c]["recall"] for r in reps])
                s[m][c][f"margin_{d}"] = ci([r[m][d]["text"][c]["margin"] for r in reps if r[m][d]["text"][c] is not None])
        if m != "clip":
            s[m]["accuracy_gap_diff_vs_clip_pp"] = ci([r[m]["gap"]["accuracy_pp"] - r["clip"]["gap"]["accuracy_pp"] for r in reps])
    return s


def write_table(results, with_boot):
    L = ["# Ground-truth crop classification and text alignment", "",
         f"Prompt: \"{results['prompt']}\". Zero-shot argmax over the task's candidates. "
         "Gap = overhead - UAV (positive = worse on UAV). delta_m > 0 = smaller UAV centroid margin.",
         "Fine-tuned CLIP was trained on the UAV side of P1 (other images) - its UAV numbers are not zero-shot.",
         "A_c is an absolute value, so its bootstrap interval cannot cross 0 and can sit above a near-zero point estimate; "
         "read the sign and size of a_overhead - a_UAV from the a columns instead.", ""]
    for task, r in results["tasks"].items():
        cats = r["categories"]
        L += [f"## {task}  candidates: {', '.join(cats)}", "", "| Model | Acc overhead | Acc UAV | Acc gap (pp) | Macro-F1 overhead | Macro-F1 UAV |",
              "|---|---|---|---|---|---|"]
        for m in r["models"]:
            p = r["point"][m]
            gap = f"{p['gap']['accuracy_pp']:+.1f}"
            if with_boot:
                b = r["bootstrap"]["image"][m]["accuracy_gap_pp"]
                gap += f" [{b[0]:+.1f}, {b[1]:+.1f}]"
            L.append(f"| {MODEL_NAMES[m]} | {100*p['overhead']['accuracy']:.1f}% | {100*p['uav']['accuracy']:.1f}% | {gap} | "
                     f"{p['overhead']['macro_f1']:.3f} | {p['uav']['macro_f1']:.3f} |")
        L += ["", "| Category | Model | Recall overhead | Recall UAV | Recall gap (pp) | a overhead | a UAV | A_c | margin overhead | margin UAV | delta_m |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in cats:
            for m in r["models"]:
                p = r["point"][m]
                rg, dm = f"{p['gap']['recall_pp'][c]:+.1f}", f"{p['gap']['delta_m'][c]:+.4f}"
                if with_boot:
                    b = r["bootstrap"]["image"][m][c]
                    rg += f" [{b['recall_gap_pp'][0]:+.1f}, {b['recall_gap_pp'][1]:+.1f}]"
                    dm += f" [{b['delta_m'][0]:+.4f}, {b['delta_m'][1]:+.4f}]"
                L.append(f"| {c} | {MODEL_NAMES[m]} | {100*p['overhead']['per_category'][c]['recall']:.1f}% | {100*p['uav']['per_category'][c]['recall']:.1f}% | "
                         f"{rg} | {p['overhead']['text'][c]['a']:.4f} | {p['uav']['text'][c]['a']:.4f} | {p['gap']['A'][c]:.4f} | "
                         f"{p['overhead']['text'][c]['margin']:+.4f} | {p['uav']['text'][c]['margin']:+.4f} | {dm} |")
        L += ["", "Confusion matrices (rows = true, columns = predicted, order: " + ", ".join(cats) + "):", ""]
        for m in r["models"]:
            for d in ("overhead", "uav"):
                L.append(f"- {MODEL_NAMES[m]}, {d}: {r['point'][m][d]['confusion']}")
        L.append("")
    name = "classification_bootstrap_table.md" if with_boot else "classification_table.md"
    with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
