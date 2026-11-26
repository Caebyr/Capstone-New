"""
Checkpoint C2: same-backbone control. RemoteCLIP is ViT-L/14; the paper's CLIP is ViT-B/32. Is RemoteCLIP's
difference from CLIP due to remote-sensing training, or just a bigger model? Add OpenAI CLIP ViT-L/14 (the same
architecture RemoteCLIP fine-tunes) on exactly RemoteCLIP's full-scale crops.

  python encode_full_scale.py encode --model clip_l14   # GPU, batch 4: embeddings/full_scale/clip_l14_feats.npy
  python same_backbone.py                              # CPU (text prompts encoded on CPU): results/full_scale/same_backbone.{json,md}

For P1, P1a, P2+P3 (and van): full-cache drift (cos, L2, ratio, Y), zero-shot accuracy, balanced accuracy and the
text margin gap delta_m, for CLIP B/32, CLIP L/14 and RemoteCLIP L/14; image-bootstrap 95% CIs, with the paired
contrasts "RemoteCLIP vs CLIP-L/14" (= effect of remote-sensing training at fixed architecture) and
"CLIP-L/14 vs CLIP-B/32" (= effect of model size at fixed training data). Also the five prompt templates of A5 and
the A4 same-domain baseline levels, for CLIP-L/14.
"""
import json
import os
import time

import numpy as np

from drift_full_scale import TASKS as BASE_TASKS, build, classification, load_cache
from drift_stats import task_stats
from same_domain_baseline import comparisons, side
from src.stats import ci, cluster_weights, drift

OUT, EMB = "results/full_scale", "embeddings/full_scale"
CLIP_L14, CLIP_L14_REV = "openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"
MODELS = ["clip", "clip_l14", "remoteclip"]
NAMES = {"clip": "CLIP ViT-B/32", "clip_l14": "CLIP ViT-L/14", "remoteclip": "RemoteCLIP ViT-L/14"}
TEMPLATES = ["a photo of a {}", "a satellite photo of a {}", "an aerial photo of a {}", "a drone photo of a {}", "{}"]
WORDS = {"small-vehicle": "small vehicle", "large-vehicle": "large vehicle"}
N_BOOT, N_BOOT_BASE, SEED = 500, 300, 20261130
TASKS = {t: dict(s, models=MODELS) for t, s in BASE_TASKS.items() if t in ("P1", "P1a", "P2+P3", "P2van")}
for _t, _s in TASKS.items():  # build() / classification() look tasks up in drift_full_scale.TASKS
    BASE_TASKS[f"_sb_{_t}"] = _s


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def l14_text():
    """{task: {template: [K, D]}} for CLIP-L/14, encoded on CPU (no GPU contention), cached."""
    path = f"{OUT}/clip_l14_text_embeddings.npz"
    if not os.path.exists(path):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained(CLIP_L14, revision=CLIP_L14_REV).eval()
        proc = CLIPProcessor.from_pretrained(CLIP_L14, revision=CLIP_L14_REV)
        out = {}
        with torch.no_grad():
            for t in ("P1", "P2+P3"):
                for tp in TEMPLATES:
                    e = model.get_text_features(**proc(text=[tp.format(WORDS.get(c, c)) for c in BASE_TASKS[t]["cats"]],
                                                       return_tensors="pt", padding=True)).float()
                    out[f"{t}||{tp}"] = (e / e.norm(dim=-1, keepdim=True)).numpy().astype(np.float64)
        np.savez(path, **out)
    z = np.load(path)
    return {t: {tp: z[f"{t}||{tp}"] for tp in TEMPLATES} for t in ("P1", "P2+P3")}


def text_banks():
    """{model: {task: {template: [K, D]}}}; CLIP / RemoteCLIP from A5's prompt bank (same prompts, same order)."""
    z = np.load(f"{OUT}/prompt_text_embeddings.npz")
    bank = {m: {t: {tp: z[f"{m}||{t}||{tp}"] for tp in TEMPLATES} for t in ("P1", "P2+P3")} for m in ("clip", "remoteclip")}
    bank["clip_l14"] = l14_text()
    # the paper prompt in A3's own text file must match A5's bank (guards against a silent prompt mismatch)
    a3 = np.load(f"{OUT}/text_embeddings.npz")
    for m in ("clip", "remoteclip"):
        for t in ("P1", "P2+P3"):
            assert np.allclose(a3[f"{m}|{t}"], bank[m][t]["a photo of a {}"], atol=1e-5), (m, t)
    return bank


def feats_with_l14():
    man, member, feats = load_cache()
    ids = open(f"{EMB}/clip_l14_box_ids.txt").read().split()
    feats["clip_l14"] = (np.load(f"{EMB}/clip_l14_feats.npy").astype(np.float64), {b: i for i, b in enumerate(ids)})
    return man, member, feats


def balanced(res, cats):
    return float(np.mean([res["per_category"][c]["recall"] for c in cats]))


def summarize_cls(cls, cats):
    return {m: {"acc_overhead": cls[m]["overhead"]["accuracy"], "acc_uav": cls[m]["uav"]["accuracy"],
                "bal_overhead": balanced(cls[m]["overhead"], cats), "bal_uav": balanced(cls[m]["uav"], cats),
                "delta_m": cls[m]["gap"]["delta_m"],
                "margin_overhead": {c: cls[m]["overhead"]["text"][c]["margin"] for c in cats},
                "margin_uav": {c: cls[m]["uav"]["text"][c]["margin"] for c in cats},
                "pred_counts_on_vehicles_overhead": cls[m]["overhead"]["pred_counts_on_vehicles"]} for m in cls}


def main():
    t0 = time.time()
    man, member, feats = feats_with_l14()
    bank = text_banks()
    text = {m: {t: bank[m][t]["a photo of a {}"] for t in ("P1", "P2+P3")} for m in MODELS}
    rng = np.random.default_rng(SEED)
    res = {"settings": {"n_boot": N_BOOT, "n_boot_baseline": N_BOOT_BASE, "seed": SEED,
                        "clip_l14": {"model": CLIP_L14, "revision": CLIP_L14_REV}}, "tasks": {}, "prompts": {}, "baseline": []}
    for task, spec in TASKS.items():
        key = f"_sb_{task}"
        data = build(key, man, member, feats)
        cats = spec["cats"]
        n = {d: {c: int((data["clip"][d]["labels"] == c).sum()) for c in cats} for d in ("overhead", "uav")}
        ones = {d: np.ones(len(data["clip"][d]["labels"])) for d in ("overhead", "uav")}
        full = task_stats(data, cats, MODELS, ones["overhead"], ones["uav"])
        cls = summarize_cls(classification(key, data, text), cats) if spec["text_task"] else None
        B = {m: {c: {"cos": [], "l2": [], "ratio": [], "Y": []} for c in cats} for m in MODELS}
        D = {k: {c: [] for c in cats} for k in ("remoteclip_vs_clip_l14", "clip_l14_vs_clip")}
        BC = {m: {"acc_overhead": [], "acc_uav": [], "bal_overhead": [], "bal_uav": [], **{f"dm_{c}": [] for c in cats}} for m in MODELS}
        BCD = {"remoteclip_vs_clip_l14_bal_overhead": [], "clip_l14_vs_clip_bal_overhead": []}
        for b in range(N_BOOT):
            w = {d: cluster_weights(rng, data["clip"][d]["images"]) for d in ("overhead", "uav")}
            s = task_stats(data, cats, MODELS, w["overhead"], w["uav"])
            for m in MODELS:
                for c in cats:
                    if s[m][c] is None:
                        continue
                    for k in ("cos", "l2", "ratio", "Y"):
                        if k in s[m][c]:
                            B[m][c][k].append(s[m][c][k])
            for c in cats:
                if all(s[m][c] is not None for m in MODELS):
                    D["remoteclip_vs_clip_l14"][c].append(100 * (s["remoteclip"][c]["ratio"] / s["clip_l14"][c]["ratio"] - 1))
                    D["clip_l14_vs_clip"][c].append(100 * (s["clip_l14"][c]["ratio"] / s["clip"][c]["ratio"] - 1))
            if cls is not None and b < N_BOOT // 2:
                cb = summarize_cls(classification(key, data, text, w), cats)
                for m in MODELS:
                    for k in ("acc_overhead", "acc_uav", "bal_overhead", "bal_uav"):
                        BC[m][k].append(cb[m][k])
                    for c in cats:
                        BC[m][f"dm_{c}"].append(cb[m]["delta_m"][c])
                BCD["remoteclip_vs_clip_l14_bal_overhead"].append(100 * (cb["remoteclip"]["bal_overhead"] - cb["clip_l14"]["bal_overhead"]))
                BCD["clip_l14_vs_clip_bal_overhead"].append(100 * (cb["clip_l14"]["bal_overhead"] - cb["clip"]["bal_overhead"]))
        res["tasks"][task] = {
            "n_crops": n, "full": full, "classification": cls,
            "drift_ci": {m: {c: {k: ci(v) for k, v in B[m][c].items() if v} for c in cats} for m in MODELS},
            "ratio_change_pct_ci": {k: {c: ci(v) for c, v in d.items() if v} for k, d in D.items()},
            "classification_ci": {m: {k: ci(v) for k, v in d.items() if v} for m, d in BC.items()} if cls else None,
            "balanced_overhead_diff_pp_ci": {k: ci(v) for k, v in BCD.items() if v} if cls else None}
        log(f"{task}: " + " | ".join(f"{m} " + ", ".join(f"{c} ratio {full[m][c]['ratio']:.3f}" for c in cats) for m in MODELS))
        if task in ("P1", "P2+P3"):
            from classify_crops import class_metrics, classify
            res["prompts"][task] = {}
            for m in MODELS:
                res["prompts"][task][m] = {}
                for tp in TEMPLATES:
                    r = {}
                    for d in ("overhead", "uav"):
                        _, pred = classify(data[m][d]["feats"], bank[m][task][tp], cats)
                        cm = class_metrics(data[m][d]["labels"], pred, cats)
                        r[d] = {"acc": cm["accuracy"], "bal": float(np.mean([cm["per_category"][c]["recall"] for c in cats]))}
                    res["prompts"][task][m][tp] = r
    # A4 same-domain baseline, all three encoders (CLIP / RemoteCLIP values recomputed here so the bootstrap is shared)
    for level, name, cat, a, b in comparisons(man, member):
        rec = {"level": level, "comparison": name, "category": cat, "n": [len(a), len(b)], "models": {}}
        for m in MODELS:
            fa, ia = side(a, man, feats, m)
            fb, ib = side(b, man, feats, m)
            pt = drift(fa, np.ones(len(fa)), fb, np.ones(len(fb)))
            bs = [drift(fa, cluster_weights(rng, ia), fb, cluster_weights(rng, ib))["cos"] for _ in range(N_BOOT_BASE)]
            rec["models"][m] = {"cos": pt["cos"], "ratio": pt["ratio"], "cos_ci": ci(bs)}
        res["baseline"].append(rec)
    log("baseline levels done")
    res["seconds"] = round(time.time() - t0)
    with open(f"{OUT}/same_backbone.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    write_md(res)
    log(f"wrote {OUT}/same_backbone.json and .md ({res['seconds']} s)")


def write_md(R):
    f3 = lambda v: f"{v:.3f}"
    cif = lambda c, nd=3: f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]"
    L = ["# Same-backbone control: CLIP ViT-L/14 vs RemoteCLIP ViT-L/14 (C2)", "",
         "RemoteCLIP is CLIP ViT-L/14 further trained on remote-sensing image-text pairs. Comparing it with OpenAI's CLIP "
         "ViT-L/14 on the same crops isolates the effect of that training; comparing CLIP-L/14 with the paper's CLIP-B/32 "
         "isolates model size. Full-scale cache (RemoteCLIP's crops), prompt \"a photo of a {}\" unless stated. "
         f"95% image-bootstrap CIs ({R['settings']['n_boot']} reps; classification on half of them).", ""]
    for task, T in R["tasks"].items():
        cats = list(T["full"]["clip"])
        L += [f"## {task}", "", "| Category | " + " | ".join(f"{NAMES[m]} ratio [CI]" for m in MODELS)
              + " | RemoteCLIP vs CLIP-L/14, ratio change % [CI] | CLIP-L/14 vs CLIP-B/32, ratio change % [CI] |", "|---|" + "---|" * 5]
        contrasts = (("remoteclip_vs_clip_l14", "remoteclip", "clip_l14"), ("clip_l14_vs_clip", "clip_l14", "clip"))
        for c in cats:
            cells = [f"{f3(T['full'][m][c]['ratio'])} {cif(T['drift_ci'][m][c]['ratio'])}" for m in MODELS]
            for k, a, b in contrasts:
                pt = 100 * (T["full"][a][c]["ratio"] / T["full"][b][c]["ratio"] - 1)
                cells.append(f"{pt:+.1f} {cif(T['ratio_change_pct_ci'][k][c], 1)}")
            L.append(f"| {c} | " + " | ".join(cells) + " |")
        L += ["", "| Category | " + " | ".join(f"{NAMES[m]} cos drift" for m in MODELS) + " |" + (" " + " | ".join(f"{NAMES[m]} Y" for m in MODELS) + " |" if "Y" in T["full"]["clip"][cats[0]] else ""),
              "|---|" + "---|" * (3 + (3 if "Y" in T["full"]["clip"][cats[0]] else 0))]
        for c in cats:
            row = f"| {c} | " + " | ".join(f"{T['full'][m][c]['cos']:.4f}" for m in MODELS) + " |"
            if "Y" in T["full"]["clip"][c]:
                row += " " + " | ".join(f"{T['full'][m][c]['Y']:.2f} {cif(T['drift_ci'][m][c]['Y'], 2)}" for m in MODELS) + " |"
            L.append(row)
        if T["classification"]:
            C, CC = T["classification"], T["classification_ci"]
            L += ["", "| Encoder | accuracy O / U | balanced accuracy O [CI] | balanced accuracy U [CI] | " + " | ".join(f"Δm {c} [CI]" for c in cats) + " |",
                  "|---|---|---|---|" + "---|" * len(cats)]
            for m in MODELS:
                L.append(f"| {NAMES[m]} | {100 * C[m]['acc_overhead']:.1f} / {100 * C[m]['acc_uav']:.1f} | "
                         f"{100 * C[m]['bal_overhead']:.1f} [{100 * CC[m]['bal_overhead'][0]:.1f}, {100 * CC[m]['bal_overhead'][1]:.1f}] | "
                         f"{100 * C[m]['bal_uav']:.1f} [{100 * CC[m]['bal_uav'][0]:.1f}, {100 * CC[m]['bal_uav'][1]:.1f}] | "
                         + " | ".join(f"{C[m]['delta_m'][c]:+.4f} {cif(CC[m][f'dm_{c}'], 4)}" for c in cats) + " |")
            d = T["balanced_overhead_diff_pp_ci"]
            L += ["", f"Overhead balanced accuracy, RemoteCLIP − CLIP-L/14: "
                  f"{100 * (C['remoteclip']['bal_overhead'] - C['clip_l14']['bal_overhead']):+.1f} pp {cif(d['remoteclip_vs_clip_l14_bal_overhead'], 1)}; "
                  f"CLIP-L/14 − CLIP-B/32: {100 * (C['clip_l14']['bal_overhead'] - C['clip']['bal_overhead']):+.1f} pp {cif(d['clip_l14_vs_clip_bal_overhead'], 1)}."]
            pv = {m: max(C[m]["pred_counts_on_vehicles_overhead"], key=C[m]["pred_counts_on_vehicles_overhead"].get) for m in MODELS}
            L += ["", "Most common label given to overhead vehicle crops: " + "; ".join(f"{NAMES[m]} '{pv[m]}'" for m in MODELS) + "."]
        L.append("")
    L += ["## Prompt templates (full cache; balanced accuracy overhead / UAV, %)", ""]
    for task, P in R["prompts"].items():
        L += [f"**{task}**", "", "| Template | " + " | ".join(NAMES[m] for m in MODELS) + " |", "|---|---|---|---|"]
        for tp in TEMPLATES:
            L.append(f"| `{tp}` | " + " | ".join(f"{100 * P[m][tp]['overhead']['bal']:.1f} / {100 * P[m][tp]['uav']['bal']:.1f}" for m in MODELS) + " |")
        L.append("")
    L += ["## Same-domain baseline levels (A4), cosine drift [95% CI]", "", "| Level | Comparison | Category | " + " | ".join(NAMES[m] for m in MODELS) + " |", "|---|---|---|---|---|---|"]
    for r in R["baseline"]:
        L.append(f"| {r['level']} | {r['comparison']} | {r['category']} | " + " | ".join(f"{r['models'][m]['cos']:.4f} {cif(r['models'][m]['cos_ci'], 4)}" for m in MODELS) + " |")
    with open(f"{OUT}/same_backbone.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
