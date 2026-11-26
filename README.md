# Visual Drift of Vision-Language Embeddings Between Overhead and UAV Imagery

Capstone project (CPG No. 141), Thapar Institute of Engineering and Technology, Patiala. Mentor: Dr. Geeta Kasana.
Team: Anoop Singh Sachdev · Arnav Gupta · Kabir Oberoi · Khushpreet Kaur · Siddhant Saxena

We measure how far a category's embedding centroid moves between overhead imagery (DOTA, xView) and UAV imagery (VisDrone, UAVid), using one frozen encoder per comparison: CLIP ViT-B/32, RemoteCLIP ViT-L/14, and a CLIP fine-tuned on held-out VisDrone images. We then test whether that movement goes with a loss of zero-shot recognition.

- **New to the project?** Read [PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md), a plain-English account of every concept, step, result and mistake fixed.
- **The paper:** [docs/paper/RAIT_paper_submitted.pdf](docs/paper/RAIT_paper_submitted.pdf), submitted to RAIT 2027 and under review.

## Status (26 Nov 2026)

**In the paper** (300 ground-truth crops per category and domain, seed 42):
- Drift for six categories in three comparisons: P1 DOTA–VisDrone (small and large vehicle, three encoders), P2 xView–VisDrone (car, bus, truck) and P3 xView–UAVid (building).
- Image-level permutation tests and bootstrap intervals.
- Zero-shot crop classification and the text-alignment measures A_c and Δm_c.
- A crop manifest that links every cached embedding to its source image and box.

**Added in November** (details: [results/NEW_EXPERIMENTS_2026-11.md](results/NEW_EXPERIMENTS_2026-11.md)):
- The paper's claims re-tested on the full datasets (up to 5,000 crops per category and domain) and over 20 seeds. All hold.
- A same-domain baseline, a resolution-matching control and five prompt templates.
- Zero-shot detection on a five-set overhead/UAV benchmark, with COCO AP and found-vs-named diagnostics:
  - OWLv2;
  - Grounding DINO;
  - SAM proposals named by CLIP / RemoteCLIP.
- Drift measured inside OWLv2's embedding space, linked to per-category detection loss.
- Follow-up controls:
  - an object-level test of text margin vs detection, with size held fixed (28,000 objects);
  - a same-backbone encoder (OpenAI CLIP ViT-L/14);
  - a blur / colour control with a placebo.
- Limitations are listed as threats to validity in `PROJECT_EXPLAINED.md` §9.

**Next** (see `PROJECT_EXPLAINED.md` §10): viewing angle and altitude within one dataset (UAVDT), and adaptation aimed at the text margin.

## Repository layout

```
README.md, PROJECT_EXPLAINED.md   start here
environment.yml, requirements.txt the pinned `hf` environment

src/
  dataset.py        ground-truth crop extraction (DOTA, VisDrone, xView streaming, UAVid masks), label mappings, VisDrone 80/20 split
  embeddings.py     CLIP / RemoteCLIP loading, batched encoding, fine-tuning helpers
  crop_sets.py      where each cached embedding set lives; task definitions for the October analyses
  stats.py          image-level permutation and bootstrap helpers
  full_scale.py     full box pools (same filters as dataset.py), local xView parquet reader, seeded sampling
  detection_data.py COCO-format detection benchmark (DOTA tiles, VisDrone held-out, xView val, UAVid)
  detection_eval.py pycocotools scoring, class-agnostic AP, found-vs-named diagnostics (+ self-test)

run_remoteclip_vs_clip.py   P1 embeddings + drift (CLIP, RemoteCLIP)           -> embeddings/remoteclip_vs_clip, results/remoteclip_vs_clip
run_xview_vs_visdrone.py    P2 embeddings + drift (car, bus, truck)            -> embeddings/xview_vs_visdrone, results/xview_vs_visdrone
run_uavid_building.py       P3 embeddings + drift (building), merged into 4-class results
build_crop_manifests.py     replays crop extraction, writes + verifies crop manifests -> results/crop_manifests
drift_stats.py              permutation tests, image/sequence bootstrap            -> results/drift_stats
classify_crops.py           zero-shot crop classification, A_c, Δm_c (--bootstrap) -> results/classification
join_results.py             drift x classification x text-alignment table          -> results/joined
verify_crop_identity.py     re-encodes every crop and matches it to its cached row -> results/paper_facts
collect_paper_facts.py      evidence collection for the paper fact sheet           -> results/paper_facts

November (full scale, controls, detection; see results/NEW_EXPERIMENTS_2026-11.md):
build_full_pools.py         A1 every eligible GT box in every dataset           -> results/full_scale/pools
encode_full_scale.py        A2 sample <=5,000 crops per category/domain, encode -> embeddings/full_scale
drift_full_scale.py         A3 paper claims at full scale and over 20 seeds     -> results/full_scale
same_domain_baseline.py     A4 noise floor / same platform / cross platform     -> results/full_scale
controls_a5.py              A5 resolution-matched UAV crops, prompt templates   -> results/full_scale
build_detection_sets.py     B1 COCO ground truth for the 5 benchmark sets       -> data/processed/detection, results/detection
run_detectors.py            B2 OWLv2 / Grounding DINO zero-shot detections      -> data/processed/detection/dets
eval_detection.py           B2/B3 COCO AP, found-vs-named, bootstrap CIs        -> results/detection
sam_pipeline.py             B4 SAM proposals named by CLIP / RemoteCLIP         -> results/detection
drift_detection_link.py     B5 drift in OWLv2's space; drift vs detection loss  -> results/detection
object_level.py             C1 per-object text margin vs detection, size held fixed -> results/detection
same_backbone.py            C2 CLIP ViT-L/14 vs RemoteCLIP (encode: encode_full_scale.py --model clip_l14) -> results/full_scale
controls_quality.py         C3 blur / colour matched UAV crops, UAV-donor placebo -> results/full_scale
scripts/plots/              mid-semester plots (yardstick, drift ratio, PCA/t-SNE, % of CLIP)

notebooks/
  01_phase1_reproduction.ipynb    local reproduction of the original cloud Phase-1 experiment
  03_remoteclip.ipynb             loads P1 results, CLIP vs RemoteCLIP analysis and plot
  04_finetune_visdrone_clip.ipynb fine-tuned CLIP arm (held-out split), three-way comparison

results/            all result JSON/CSV/tables/figures (committed); results/archive/ holds superseded ones
final_premst_results/  paper-ready figures + make_final_figures.py (the paper's t-SNE figure comes from here)
configs/experiment.yaml   Phase-1 reproduction configuration
diagrams/           draw.io sources for the midsem report diagrams
docs/               working log and rules (PROJECT_CONTEXT.md), technical report, midsem material, paper material

data/  models/  embeddings/    git-ignored (multi-GB); see below
```

## Environment setup (from scratch)

1. Install Miniconda, then once, in Anaconda Prompt: `conda init powershell`. Reopen VS Code.
2. From the repo root:
   ```
   conda env create -f environment.yml
   conda activate hf
   python -m ipykernel install --user --name hf --display-name "Python (hf)"
   python -m ipykernel install --user --name hf-capstone --display-name "Python (hf-capstone)"
   ```
3. Check: `python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"` should print `2.5.1+cu118 True 4.49.0`.
4. In VS Code, run **Python: Select Interpreter** and pick the `hf` env. In notebooks, pick the kernel **Python (hf)** (notebook 03 uses **Python (hf-capstone)**).

Pinned on purpose: `torch 2.5.1+cu118`, `transformers 4.49.0` (newer versions refuse to `torch.load` this CLIP checkpoint under torch < 2.6), `scikit-learn 1.3.2`, `numpy 1.26.4`. Don't upgrade these without a demonstrated need.

GPU: RTX 3050 Laptop, 4 GB. Keep batch sizes at 1–4, use `torch.no_grad()`, and load one encoder at a time.

Hugging Face cache: the CLIP revision used throughout (`openai/clip-vit-base-patch32` @ `3d74acf9…`) and the xView/UAVid caches are kept at `D:\C drive data\Kabir Oberoi\.cache\huggingface`. Point `HF_HOME` there to reuse them instead of downloading again.

## Data and models (not in git)

```
data/raw/DOTA/images/, labelTxt/           part1 (469 images, the paper's set); labels for all parts
data/raw/DOTA/images_part2_part3/          parts 2-3, extracted Nov 2026 (used only by the full-scale scripts)
data/raw/VisDrone/VisDrone2019-DET-train/  images/ + annotations/ (6,471)
data/raw/UAVid/modified_uavid_dataset/     train_data, val_data, test_data (Kaggle mirror)
xView                                      HichTala/xview parquet in the HF cache (pinned revision); the paper's scripts stream it
models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt
models/finetuned_clip_visdrone_heldout/finetuned_clip_visdrone_heldout_epoch3.pt
embeddings/{phase1_lightning,phase1_local,remoteclip_vs_clip,xview_vs_visdrone}/   the paper's cached *_feats.npy / *_labels.npy
embeddings/full_scale/                     November full-scale, control and OWLv2-space embeddings
data/processed/detection/                  detection benchmark images, COCO ground truth, detections, SAM proposals
```

Checkpoint checksums and dataset revisions are in `results/NEW_EXPERIMENTS_2026-10.md`; detector, SAM and CLIP-L revisions in `results/detection/model_revisions.json`. Set `HF_HUB_OFFLINE=1` to load everything from the cache.

## Reproducing

Run everything from the repo root with the `hf` env active. Heavy GPU jobs are plain scripts, not notebooks: an earlier notebook-driven run hung (see `docs/PROJECT_CONTEXT.md` §7b).

```
python run_remoteclip_vs_clip.py        # P1 (fine-tuned arm: notebooks/04_finetune_visdrone_clip.ipynb)
python run_xview_vs_visdrone.py         # P2
python run_uavid_building.py            # P3
python build_crop_manifests.py          # ~25 min, GPU, streams xView
python drift_stats.py                   # ~13 min, CPU
python classify_crops.py --bootstrap    # ~7 min
python join_results.py
```

The November full-scale, control and detection runs (commands and run times): [results/NEW_EXPERIMENTS_2026-11.md](results/NEW_EXPERIMENTS_2026-11.md).

All reported numbers come from the cached embeddings. Re-extracting them with today's pip Pillow (libjpeg-turbo) shifts JPEG-derived values slightly, within the reported intervals. See `docs/paper/CLARIFICATIONS_FOR_PAPER.md`, item 10.
