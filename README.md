# Visual Drift of Vision-Language Embeddings Between Overhead and UAV Imagery

Capstone project (CPG No. 141), Thapar Institute of Engineering and Technology, Patiala. Mentor: Dr. Geeta Kasana.
Team: Anoop Singh Sachdev · Arnav Gupta · Kabir Oberoi · Khushpreet Kaur · Siddhant Saxena

We measure how far a category's embedding centroid moves between overhead imagery (DOTA, xView) and UAV imagery (VisDrone, UAVid), using one frozen encoder per comparison: CLIP ViT-B/32, RemoteCLIP ViT-L/14, and a CLIP fine-tuned on held-out VisDrone images. We then test whether that movement goes with a loss of zero-shot recognition.

- **New to the project?** Read [PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md), a plain-English account of every concept, step, result and mistake fixed.
- **The paper:** [docs/paper/RAIT_paper_submitted.pdf](docs/paper/RAIT_paper_submitted.pdf), submitted to RAIT and under review.

## Status (25 Nov 2026)

**Done** (300 ground-truth crops per category and domain, seed 42):
- Drift for six categories in three comparisons: P1 DOTA–VisDrone (small and large vehicle, three encoders), P2 xView–VisDrone (car, bus, truck) and P3 xView–UAVid (building).
- Image-level permutation tests and bootstrap intervals.
- Zero-shot crop classification and the text-alignment measures A_c and Δm_c.
- A crop manifest that links every cached embedding to its source image and box.

**Next** (see `PROJECT_EXPLAINED.md` §10):
- a same-domain baseline;
- a same-backbone encoder (CLIP ViT-L/14);
- multi-seed and full-dataset runs;
- controlled altitude and resolution tests;
- detection experiments.

## Repository layout

```
README.md, PROJECT_EXPLAINED.md   start here
environment.yml, requirements.txt the pinned `hf` environment

src/
  dataset.py        ground-truth crop extraction (DOTA, VisDrone, xView streaming, UAVid masks), label mappings, VisDrone 80/20 split
  embeddings.py     CLIP / RemoteCLIP loading, batched encoding, fine-tuning helpers
  crop_sets.py      where each cached embedding set lives; task definitions for the October analyses
  stats.py          image-level permutation and bootstrap helpers

run_remoteclip_vs_clip.py   P1 embeddings + drift (CLIP, RemoteCLIP)           -> embeddings/remoteclip_vs_clip, results/remoteclip_vs_clip
run_xview_vs_visdrone.py    P2 embeddings + drift (car, bus, truck)            -> embeddings/xview_vs_visdrone, results/xview_vs_visdrone
run_uavid_building.py       P3 embeddings + drift (building), merged into 4-class results
build_crop_manifests.py     replays crop extraction, writes + verifies crop manifests -> results/crop_manifests
drift_stats.py              permutation tests, image/sequence bootstrap            -> results/drift_stats
classify_crops.py           zero-shot crop classification, A_c, Δm_c (--bootstrap) -> results/classification
join_results.py             drift x classification x text-alignment table          -> results/joined
verify_crop_identity.py     re-encodes every crop and matches it to its cached row -> results/paper_facts
collect_paper_facts.py      evidence collection for the paper fact sheet           -> results/paper_facts
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
data/raw/DOTA/images/, labelTxt/           part1 (469 images); part2.zip, part3.zip not yet extracted
data/raw/VisDrone/VisDrone2019-DET-train/  images/ + annotations/ (6,471)
data/raw/UAVid/modified_uavid_dataset/     train_data, val_data, test_data (Kaggle mirror)
xView                                      streamed from Hugging Face (HichTala/xview), nothing stored
models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt
models/finetuned_clip_visdrone_heldout/finetuned_clip_visdrone_heldout_epoch3.pt
embeddings/{phase1_lightning,phase1_local,remoteclip_vs_clip,xview_vs_visdrone}/   cached *_feats.npy / *_labels.npy
```

Checkpoint checksums and dataset revisions are in `results/NEW_EXPERIMENTS_2026-10.md`.

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

All reported numbers come from the cached embeddings. Re-extracting them with today's pip Pillow (libjpeg-turbo) shifts JPEG-derived values slightly, within the reported intervals. See `docs/paper/CLARIFICATIONS_FOR_PAPER.md`, item 10.
