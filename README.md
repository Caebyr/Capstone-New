# Semantic Drift in Open-Vocabulary Object Detection Across Altitudes

Capstone project (CPG No. 141), BE Third/Fourth Year COE/CSE, Thapar Institute of Engineering and Technology, Patiala.
Mentor: Dr. Geeta Kasana, Associate Professor, CSE.

**Team:** Anoop Singh Sachdev (Experimentation & Evaluation) · Arnav Gupta (Baseline & Model Implementation) · Kabir Oberoi (Semantic Analysis & Visualization) · Khushpreet Kaur (Data Analysis & Preprocessing) · Siddhant Saxena (Ablation Studies & Vocabulary Formulation)

---

## Research question

Open-vocabulary object detectors (CLIP, GroundingDINO, etc.) assume the relationship between a text concept (e.g. "vehicle") and its visual representation stays consistent across domains. In aerial imagery it doesn't: satellite imagery is high-altitude, low-resolution, and top-down; UAV/drone imagery is lower-altitude, high-resolution, and more oblique. The same nominal class can end up occupying different regions of a vision-language embedding space depending on which domain produced it — **semantic drift**.

This project quantifies that drift between satellite (DOTA) and UAV (VisDrone) imagery using CLIP-family vision-language embeddings, and works toward an altitude-aware alignment mechanism that reduces it. See `PROJECT_CONTEXT.md` for the detailed working log and standing project rules. (The submitted proposal PDF — overview, literature survey, methodology, six-month work plan — isn't in this repo yet; drop it in as `Capstone Proposal.pdf` if you want it version-controlled alongside the code.)

## Status (as of 2026-08-13)

**Done:**
- Phase-1 semantic-drift experiment (50 → 300 images/domain) originally run on Lightning AI Studio using CLIP ViT-B/32 zero-shot whole-image classification against a shared 7-class vocabulary. Found non-trivial drift across every shared class.
- Project migrated from Lightning AI Studio to a local Windows/RTX 3050 setup (conda env `hf`: Python 3.9, PyTorch 2.5.1 + CUDA 11.8, `transformers` pinned to 4.49.0 for local checkpoint-loading compatibility).
- DOTA (part 1, 469 images) and VisDrone (full, 6,471 images) extracted locally.
- Phase-1 reproduced locally end-to-end (`notebooks/01_phase1_reproduction.ipynb`) — drift magnitudes replicate the historical Lightning run at the right order (0.46–0.82 locally vs. 0.53–0.72 historically), though per-class ranking shifts somewhat, which is expected given the labels used are the model's own zero-shot predictions rather than ground truth.
- A separate, larger Lightning-side experiment (`results/phase1_lightning/`) attempted to compare RemoteCLIP embeddings (DOTA) against a fine-tuned CLIP (VisDrone) — **this comparison is currently invalid** (see Known issues below) and its numbers should not be cited yet.

**Known issues / open questions:**
- The large-scale RemoteCLIP-vs-fine-tuned-CLIP drift numbers compare two *different* models via two *independently fit* PCA projections — there is no shared embedding space underlying that comparison, so it conflates real drift with model-identity and PCA-basis artifacts. Needs to be redone with the *same* frozen encoder applied to both domains (matching the proposal's own Phase 6 design: compare "CLIP drift" vs. "RemoteCLIP drift", never mix models across domains within one drift score).
- Phase-1's drift scores currently group images by the model's own predicted class, not ground-truth DOTA/VisDrone annotations — a first-pass estimate, not final evidence.
- PCA/t-SNE visualizations from the historical run haven't been fully audited for whether projections were fit jointly or per-domain (matters for whether the coordinates are comparable).

**Not yet started:** full-dataset DOTA/VisDrone runs, standardized CLIP-vs-RemoteCLIP comparison at scale, the altitude-conditioned alignment adapter (Phase 5 of the proposal), final evaluation metrics (mAP, cross-altitude gap, semantic confusion rate).

## Repository structure

```
data/
  raw/DOTA/            DOTA images + labelTxt (not committed — see .gitignore)
  raw/VisDrone/         VisDrone2019-DET-train images + annotations (not committed)
  processed/            Derived/processed data (not committed)
models/
  RemoteCLIP/            RemoteCLIP-ViT-L-14.pt checkpoint (not committed)
embeddings/
  phase1_lightning/      Historical Lightning-run embeddings (not committed, large)
  phase1_local/           Local Phase-1 reproduction embeddings (not committed, large)
  clip/ remoteclip/ finetuned/   Reserved for future extraction runs
results/
  phase1_lightning/       Historical reference notebook + figures (committed — small)
  phase1_local/           Local reproduction figures + drift scores (committed — small)
  full_dataset/ figures/  Reserved for later phases
notebooks/
  01_phase1_reproduction.ipynb   Local reproduction of the Lightning Phase-1 pipeline
  02_full_dataset.ipynb           Full DOTA/VisDrone extraction + embedding (planned)
  03_remoteclip.ipynb             RemoteCLIP vs. CLIP comparison, done properly (planned)
  04_drift_analysis.ipynb         Final drift analysis across all embeddings (planned)
src/                    Shared pipeline code (dataset/embeddings/metrics/visualization — to be filled in as notebooks stabilize)
configs/
  experiment.yaml        Recorded configuration for the Phase-1 reproduction run
PROJECT_CONTEXT.md       Detailed working log / standing rules for how this project should be developed
Capstone Proposal.pdf    Original submitted proposal (add locally — not yet in this repo)
```

## Environment

Use the `hf` conda environment (do not use `tf`, and don't recreate environments without a demonstrated compatibility issue):

```
conda activate hf
```

Key packages: `torch==2.5.1+cu118`, `transformers==4.49.0` (pinned — newer versions refuse to load this project's CLIP checkpoint under torch<2.6), `open_clip_torch`, `scikit-learn==1.3.2`, `nbconvert`/`nbclient` (for running notebooks headlessly).

GPU: NVIDIA RTX 3050 Laptop, ~4GB VRAM — keep batch sizes small (1–4) and use `torch.no_grad()`.

## Datasets and large files

DOTA and VisDrone are **not** stored in this repository (multi-GB). Expected local layout:

```
data/raw/DOTA/images/          (from part1.zip; part2/part3 not yet extracted — see disk space notes in PROJECT_CONTEXT.md)
data/raw/DOTA/labelTxt/
data/raw/VisDrone/VisDrone2019-DET-train/images/
data/raw/VisDrone/VisDrone2019-DET-train/annotations/
models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt
```

Model checkpoints, raw datasets, and large embedding arrays (`.npy`, `.pt`, `.zip`, `.tar.gz`) are gitignored. Small result artifacts (figures, drift-score JSON) are committed under `results/`.

## Reproducing Phase-1 locally

```
conda activate hf
jupyter nbconvert --to notebook --execute --inplace ^
  --ExecutePreprocessor.kernel_name=hf-capstone --ExecutePreprocessor.timeout=1800 ^
  notebooks/01_phase1_reproduction.ipynb
```

Outputs land in `embeddings/phase1_local/` and `results/phase1_local/`.

## Next steps

Following the proposal's phase order (see `PROJECT_CONTEXT.md` §15 for the full progression):

1. Resolve the RemoteCLIP-vs-fine-tuned-CLIP methodology gap — redo the large-scale comparison with a single shared encoder per comparison.
2. Reproduce Phase-1 with ground-truth labels instead of model-predicted labels, to separate real drift from classification noise.
3. Scale to the full DOTA (parts 2–3) and VisDrone datasets once disk space allows.
4. Run a proper CLIP-vs-RemoteCLIP drift comparison at scale (does domain specialization reduce drift?).
5. Implement the altitude-conditioned alignment adapter (Phase 5) and evaluate mAP / cross-altitude gap / drift reduction (Phase 6).
6. Final figures, tables, and report writeup.
