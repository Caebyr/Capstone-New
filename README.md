# Semantic Drift in Open-Vocabulary Object Detection Across Altitudes

Capstone project (CPG No. 141), BE Third/Fourth Year COE/CSE, Thapar Institute of Engineering and Technology, Patiala.
Mentor: Dr. Geeta Kasana, Associate Professor, CSE.

**Team:** Anoop Singh Sachdev (Experimentation & Evaluation) · Arnav Gupta (Baseline & Model Implementation) · Kabir Oberoi (Semantic Analysis & Visualization) · Khushpreet Kaur (Data Analysis & Preprocessing) · Siddhant Saxena (Ablation Studies & Vocabulary Formulation)

---

## Research question

Open-vocabulary object detectors (CLIP, GroundingDINO, etc.) assume the relationship between a text concept (e.g. "vehicle") and its visual representation stays consistent across domains. In aerial imagery it doesn't: satellite imagery is high-altitude, low-resolution, and top-down; UAV/drone imagery is lower-altitude, high-resolution, and more oblique. The same nominal class can end up occupying different regions of a vision-language embedding space depending on which domain produced it — **semantic drift**.

This project quantifies that drift between satellite (DOTA) and UAV (VisDrone) imagery using CLIP-family vision-language embeddings, and works toward an altitude-aware alignment mechanism that reduces it. See `PROJECT_CONTEXT.md` for the detailed working log and standing project rules. (The submitted proposal PDF — overview, literature survey, methodology, six-month work plan — isn't in this repo yet; drop it in as `Capstone Proposal.pdf` if you want it version-controlled alongside the code.)

## Status (as of 2026-08-20)

**Done:**
- Phase-1 semantic-drift experiment (Lightning AI Studio, CLIP ViT-B/32 zero-shot, 7-class vocabulary) reproduced locally end-to-end (`notebooks/01_phase1_reproduction.ipynb`) — confirms the base phenomenon isn't a fluke of the original cloud environment.
- A separate Lightning-side attempt to compare RemoteCLIP (DOTA) against a fine-tuned CLIP (VisDrone) via two independently-fit PCA projections was found invalid (two different models, no shared space) and **redone correctly**: same frozen encoder applied to both domains, ground-truth labels, no PCA.
- **DOTA vs. VisDrone, 3-way comparison** (CLIP / RemoteCLIP / a CLIP fine-tuned on VisDrone) — `notebooks/04_finetune_visdrone_clip.ipynb`. Two more issues found and fixed before trusting the numbers: a bicycle→small-vehicle mapping bug, and train/eval leakage in the fine-tuned arm (it had been evaluated on images it was trained on). Fixed via a held-out VisDrone split; corrected results: RemoteCLIP and the fine-tuned model both show consistently lower drift than plain CLIP, on genuinely unseen data. See `PROJECT_CONTEXT.md` §7c for the full numbers.
- **xView vs. VisDrone/UAVid, 4-class comparison** — DOTA structurally caps out at 2 shared classes with VisDrone (small/large-vehicle only), so xView (streamed from Hugging Face, no local download) was integrated for a richer vehicle taxonomy (car/bus/truck), and UAVid (via Kaggle) for a `building` class VisDrone doesn't have at all. RemoteCLIP beats CLIP on all 4 classes; building shows the largest drift and the largest reduction (30%) of anything measured in the project. See `PROJECT_CONTEXT.md` §7d.
- Mid-semester presentation materials written: `PROGRESS_SUMMARY.md` (plain-language writeup) and `PRESENTATION_SCRIPT.md` (spoken script), both kept in sync with the corrected/expanded results above.

**Known issues / open questions:**
- All drift numbers so far are validation-scale (300 crops/class, one seed) — need a seed/sample-size sensitivity check before being treated as final.
- `van` (xView↔VisDrone) measured but excluded — only 25 samples in a 5,000-image xView scan, below the reliability threshold; needs a bigger scan.
- Phase-1's own whole-image drift scores still group images by the model's predicted class, not ground truth (separate from the RemoteCLIP/CLIP work above, which already uses ground truth).
- PCA/t-SNE visualizations from the historical Lightning run haven't been fully audited for whether projections were fit jointly or per-domain.

**Not yet started:** full-dataset DOTA (parts 2–3)/VisDrone/xView scale-up, the altitude-conditioned alignment adapter, final evaluation metrics (mAP, cross-altitude gap), SAM-based detection integration with a teammate's work.

## Repository structure

```
data/
  raw/DOTA/            DOTA images + labelTxt (not committed — see .gitignore)
  raw/VisDrone/         VisDrone2019-DET-train images + annotations (not committed)
  processed/            Derived/processed data (not committed)
models/
  RemoteCLIP/            RemoteCLIP-ViT-L-14.pt checkpoint (not committed)
embeddings/
  phase1_lightning/       Historical Lightning-run embeddings (not committed, large)
  phase1_local/           Local Phase-1 reproduction embeddings (not committed, large)
  remoteclip_vs_clip/     RemoteCLIP-vs-CLIP validation-scale embeddings (not committed, large)
  clip/ remoteclip/ finetuned/   Reserved for future full-scale extraction runs
results/
  phase1_lightning/       Historical reference notebook + figures (committed — small)
  phase1_local/           Local reproduction figures + drift scores (committed — small)
  remoteclip_vs_clip/     RemoteCLIP-vs-CLIP drift comparison + figure (committed — small)
  full_dataset/ figures/  Reserved for later phases
notebooks/
  01_phase1_reproduction.ipynb   Local reproduction of the Lightning Phase-1 pipeline (done)
  02_full_dataset.ipynb           Full DOTA/VisDrone extraction + embedding (planned)
  03_remoteclip.ipynb             RemoteCLIP vs. CLIP comparison, done properly (done, validation-scale)
  04_drift_analysis.ipynb         Final drift analysis across all embeddings (planned)
src/
  dataset.py              DOTA/VisDrone ground-truth crop extraction + class mapping + subsampling
  embeddings.py            RemoteCLIP/CLIP loading + batched encoding with progress logging
  centroids.py metrics.py visualization.py   Reserved for later phases
run_remoteclip_vs_clip.py  Standalone driver for the RemoteCLIP-vs-CLIP comparison (run directly with
                           the hf env's python for real streaming progress -- see note below)
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

## Reproducing the RemoteCLIP-vs-CLIP comparison

```
conda activate hf
python run_remoteclip_vs_clip.py
```

Run this as a plain script, not through `jupyter nbconvert` — an earlier attempt driving the same
extraction/encoding logic through a notebook cell hung for a full 30-minute cell timeout despite
isolated benchmarks showing it should take ~4 minutes, and `nbconvert` gives zero progress
visibility until the whole notebook finishes (see `PROJECT_CONTEXT.md` Section 7b). The script
streams real per-batch throughput to stdout instead. Once it finishes, `notebooks/03_remoteclip.ipynb`
just loads the saved embeddings/results for analysis and plotting (fast, no GPU needed) — rerun
that notebook any time without repeating the encoding step.

## Next steps

Following the proposal's phase order (see `PROJECT_CONTEXT.md` §15 for the full progression):

1. ~~Resolve the RemoteCLIP-vs-fine-tuned-CLIP methodology gap~~ — done, and hardened further (bicycle-mapping + train/eval-leakage fixes, see §7c); check robustness across seeds/sample sizes before treating as final.
2. ~~Expand beyond DOTA's 2-class ceiling~~ — done via xView + UAVid, 4 shared classes now (§7d); recover `van` as a 5th with a bigger xView scan.
3. Reproduce Phase-1's own whole-image drift scores with ground-truth labels instead of model-predicted labels (the RemoteCLIP-vs-CLIP comparison already uses ground truth; Phase-1's original numbers don't yet).
4. Scale to the full DOTA (parts 2–3), VisDrone, and xView datasets once disk space allows.
5. Implement the altitude-conditioned alignment adapter (Phase 5) and evaluate mAP / cross-altitude gap / drift reduction (Phase 6).
6. Integrate with the teammate's SAM-based detection pipeline (real detection, not just classification of pre-cropped regions) to test whether drift predicts detection failures.
7. Final figures, tables, and report writeup.
