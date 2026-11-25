# TECHNICAL REPORT — Semantic Drift in Open-Vocabulary Object Detection Across Altitudes

*Repository:* `C:\Users\Kabir Oberoi\Desktop\Capstone` · *branch:* `main` · *HEAD:* `c401c83` (`c401c839550ed632dfb2ef048b62af4f69e7c49a`) · *report generated:* 2026-09-19

*Purpose:* a factual extraction of everything in this repository that a paper on the project would need — exact code, exact values, exact file paths. Nothing here is pre-filtered; deciding what is paper-worthy is left to the reader.

**Contents:** [Preface](#preface-not-one-of-the-ten-numbered-sections) · [1. Repo overview](#1-repo-overview) · [2. Datasets](#2-datasets) · [3. Models](#3-models) · [4. Metric implementations](#4-metric-implementations) · [5. All experiment results](#5-all-experiment-results) · [6. Detection pipeline](#6-detection-pipeline-teammates-sam-based-work) · [7. Figures](#7-figures--visualizations-already-generated) · [8. Known issues, TODOs, limitations](#8-known-issues-todos-limitations) · [9. Chronology](#9-chronology) · [10. Anything else notable](#10-anything-else-notable)

## Preface (not one of the ten numbered sections)

### Provenance tags used throughout

| Tag | Meaning |
|---|---|
| **[REPO]** | Quoted verbatim from a file inside the repository folder (path, and line range where useful, is given). |
| **[LOG]** | Printed output that is stored inside a notebook (the only surviving run logs — no `.log` files exist). |
| **[DERIVED]** | Computed for this report from repo artifacts (cached embeddings, raw annotation files, checkpoints). Not stored anywhere in the repo. The method is stated next to each such item. |
| **[EXTERNAL]** | Comes from outside the repo folder: the proposal PDF and slide deck on the Desktop, the Hugging Face cache on this machine, or a live query of this machine. |
| **[ANALYSIS]** | An observation by the report author about how the code behaves. It is not a claim made anywhere in the repo. |

### How the facts were gathered

All source files, notebooks (code cells *and* saved outputs), result JSONs, git history and file-system timestamps were read. Three read-only verification passes were then run with the project's own `hf` environment: (1) the raw annotation files were re-parsed with the repo's own `src/dataset.py` functions to recount crops (this reproduced the historical counts **9,570** DOTA and **153,406** VisDrone crops exactly, and the fine-tuning train-pool counts printed in notebook 04 exactly); (2) **every** drift number stored in the result JSONs (16 distinct class×encoder entries; 20 comparisons, because the CLIP/RemoteCLIP DOTA entries are stored in two JSON files) was recomputed from the cached `.npy` embeddings — all 20 comparisons match to the stored 4 decimals (max absolute difference 0.0); (3) each fine-tuned checkpoint was diffed tensor-by-tensor against the base CLIP weights. No experiment was re-run and no repo file was modified other than creating this report.

### Quick answers to the items previously flagged as “not yet done / not yet computed”

| Item | Current status (details in the section given) |
|---|---|
| Text-embedding metrics (anchor alignment, semantic margin, or anything similar) | **Not implemented — not even partially.** No code in any tracked file, notebook, branch (only `main`), stash (none) or untracked file computes a similarity between a visual centroid and a text embedding, or a margin against competing categories. `src/metrics.py` is a 0-byte file. The only uses of the text encoder are Phase-1 zero-shot *label assignment* and the contrastive fine-tuning loss. → §4.6 |
| Results using >300 crops per category | **None.** Every embedded evaluation sample is capped at 300/class (xView `van` has only 25). Larger crop *pools* were scanned (e.g. 146,631 VisDrone vehicle crops) but only 300/class were embedded for drift. → §5.3 |
| Results using >1 random seed | **None on disk.** All headline results use `seed=42` (evaluation sampling) and `seed=1234` (VisDrone split). `verify_drift_seeds.py` (seeds 42, 123, 7, 2024, 99) exists, but its output `results/remoteclip_vs_clip/seed_sensitivity.json` **does not exist** and was never committed; and the script as written loads the *leaky* first-run checkpoint and samples from *all* VisDrone images. → §5.3 |
| Results on the full DOTA dataset | **No.** Only `part1.zip` (469 images) + `labelTxt` (1,411 label files, of which 469 are used) are extracted. `part2.zip` and `part3.zip` are on disk, unextracted. → §2.1, §5.3 |
| xView-based results | **Done at validation scale.** car/bus/truck (5,000-image streamed scan of xView `train`, of 20,881 images) and building (500-image scan); 300 crops/class; seed 42; single run. `van` measured but excluded (25 samples < 30 threshold). → §2.3, §5 |
| Detection pipeline (teammate's SAM work) | **Not in this repository.** No SAM/detector code, no mAP numbers, no drift-vs-detection-accuracy correlation anywhere. → §6 |
| Fine-tuned model | CLIP ViT-B/32; last 2 vision-encoder layers + `post_layernorm` + `visual_projection` unfrozen (14,570,496 of 151,277,313 parameters); symmetric InfoNCE on (VisDrone crop, `"a drone photo of a {class}"`) pairs; AdamW lr 1e-5, cosine schedule, batch 8, 3 epochs, 167,699 crops from 5,177 VisDrone images. Two runs exist: a first run with train/eval leakage (superseded) and the held-out run used for the headline numbers. → §3.4 |

---

## 1. Repo overview

### 1.1 Full directory tree

Excluded as requested: `node_modules`, `.git`, `__pycache__`, `venv`/`.venv`, `data/raw`. Tags: **[T]** tracked by git · **[U]** untracked (not ignored) · **[I]** on disk but git-ignored (`.gitignore` excludes `models/**`, `embeddings/**`, `data/raw/**`, `data/processed/**`, `*.pt`, `*.tar.gz`, `*.zip`, `.claude/settings.local.json`). Sizes are on-disk sizes.

```text
Capstone/
├── .claude/
│   └── settings.local.json   [I] 2.6 KB
├── .vscode/
│   └── settings.json   [T] 135 B
├── configs/
│   └── experiment.yaml   [T] 2.4 KB
├── data/
│   ├── processed/
│   │   └── .gitkeep   [T] 0 B
│   └── raw/   [excluded from tree as requested — contents summarised in §2]
├── diagrams/
│   ├── fig4_1_architecture.drawio   [T] 9.8 KB
│   ├── fig4_1_architecture_preview.png   [T] 57.3 KB
│   ├── fig4_2_activity_diagram.drawio   [T] 8.7 KB
│   ├── fig4_2_activity_diagram_preview.png   [T] 70.0 KB
│   ├── fig4_3_drift_computation.drawio   [T] 4.8 KB
│   ├── fig4_3_drift_computation_preview.png   [T] 39.9 KB
│   ├── fig4_4_domain_adaptation_evaluation.drawio   [T] 5.5 KB
│   └── fig4_4_domain_adaptation_evaluation_preview.png   [T] 35.4 KB
├── embeddings/
│   ├── clip/
│   │   └── .gitkeep   [I] 0 B
│   ├── finetuned/
│   │   └── .gitkeep   [I] 0 B
│   ├── phase1_lightning/
│   │   ├── dota_feats.npy   [I] 893.1 KB
│   │   ├── dota_labels.npy   [I] 45.5 KB
│   │   ├── finetuned_visdrone_features.npy   [I] 424.1 MB
│   │   ├── finetuned_visdrone_labels.npy   [I] 10.8 MB
│   │   ├── remoteclip_dota_features.npy   [I] 84.3 MB
│   │   ├── remoteclip_dota_labels.npy   [I] 2.0 MB
│   │   ├── vis_feats.npy   [I] 312.1 KB
│   │   └── vis_labels.npy   [I] 18.4 KB
│   ├── phase1_local/
│   │   ├── dota_feats.npy   [I] 1.2 MB
│   │   ├── dota_labels.npy   [I] 30.1 KB
│   │   ├── dota_whole_image_feats.npy   [I] 600.1 KB
│   │   ├── dota_whole_image_predicted_labels.npy   [I] 16.5 KB
│   │   ├── vis_feats.npy   [I] 658.1 KB
│   │   ├── vis_labels.npy   [I] 19.4 KB
│   │   ├── visdrone_whole_image_feats.npy   [I] 600.1 KB
│   │   └── visdrone_whole_image_predicted_labels.npy   [I] 16.5 KB
│   ├── remoteclip/
│   │   └── .gitkeep   [I] 0 B
│   ├── remoteclip_vs_clip/
│   │   ├── dota_clip_feats.npy   [I] 1.2 MB
│   │   ├── dota_clip_labels.npy   [I] 30.6 KB
│   │   ├── dota_finetuned_clip_feats.npy   [I] 1.2 MB
│   │   ├── dota_finetuned_clip_heldout_feats.npy   [I] 1.2 MB
│   │   ├── dota_finetuned_clip_heldout_labels.npy   [I] 30.6 KB
│   │   ├── dota_finetuned_clip_labels.npy   [I] 30.6 KB
│   │   ├── dota_remoteclip_feats.npy   [I] 1.8 MB
│   │   ├── dota_remoteclip_labels.npy   [I] 30.6 KB
│   │   ├── visdrone_clip_feats.npy   [I] 1.2 MB
│   │   ├── visdrone_clip_labels.npy   [I] 30.6 KB
│   │   ├── visdrone_finetuned_clip_feats.npy   [I] 1.2 MB
│   │   ├── visdrone_finetuned_clip_heldout_feats.npy   [I] 1.2 MB
│   │   ├── visdrone_finetuned_clip_heldout_labels.npy   [I] 30.6 KB
│   │   ├── visdrone_finetuned_clip_labels.npy   [I] 30.6 KB
│   │   ├── visdrone_remoteclip_feats.npy   [I] 1.8 MB
│   │   └── visdrone_remoteclip_labels.npy   [I] 30.6 KB
│   └── xview_vs_visdrone/
│       ├── uavid_building_clip_feats.npy   [I] 600.1 KB
│       ├── uavid_building_remoteclip_feats.npy   [I] 900.1 KB
│       ├── visdrone_clip_feats.npy   [I] 2.3 MB
│       ├── visdrone_clip_labels.npy   [I] 23.6 KB
│       ├── visdrone_remoteclip_feats.npy   [I] 3.5 MB
│       ├── visdrone_remoteclip_labels.npy   [I] 23.6 KB
│       ├── xview_building_clip_feats.npy   [I] 600.1 KB
│       ├── xview_building_remoteclip_feats.npy   [I] 900.1 KB
│       ├── xview_clip_feats.npy   [I] 1.8 MB
│       ├── xview_clip_labels.npy   [I] 18.2 KB
│       ├── xview_remoteclip_feats.npy   [I] 2.7 MB
│       └── xview_remoteclip_labels.npy   [I] 18.2 KB
├── models/
│   ├── finetuned_clip_visdrone/
│   │   ├── finetuned_clip_visdrone_epoch1.pt   [I] 577.2 MB
│   │   ├── finetuned_clip_visdrone_epoch2.pt   [I] 577.2 MB
│   │   └── finetuned_clip_visdrone_epoch3.pt   [I] 577.2 MB
│   ├── finetuned_clip_visdrone_heldout/
│   │   ├── finetuned_clip_visdrone_heldout_epoch1.pt   [I] 577.2 MB
│   │   ├── finetuned_clip_visdrone_heldout_epoch2.pt   [I] 577.2 MB
│   │   └── finetuned_clip_visdrone_heldout_epoch3.pt   [I] 577.2 MB
│   └── RemoteCLIP/
│       ├── .gitkeep   [I] 0 B
│       └── RemoteCLIP-ViT-L-14.pt   [I] 1.6 GB
├── notebooks/
│   ├── 01_phase1_reproduction.ipynb   [T] 598.6 KB
│   ├── 02_full_dataset.ipynb   [T] 240 B
│   ├── 03_remoteclip.ipynb   [T] 58.1 KB
│   ├── 04_drift_analysis.ipynb   [T] 240 B
│   └── 04_finetune_visdrone_clip.ipynb   [T] 427.7 KB
├── results/
│   ├── category_yardstick/
│   │   ├── drift_ratio_comparison.png   [T] 101.5 KB
│   │   ├── normalized_drift_reduction.png   [T] 116.7 KB
│   │   ├── yardstick_dota_vs_visdrone.png   [T] 80.7 KB
│   │   ├── yardstick_table.json   [T] 2.7 KB
│   │   ├── yardstick_xview_visdrone_uavid.png   [T] 82.5 KB
│   │   └── yardstick_xview_vs_visdrone.png   [T] 73.8 KB
│   ├── figures/
│   │   └── .gitkeep   [T] 0 B
│   ├── full_dataset/
│   │   └── .gitkeep   [T] 0 B
│   ├── pca_tsne/
│   │   ├── pca_dota_vs_visdrone.png   [T] 338.2 KB
│   │   ├── pca_xview_visdrone_uavid.png   [T] 423.1 KB
│   │   ├── pca_xview_vs_visdrone.png   [T] 430.9 KB
│   │   ├── tsne_dota_vs_visdrone.png   [T] 350.2 KB
│   │   ├── tsne_xview_visdrone_uavid.png   [T] 537.8 KB
│   │   └── tsne_xview_vs_visdrone.png   [T] 518.0 KB
│   ├── phase1/
│   │   └── .gitkeep   [T] 0 B
│   ├── phase1_lightning/
│   │   ├── Capstone_v1.ipynb   [T] 2.3 MB
│   │   ├── drift_scores.png   [T] 40.6 KB
│   │   └── tsne_drift.png   [T] 178.3 KB
│   ├── phase1_local/
│   │   ├── distance_heatmap_local.png   [T] 72.1 KB
│   │   ├── drift_scores_local.json   [T] 229 B
│   │   ├── drift_scores_local.png   [T] 49.6 KB
│   │   ├── pca_combined.png   [T] 201.3 KB
│   │   ├── pca_dota.png   [T] 123.4 KB
│   │   └── tsne_drift_local.png   [T] 193.9 KB
│   ├── remoteclip_vs_clip/
│   │   ├── clip_vs_remoteclip_drift.png   [T] 53.5 KB
│   │   ├── drift_comparison.json   [T] 1.1 KB
│   │   ├── drift_comparison_3way.json   [T] 1.6 KB
│   │   ├── drift_comparison_3way_heldout.json   [T] 1.6 KB
│   │   ├── three_way_drift_comparison.png   [T] 64.5 KB
│   │   └── three_way_drift_comparison_heldout.png   [T] 63.5 KB
│   └── xview_vs_visdrone/
│       ├── drift_comparison.json   [T] 1.4 KB
│       ├── drift_comparison_4class.json   [T] 2.0 KB
│       ├── four_class_drift_comparison.png   [T] 50.0 KB
│       └── xview_vs_visdrone_drift.png   [T] 58.2 KB
├── src/
│   ├── centroids.py   [T] 0 B
│   ├── dataset.py   [T] 11.5 KB
│   ├── embeddings.py   [T] 7.9 KB
│   ├── metrics.py   [T] 0 B
│   └── visualization.py   [T] 0 B
├── .gitignore   [T] 552 B
├── Capstone Research Proposal Presentation-2 - UPDATED.pptx   [U] 12.6 MB
├── capstone_phase1_export.tar.gz   [I] 475.4 MB
├── make_category_yardstick_plots.py   [T] 7.8 KB
├── make_drift_ratio_plot.py   [T] 2.8 KB
├── make_normalized_drift_plot.py   [T] 4.0 KB
├── make_pca_tsne_plots.py   [T] 9.2 KB
├── Midsem-ppt.pdf   [U] 3.8 MB
├── PRESENTATION_SCRIPT.md   [T] 13.4 KB
├── PROGRESS_SUMMARY.md   [T] 17.5 KB
├── PROJECT_CONTEXT.md   [T] 34.6 KB
├── README.md   [T] 9.6 KB
├── requirements.txt   [T] 0 B
├── run_remoteclip_vs_clip.py   [T] 7.1 KB
├── run_uavid_building.py   [T] 6.2 KB
├── run_xview_vs_visdrone.py   [T] 7.3 KB
├── test.py   [T] 220 B
└── verify_drift_seeds.py   [T] 4.8 KB
```

Counts: 70 files tracked by git; 2 untracked-not-ignored files (`Capstone Research Proposal Presentation-2 - UPDATED.pptx`, `Midsem-ppt.pdf`) — matches the `git status` snapshot at session start. (This report, `TECHNICAL_REPORT.md`, is created after the tree is generated and is therefore not listed; it is untracked.)

### 1.2 README files

Only one README exists in the repository (`README.md` at the root). A search for `README*` in every subfolder (excluding `.git` and `data/raw`) found no other. Full contents follow.

`README.md` (131 lines, verbatim)

````markdown
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
````

Other markdown documents (not READMEs; quoted where relevant in later sections rather than duplicated here):

| File | Lines | Role |
|---|---|---|
| `PROJECT_CONTEXT.md` | 967 | authoritative working log and standing rules; contains the only record of the missing historical `model.ipynb` (§7a) and of the superseded first-pass numbers (§7b) |
| `PROGRESS_SUMMARY.md` | 127 | plain-language mid-semester summary + a guide to which figures to use/skip (quoted in §7) |
| `PRESENTATION_SCRIPT.md` | 244 | spoken script for the mid-semester evaluation (numbers restated from the JSONs) |

### 1.3 Dependency / environment files

`requirements.txt` — **empty (0 bytes)**; it was committed empty in the initial commit `36209f7` and never filled in:

```text
(file is empty)
```

Searched for and **not found** anywhere in the repo: `environment.yml`, `environment.yaml`, `pyproject.toml`, `setup.py`, `setup.cfg`, `Pipfile`, `poetry.lock`, `conda.yml`, `Dockerfile`, `Makefile`; also no `*.sh`, `*.slurm` or `*.sbatch` files. So the only dependency information in the repo is prose: the `README.md` “Environment” section (quoted in §1.2) and `PROJECT_CONTEXT.md` §8. A full `pip list` of the working environment is in §1.4 so the environment can be reconstructed.

### 1.4 Python and key package versions

**Local environment** (conda env `hf`, interpreter `C:\anaconda\envs\hf\python.exe`) — **[EXTERNAL]** live query on 2026-09-19 via `importlib.metadata`:

| Package | Installed version |
|---|---|
| Python | 3.9.25 |
| torch | 2.5.1 |
| torchvision | 0.20.1 |
| transformers | 4.49.0 |
| open_clip_torch | 3.3.0 |
| OpenAI `clip` package | not installed |
| numpy | 1.26.4 |
| scikit-learn | 1.3.2 |
| scipy | 1.13.1 |
| Pillow | 11.1.0 |
| datasets (HF) | 4.4.1 |
| huggingface_hub | 0.36.0 |
| safetensors | 0.7.0 |
| tokenizers | 0.21.4 |
| matplotlib | 3.9.1 |
| pandas | 2.3.1 |
| tqdm | 4.67.1 |
| nbconvert | 7.17.1 |
| nbformat | 5.10.4 |
| ipykernel | 6.30.1 |
| segment-anything / ultralytics / groundingdino / pycocotools / opencv | segment_anything: not installed, ultralytics: not installed, pycocotools: not installed, opencv-python: not installed |
| CUDA (torch build) | 11.8 |
| CUDA available | True |

**Versions as documented in the repo** — **[REPO]**:

| Source | Statement |
|---|---|
| `README.md` “Environment” | `torch==2.5.1+cu118`, `transformers==4.49.0` (pinned — newer versions refuse to load this project's CLIP checkpoint under torch<2.6), `open_clip_torch`, `scikit-learn==1.3.2`, `nbconvert`/`nbclient`; “Use the `hf` conda environment (do not use `tf`…)” |
| `PROJECT_CONTEXT.md` §8 | hf env: Python 3.9.25, PyTorch 2.5.1, CUDA 11.8, CUDA available: True, RTX 3050 Laptop ~4 GB VRAM, Transformers pinned to 4.49.0 (CVE-2025-32434 guard explanation), OpenCLIP installed. Rejected env: `tf` (Python 3.9.23, PyTorch 1.10.2, CUDA False). |

**Historical Lightning AI Studio environment** — **[LOG]** from `results/phase1_lightning/Capstone_v1.ipynb` saved outputs (kernel metadata: Python **3.12.3**, kernelspec `Python 3 (ipykernel)`). The notebook's pip output shows *two different* site-packages trees (`/usr/local/lib/python3.12/dist-packages` in the first session and `/home/zeus/miniconda3/envs/cloudspace/...` in the second), so the exact versions the kernel had loaded at the time of each model call cannot be pinned from the outputs alone:

| Item | Where | Saved output line |
|---|---|---|
| torch / torchvision | cell 37 | `Requirement already satisfied: torch in /home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages (from clip==1.0) (2.8.0+cu128)` |
| torch / torchvision | cell 37 | `Requirement already satisfied: torchvision in /home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages (from clip==1.0) (0.23.0+cu128)` |
| torch / torchvision | cell 37 | `Requirement already satisfied: torch in /home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages (2.8.0+cu128)` |
| torch / torchvision | cell 37 | `Requirement already satisfied: torchvision in /home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages (0.23.0+cu128)` |
| torch / torchvision | cell 45 | `Requirement already satisfied: torch in /usr/local/lib/python3.12/dist-packages (2.6.0+cu124)` |
| torch / torchvision | cell 45 | `Requirement already satisfied: torchvision in /usr/local/lib/python3.12/dist-packages (0.21.0+cu124)` |
| transformers | cell 45 | `Requirement already satisfied: transformers in /usr/local/lib/python3.12/dist-packages (4.57.6)` |
| scikit-learn | cell 28 | `scikit-learn 1.3.2 requires numpy<2.0,>=1.17.3, but you have numpy 2.4.4 which is incompatible.` |
| scikit-learn | cell 45 | `Requirement already satisfied: scikit-learn in /usr/local/lib/python3.12/dist-packages (1.8.0)` |
| Pillow | cell 26 | `Requirement already satisfied: pillow>=8 in /home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages (from matplotlib) (12.1.1)` |
| Pillow | cell 37 | `Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages (from torchvision->clip==1.0) (12.1.1)` |
| Pillow | cell 37 | `Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages (from torchvision) (12.1.1)` |
| numpy after `pip install opencv-python-headless`; plus the versions it reports as conflicting (matplotlib, scipy, pandas, scikit-learn) — i.e. the `cloudspace` env | cell 28 | `pandas 2.1.4 requires numpy<2,>=1.26.0; python_version >= "3.12", but you have numpy 2.4.4 which is incompatible.` |
| numpy after `pip install opencv-python-headless`; plus the versions it reports as conflicting (matplotlib, scipy, pandas, scikit-learn) — i.e. the `cloudspace` env | cell 28 | `matplotlib 3.8.2 requires numpy<2,>=1.21, but you have numpy 2.4.4 which is incompatible.` |
| numpy after `pip install opencv-python-headless`; plus the versions it reports as conflicting (matplotlib, scipy, pandas, scikit-learn) — i.e. the `cloudspace` env | cell 28 | `scipy 1.11.4 requires numpy<1.28.0,>=1.21.6, but you have numpy 2.4.4 which is incompatible.` |
| numpy after `pip install opencv-python-headless`; plus the versions it reports as conflicting (matplotlib, scipy, pandas, scikit-learn) — i.e. the `cloudspace` env | cell 28 | `scikit-learn 1.3.2 requires numpy<2.0,>=1.17.3, but you have numpy 2.4.4 which is incompatible.` |
| numpy after `pip install opencv-python-headless`; plus the versions it reports as conflicting (matplotlib, scipy, pandas, scikit-learn) — i.e. the `cloudspace` env | cell 28 | `Successfully installed numpy-2.4.4 opencv-python-headless-4.13.0.92` |
| OpenAI CLIP package | cell 37 | `Resolved https://github.com/openai/CLIP.git to commit d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` |

Lightning GPU (cell 0 output): `GPU Name: NVIDIA L40S`, `GPU Count: 1`. (`!nvidia-smi` in cell 1 printed `Failed to initialize NVML: Unknown Error`.)

<details><summary>Full <code>pip list --format=freeze</code> of the local <code>hf</code> environment (137 packages) — [EXTERNAL] queried 2026-09-19</summary>

```text
accelerate==1.10.1
aiohappyeyeballs==2.6.1
aiohttp==3.13.2
aiosignal==1.4.0
anyio==4.12.0
asttokens==3.0.0
async-timeout==5.0.1
attrs==25.4.0
beautifulsoup4==4.15.0
bleach==6.2.0
brotlicffi==1.0.9.2
certifi==2025.10.5
cffi==2.0.0
charset-normalizer==3.4.4
colorama==0.4.6
comm==0.2.3
contourpy==1.3.0
cycler==0.12.1
datasets==4.4.1
debugpy==1.8.16
decorator==5.2.1
defusedxml==0.7.1
dill==0.4.0
evaluate==0.4.6
exceptiongroup==1.3.0
executing==2.2.0
fastjsonschema==2.21.2
filelock==3.17.0
fonttools==4.59.1
frozenlist==1.8.0
fsspec==2025.10.0
ftfy==6.3.1
gmpy2==2.2.1
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==0.36.0
idna==3.11
importlib_metadata==8.7.0
importlib_resources==6.5.2
ipykernel==6.30.1
ipython==8.18.1
jedi==0.19.2
Jinja2==3.1.6
joblib==1.5.1
jsonschema==4.25.1
jsonschema-specifications==2025.9.1
jupyter_client==8.6.3
jupyter_core==5.8.1
jupyterlab_pygments==0.3.0
kaggle==1.7.4.5
kiwisolver==1.4.7
lxml==6.1.2
MarkupSafe==3.0.2
matplotlib==3.9.1
matplotlib-inline==0.1.7
mistune==3.3.4
mkl_fft==1.3.11
mkl_random==1.2.8
mkl-service==2.4.0
mpmath==1.3.0
multidict==6.7.0
multiprocess==0.70.18
munkres==1.1.4
nbclient==0.10.2
nbconvert==7.17.1
nbformat==5.10.4
nest_asyncio==1.6.0
networkx==3.2.1
numpy==1.26.4
open_clip_torch==3.3.0
packaging==25.0
pandas==2.3.1
pandocfilters==1.5.1
parso==0.8.4
pickleshare==0.7.5
pillow==11.1.0
pip==25.3
platformdirs==4.3.8
prompt_toolkit==3.0.51
propcache==0.4.1
protobuf==6.33.6
psutil==7.0.0
pure_eval==0.2.3
pyarrow==21.0.0
pycparser==2.23
Pygments==2.19.2
PyMuPDF==1.26.5
pyparsing==3.2.3
PyQt5==5.15.4
PyQt5-sip==12.9.0
PySocks==1.7.1
python-dateutil==2.9.0.post0
python-pptx==1.0.2
python-slugify==8.0.4
pytz==2025.2
pywin32==311
PyYAML==6.0.2
pyzmq==27.0.2
referencing==0.36.2
regex==2025.11.3
requests==2.32.5
rpds-py==0.27.1
safetensors==0.7.0
scikit-learn==1.3.2
scipy==1.13.1
setuptools==80.9.0
sip==6.5.1
six==1.17.0
soupsieve==2.8.4
stack_data==0.6.3
sympy==1.13.1
text-unidecode==1.3
threadpoolctl==3.6.0
timm==1.0.28
tinycss2==1.4.0
tokenizers==0.21.4
toml==0.10.2
torch==2.5.1
torchaudio==2.5.1
torchvision==0.20.1
tornado==6.5.2
tqdm==4.67.1
traitlets==5.14.3
transformers==4.49.0
typing_extensions==4.15.0
tzdata==2025.2
unicodedata2==16.0.0
urllib3==2.5.0
wcwidth==0.2.13
webencodings==0.5.1
wheel==0.45.1
win_inet_pton==1.1.0
xlsxwriter==3.2.9
xxhash==3.6.0
yarl==1.22.0
zipp==3.23.0
```

</details>

### 1.5 GPU / hardware

**Where hardware is mentioned inside the repo** — **[REPO]/[LOG]** (grep for `RTX|L40S|VRAM|NVIDIA|Lightning`; notebooks searched in code and saved output):

```text
README.md:19: - Phase-1 semantic-drift experiment (Lightning AI Studio, CLIP ViT-B/32 zero-shot, 7-class vocabulary) reproduced locally end-to-end (`notebooks/01_phase1_reproduction.ipynb`) — confirms the base phen
README.md:77: Key packages: `torch==2.5.1+cu118`, `transformers==4.49.0` (pinned — newer versions refuse to load this project's CLIP checkpoint under torch<2.6), `open_clip_torch`, `scikit-learn==1.3.2`, `nbconvert
README.md:79: GPU: NVIDIA RTX 3050 Laptop, ~4GB VRAM — keep batch sizes small (1–4) and use `torch.no_grad()`.
PROJECT_CONTEXT.md:29: The first stage was executed in the previous Lightning AI Studio environment.
PROJECT_CONTEXT.md:176: The previous experiments were run in a Lightning AI Studio.
PROJECT_CONTEXT.md:391: The project is being migrated away from Lightning AI to a local Windows/VS Code setup.
PROJECT_CONTEXT.md:436: GPU: NVIDIA GeForce RTX 3050 Laptop GPU
PROJECT_CONTEXT.md:437: VRAM: approximately 4 GB
PROJECT_CONTEXT.md:457: NVIDIA GeForce RTX 3050 Laptop GPU
PROJECT_CONTEXT.md:458: ~4 GB VRAM
PROJECT_CONTEXT.md:662: This is particularly important because the local GPU has only ~4 GB VRAM.
PROJECT_CONTEXT.md:892: - [x] Local NVIDIA GPU available
PROJECT_CONTEXT.md:901: - [x] Test local RemoteCLIP on one image (loads cleanly, 0 missing/unexpected keys, ~3.4GB VRAM)
PROGRESS_SUMMARY.md:14: The project originally ran on a cloud service (Lightning AI Studio). We moved everything to a local Windows machine with its own GPU, set up the correct software environment, and got the datasets (DOT
configs/experiment.yaml:3: # run locally instead of on Lightning AI Studio.
configs/experiment.yaml:62: device: cuda               # RTX 3050 Laptop, ~4GB VRAM
notebooks/01_phase1_reproduction.ipynb cell 0: Reproduces, on the local Windows/RTX 3050 machine, the pipeline actually documented in
notebooks/01_phase1_reproduction.ipynb cell 0: `results/phase1_lightning/Capstone_v1.ipynb` (the Lightning AI Studio notebook).
notebooks/01_phase1_reproduction.ipynb cell 1: GPU: NVIDIA GeForce RTX 3050 Laptop GPU
results/phase1_lightning/Capstone_v1.ipynb cell 0: GPU Name: NVIDIA L40S
```

**Local machine** — **[EXTERNAL]** live query (`Win32_*` WMI classes, `nvidia-smi`) on 2026-09-19:

| Component | Value |
|---|---|
| Machine | ASUSTeK COMPUTER INC. ASUS TUF Gaming A15 FA506ICB |
| CPU | AMD Ryzen 7 4800H with Radeon Graphics — 8 cores / 16 logical processors |
| RAM | 15.4 GiB |
| GPU (used for all experiments) | NVIDIA GeForce RTX 3050 Laptop GPU — 4096 MiB VRAM, driver 576.02, compute capability 8.6, CUDA 11.8 build of torch (also has an integrated AMD Radeon GPU, not used) |
| OS | Microsoft Windows 11 Home Single Language 10.0.26200, 64-bit |

**Historical (Phase-1, Lightning AI Studio)**: NVIDIA L40S, 1 GPU (see above). Storage and compute cost lines from the mid-semester deck (`Midsem-ppt.pdf` p.12, **[EXTERNAL]**): “Lightning AI Studio (Phase-1, cloud credits) + local RTX 3050 Laptop GPU (already-owned hardware) + allotted GPU — TBD”; “~16 GB local disk for datasets, embeddings, checkpoints”.

**Throughput / batch-size facts recorded in logs** — **[LOG]/[REPO]:**

| Stage | Batch size | Recorded throughput / duration | Source |
|---|---|---|---|
| Phase-1 whole-image + crop encoding (HF CLIP, local) | 1 (`configs/experiment.yaml: batch_size: 1`) | ≈1.5–3 it/s at start of the 300-image loop | `notebooks/01_phase1_reproduction.ipynb` cell 10 (tqdm) |
| Encoding with CLIP ViT-B/32 (`encode_with_clip`) | 16 | ≈120–170 crops/s (600 crops in 7.5 s) | `notebooks/04_finetune_visdrone_clip.ipynb` cells 11–12 |
| Encoding with RemoteCLIP ViT-L/14 (`encode_with_remoteclip`) inside the notebook kernel | 16 | 1.7–2.4 crops/s (1,200 crops in 611.9 s); as a plain script `PROJECT_CONTEXT.md` §7b records ≈14–15 crops/s | `notebooks/04_finetune_visdrone_clip.ipynb` cell 13; `PROJECT_CONTEXT.md` §7b |
| Fine-tuning (held-out run) | 8 | ≈77–78 crops/s; 20,963 steps/epoch; 2,146 s / 2,154 s / 2,183 s per epoch | `notebooks/04_finetune_visdrone_clip.ipynb` cell 7 |
| RemoteCLIP VRAM | — | ~3.4 GB VRAM when loaded (0 missing / 0 unexpected keys) | `PROJECT_CONTEXT.md` §21 checklist |

---

## 2. Datasets

Datasets actually used: **DOTA** (satellite; local, partial), **VisDrone2019-DET-train** (UAV; local, full train split), **xView** (satellite; streamed from Hugging Face), **UAVid** (UAV; local, from Kaggle). Datasets *named but not used*: DIOR and UAVDT (mentioned only in the proposal PDF, methodology Phase 1, **[EXTERNAL]**); the HF dataset `lironui/UAVid-RIS` (its dataset card is in the local Hugging Face cache — evidence of an abandoned attempt to obtain UAVid, see §10). The loader/mapping code for all four lives in one file, quoted in full in §2.7.

### 2.1 DOTA (satellite domain)

**Source / download location as referenced in code** — **[LOG]** `results/phase1_lightning/Capstone_v1.ipynb` (Lightning, April 2026):

```text
cell 6 : wget https://captain-whu.github.io/DOTA/dataset/train.zip        -> 'HTTP request sent, awaiting response... 404 Not Found'
cell 15: kaggle datasets download -d huanghanchina/dota-v15                -> '403 Client Error: Forbidden'
cell 18: shutil.move("/workspace/part1.zip", "/workspace/datasets/dota/part1.zip")
         shutil.move("/workspace/labelTxt.zip", "/workspace/datasets/dota/labelTxt.zip")   # obtained by other means (not recorded)
cell 22: Images: 469 / Labels: 1411
```

Both scripted downloads failed; the working copy came from manually supplied `part1.zip` + `labelTxt.zip`. The origin of those zips is **not recorded anywhere in the repo**. Local copy (`PROJECT_CONTEXT.md` §4): `data/raw/DOTA/images/` (469 images, from `part1.zip`), `data/raw/DOTA/labelTxt/` (1,411 label files), plus `part2.zip` (4,757,024,280 B) and `part3.zip` (3,319,108,346 B) **still zipped** (file dates Jan 24–25 2018). Verified counts **[DERIVED]**: 469 images on disk, 1,411 label files on disk, 469 label files matched to images and used. Because DOTA-v1.0's train split has 1,411 images, the 469 extracted images are the `part1` third of it **[ANALYSIS: the 1,411 = train-split size is external knowledge; the repo only states the counts]**.

**Exact subset used** — the crop pipeline (`get_dota_crops`) scans **every** image in `data/raw/DOTA/images` (`sorted(os.listdir(img_dir))`, no “first N”), so the working set is all 469 images. The **Phase-1 whole-image experiment** instead used `image_files[:300]`: in the local reproduction the listing is `sorted(...)` (`notebooks/01_phase1_reproduction.ipynb` cell 10), in the original Lightning notebook it is unsorted `os.listdir(...)` (`results/phase1_lightning/Capstone_v1.ipynb` cell 50), so the two 300-image subsets are different sets of images (see §10).

`src/dataset.py:96-116`

```python
def get_dota_crops(img_dir, label_dir, class_map=DOTA_TO_COMMON, min_crop_size=MIN_CROP_SIZE):
    """Ground-truth object crops from DOTA, filtered/mapped to the shared vocabulary."""
    from PIL import Image

    crops = []
    for img_name in sorted(os.listdir(img_dir)):
        label_path = os.path.join(label_dir, img_name.replace(".png", ".txt"))
        if not os.path.exists(label_path):
            continue
        image = Image.open(os.path.join(img_dir, img_name)).convert("RGB")
        W, H = image.size
        for (x1, y1, x2, y2, cls) in parse_dota_label(label_path):
            mapped = class_map.get(cls)
            if mapped is None:
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
                continue
            crops.append((image.crop((x1, y1, x2, y2)), mapped))
    return crops
```

`src/dataset.py:77-93`

```python
def parse_dota_label(label_path):
    """Returns list of (x1, y1, x2, y2, classname) from a DOTA OBB label file."""
    boxes = []
    with open(label_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        try:
            coords = list(map(float, parts[:8]))
        except ValueError:
            continue
        cls = parts[8]
        xs, ys = coords[0::2], coords[1::2]
        boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)), cls))
    return boxes
```

Label-file format seen **[DERIVED]** (`data/raw/DOTA/labelTxt/P0000.txt`, first lines): `imagesource:GoogleEarth`, `gsd:0.146343590398`, then rows `x1 y1 x2 y2 x3 y3 x4 y4 <class> <difficulty>`. `parse_dota_label` keeps only rows with ≥9 tokens, converts the 4-corner oriented box to an **axis-aligned box** (min/max of the corner coordinates, truncated to `int`), and **ignores the trailing `difficult` flag**.

**Native taxonomy and raw object counts in the 469 local images** — **[DERIVED]** (recount with `parse_dota_label`; 15 native classes observed):

| DOTA native class | Objects in 469 local images | Crops in historical Lightning array `remoteclip_dota_labels.npy` (28,767 total; after the 16-px filter) | Mapped to (`DOTA_TO_COMMON`) |
|---|---:|---:|---|
| ship | 15,846 | 12,878 | — (not used) |
| small-vehicle | 7,673 | 6,537 | small-vehicle |
| large-vehicle | 3,164 | 3,033 | large-vehicle |
| harbor | 2,307 | 2,252 | — (not used) |
| plane | 1,543 | 1,538 | — (not used) |
| tennis-court | 1,133 | 1,133 | — (not used) |
| swimming-pool | 1,019 | 1,016 | — (not used) |
| basketball-court | 111 | 111 | — (not used) |
| baseball-diamond | 73 | 73 | — (not used) |
| storage-tank | 58 | 57 | — (not used) |
| soccer-ball-field | 52 | 52 | — (not used) |
| roundabout | 50 | 48 | — (not used) |
| ground-track-field | 19 | 19 | — (not used) |
| helicopter | 18 | 18 | — (not used) |
| bridge | 2 | 2 | — (not used) |

Only `small-vehicle` and `large-vehicle` are mapped; every other DOTA class is dropped in the drift experiments. After clipping to the image and the `MIN_CROP_SIZE = 16` px filter: **9,570** mapped crops (`small-vehicle` 6,537; `large-vehicle` 3,033) — identical to the 6,537 / 3,033 in the historical Lightning array and to the “9,570 DOTA” figure in `PROJECT_CONTEXT.md` §7b.

**Image / sensor metadata** **[DERIVED]** (from the 469 label-file headers and image headers): `imagesource` = `GoogleEarth` for all 469; GSD (`gsd:` header, metres/pixel) min 1.34e-06, p25 0.128, **median 0.133**, p75 0.250, max 4.282; image width min/median/max = 432 / 1238 / 7168 px, height = 387 / 1140 / 6533 px.

### 2.2 VisDrone (UAV domain)

**Source / download** — **[LOG]** `results/phase1_lightning/Capstone_v1.ipynb` cell 7: `wget -c https://github.com/VisDrone/VisDrone-Dataset/raw/master/VisDrone2019-DET-train.zip` returned **404 Not Found** in Lightning; cell 31 nevertheless unzips a `VisDrone2019-DET-train.zip` that had been supplied another way (cell 33 lists `['VisDrone2019-DET-train', 'VisDrone2019-DET-train.zip']`). Origin not recorded. Local: `data/raw/VisDrone/VisDrone2019-DET-train/{images,annotations}` — **6,471 images and 6,471 annotation files; only the DET-train split** (no val/test). Image size **[DERIVED]**: width min/median/max 480/1400/2000, height 360/1050/1500 px.

**Exact subset used** — every image is scanned (`sorted(os.listdir(ann_dir))`). What differs between experiments is *which pool of images* is scanned (`allowed_files`):

| Use | Image pool | Code |
|---|---|---|
| Fine-tuning training data | train pool = 5,177 images (80%) | `allowed_files=train_files` (notebook 04 cell 5) |
| DOTA↔VisDrone evaluation (**all three** encoders: CLIP, RemoteCLIP, fine-tuned) | held-out pool = 1,294 images (20%) | `allowed_files=holdout_files` (notebook 04 cell 9; `run_remoteclip_vs_clip.py`) |
| xView↔VisDrone evaluation (frozen encoders only) | **all 6,471 images** (no split — nothing is trained, so no leakage concern) | `get_visdrone_crops(..., class_map=VISDRONE_TO_EXPANDED)` with no `allowed_files` (`run_xview_vs_visdrone.py`) |
| Phase-1 whole-image | first 300 images by sorted filename (local) / by filesystem order (Lightning) | `notebooks/01_phase1_reproduction.ipynb` cell 15 |

Split code (`src/dataset.py`):

`src/dataset.py:252-266`

```python
def split_visdrone_images(ann_dir, holdout_fraction, seed):
    """Split VisDrone annotation filenames into a training pool and a held-out pool, at the
    image level (so no image contributes crops to both). Use this whenever a model is (or might
    be) fine-tuned on VisDrone and then also evaluated on it, to prevent train/eval leakage --
    fine-tuning should only ever draw from the training pool, evaluation only from the held-out
    pool. Returns (train_files, holdout_files), both sets of annotation filenames.
    """
    all_ann_files = sorted(os.listdir(ann_dir))
    rng = random.Random(seed)
    shuffled = all_ann_files[:]
    rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * holdout_fraction)
    holdout_files = set(shuffled[:n_holdout])
    train_files = set(shuffled[n_holdout:])
    return train_files, holdout_files
```

Split sizes **[LOG]** (`notebooks/04_finetune_visdrone_clip.ipynb` cell 3 output): `VisDrone images: 6471 total -> 5177 train-pool, 1294 held-out-pool`; `HOLDOUT_FRACTION = 0.2`, `SPLIT_SEED = 1234`; the notebook asserts `train_files.isdisjoint(holdout_files)`.

`src/dataset.py:119-158`

```python
def get_visdrone_crops(img_dir, ann_dir, class_map=VISDRONE_TO_COMMON, min_crop_size=MIN_CROP_SIZE, allowed_files=None):
    """Ground-truth object crops from VisDrone, filtered/mapped to the shared vocabulary.

    allowed_files: optional set of annotation filenames to restrict extraction to (e.g. a
    train-only or held-out-only image split, to prevent train/eval leakage when a model is
    fine-tuned on VisDrone and then evaluated on it too). Default None = no restriction.
    """
    from PIL import Image

    crops = []
    ann_files = sorted(os.listdir(ann_dir))
    if allowed_files is not None:
        ann_files = [f for f in ann_files if f in allowed_files]
    for ann_file in ann_files:
        img_name = ann_file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        with open(os.path.join(ann_dir, ann_file), "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                cls_id = int(parts[5])
            except ValueError:
                continue
            if cls_id == 0 or cls_id not in VISDRONE_CLASSES or w < min_crop_size or h < min_crop_size:
                continue
            mapped = class_map.get(VISDRONE_CLASSES[cls_id])
            if mapped is None:
                continue
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            crops.append((image.crop((x1, y1, x2, y2)), mapped))
    return crops
```

Annotation format **[DERIVED]** (`0000002_00005_d_0000014.txt`, first line `684,8,273,116,0,0,0,0`): `x,y,w,h,score,category,truncation,occlusion`. The loader reads only `x,y,w,h` and `category`; the **`score`, truncation and occlusion fields are ignored**. Category `0` (ignored regions) and `11` (“others”) are skipped.

**Native taxonomy, annotation counts and mappings** — **[DERIVED]** (all 353,550 annotation rows re-parsed):

| id | VisDrone class | annotation rows | → `VISDRONE_TO_COMMON` (DOTA comparison) | → `VISDRONE_TO_EXPANDED` (xView comparison) | → `VISDRONE_TRAIN_MAP` (fine-tuning; defined in notebook 04 cell 5) |
|---:|---|---:|---|---|---|
| 0 | ignored region | 8,813 | — | — | — |
| 1 | pedestrian | 79,337 | — | — | person |
| 2 | person | 27,059 | — | — | person |
| 3 | bicycle | 10,480 | — | — | — |
| 4 | car | 144,867 | small-vehicle | car | small-vehicle |
| 5 | van | 24,956 | large-vehicle | van | large-vehicle |
| 6 | truck | 12,875 | large-vehicle | truck | large-vehicle |
| 7 | tricycle | 4,812 | — | — | motorcycle |
| 8 | awning-tricycle | 3,246 | — | — | motorcycle |
| 9 | bus | 5,926 | large-vehicle | bus | large-vehicle |
| 10 | motor | 29,647 | — | — | motorcycle |
| 11 | others | 1,532 | — | — | — |

The `VISDRONE_TRAIN_MAP` dictionary (notebook 04, cell 5, verbatim):

```python
from collections import Counter

VISDRONE_TRAIN_MAP = dict(VISDRONE_TO_COMMON)
VISDRONE_TRAIN_MAP.update({
    "motor": "motorcycle", "tricycle": "motorcycle", "awning-tricycle": "motorcycle",
    "pedestrian": "person", "person": "person",
})
```

**Crop counts per class per use** — **[DERIVED]** (re-implementation of the loader's filter on the annotation text; the two starred rows reproduce numbers printed in the repo):

| Pool / mapping | Total crops | Per class |
|---|---|---|
| all 6,471 images, **pre-2026-08-19 mapping** (bicycle→small-vehicle) — reproduces `PROJECT_CONTEXT.md` §7b “153,406” ★ | 153,406 | small-vehicle: 116,342, large-vehicle: 37,064 |
| all 6,471 images, current `VISDRONE_TO_COMMON` | 146,631 | small-vehicle: 109,567, large-vehicle: 37,064 |
| held-out pool (1,294 images), current mapping — the pool the 300/class DOTA-comparison samples are drawn from | 29,709 | small-vehicle: 22,368, large-vehicle: 7,341 |
| train pool (5,177 images), `VISDRONE_TRAIN_MAP` — reproduces notebook 04 cell 5 output 167,699 / 87,199 / 29,723 / 19,003 / 31,774 ★ | 167,699 | small-vehicle: 87,199, large-vehicle: 29,723, motorcycle: 19,003, person: 31,774 |
| all 6,471 images, `VISDRONE_TO_EXPANDED` (xView comparison) | 146,631 | car: 109,567, van: 20,534, truck: 11,297, bus: 5,233 |

Historical Lightning fine-tuning set — **[DERIVED]** from `embeddings/phase1_lightning/finetuned_visdrone_labels.npy` (217,153 labels): `small-vehicle` 116,342, `person` 39,848, `large-vehicle` 37,064, `motorcycle` 23,899. Its `small-vehicle` (116,342) and `large-vehicle` (37,064) counts equal the pre-fix all-image counts above exactly, i.e. the historical run used the same 16-px filter and the bicycle-in-small-vehicle mapping.

### 2.3 xView (satellite domain; streamed, nothing stored locally)

**Source as referenced in code**: `load_dataset("HichTala/xview", split=split, streaming=True)` in `get_xview_crops` (default `split="train"`). The official xView source needs DIUx challenge registration and a ~20 GB download (`PROJECT_CONTEXT.md` §4/§7d), so a Hugging Face mirror is streamed instead. Dataset card **[EXTERNAL]** (`~/.cache/huggingface/hub/datasets--HichTala--xview/snapshots/00a1b977dc16f6eda77cce0236ca2c2a13ac840b/README.md`, `refs/main` = that hash): splits **train 20,881 · validation 2,762 · test 2,326** examples; features `image`, `width`, `height`, `objects{bbox_id, category, bbox[4], area}`; **60 native classes** (list below).

```text
0:Fixed-wing Aircraft, 1:Small Aircraft, 2:Cargo Plane, 3:Helicopter, 4:Passenger Vehicle, 5:Small Car, 6:Bus, 7:Pickup Truck, 8:Utility Truck, 9:Truck, 10:Cargo Truck, 11:Truck w/Box, 12:Truck Tractor, 13:Trailer, 14:Truck w/Flatbed, 15:Truck w/Liquid, 16:Crane Truck, 17:Railway Vehicle, 18:Passenger Car, 19:Cargo Car, 20:Flat Car, 21:Tank car, 22:Locomotive, 23:Maritime Vessel, 24:Motorboat, 25:Sailboat, 26:Tugboat, 27:Barge, 28:Fishing Vessel, 29:Ferry, 30:Yacht, 31:Container Ship, 32:Oil Tanker, 33:Engineering Vehicle, 34:Tower crane, 35:Container Crane, 36:Reach Stacker, 37:Straddle Carrier, 38:Mobile Crane, 39:Dump Truck, 40:Haul Truck, 41:Scraper/Tractor, 42:Front loader/Bulldozer, 43:Excavator, 44:Cement Mixer, 45:Ground Grader, 46:Hut/Tent, 47:Shed, 48:Building, 49:Aircraft Hangar, 50:Damaged Building, 51:Facility, 52:Construction Site, 53:Vehicle Lot, 54:Helipad, 55:Storage Tank, 56:Shipping container lot, 57:Shipping Container, 58:Pylon, 59:Tower
```

(The class list above is transcribed from the cached dataset card; the code reads the same names at run time via `ds.features["objects"]["category"].feature.names`.)

`src/dataset.py:208-249`

```python
def get_xview_crops(class_map=None, split="train", min_crop_size=MIN_CROP_SIZE, max_images=None, log=print):
    """Ground-truth object crops from xView, streamed from the HichTala/xview HF mirror --
    no local download needed. Images are already-decoded 512x512 chips with parsed bbox/category
    lists, so unlike get_dota_crops/get_visdrone_crops there's no raw label-file parsing step.

    class_map defaults to XVIEW_TO_EXPANDED (car/bus/truck/van). NOTE: this expanded vocabulary
    is for xView<->VisDrone comparisons specifically -- DOTA only distinguishes small-vehicle/
    large-vehicle, not car/bus/truck/van, so it can't participate in this finer scheme.
    """
    from datasets import load_dataset

    if class_map is None:
        class_map = XVIEW_TO_EXPANDED

    ds = load_dataset("HichTala/xview", split=split, streaming=True)
    category_names = ds.features["objects"]["category"].feature.names

    crops = []
    n_images = 0
    for example in ds:
        if max_images is not None and n_images >= max_images:
            break
        n_images += 1
        image = example["image"]
        W, H = image.size
        objects = example["objects"]
        for cat_id, bbox in zip(objects["category"], objects["bbox"]):
            mapped = class_map.get(category_names[cat_id])
            if mapped is None:
                continue
            x, y, w, h = bbox
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
                continue
            crops.append((image.crop((x1, y1, x2, y2)), mapped))
        if log is not None and n_images % 200 == 0:
            log(f"  [xview] scanned {n_images} images, {len(crops)} mapped crops so far", flush=True)

    if log is not None:
        log(f"  [xview] done: scanned {n_images} images, {len(crops)} mapped crops", flush=True)
    return crops
```

**Exact subset used** — no random sampling at the image level: the stream is read **in stored order** and stopped after `max_images`:

| Script | `max_images` | Class map | Line |
|---|---|---|---|
| `run_xview_vs_visdrone.py` | `MAX_XVIEW_IMAGES = 5000` (of 20,881 `train` images = 23.9%) | default `XVIEW_TO_EXPANDED` (car/bus/truck/van/building) | `run_xview_vs_visdrone.py:29` |
| `run_uavid_building.py` | `MAX_XVIEW_IMAGES_FOR_BUILDING = 500` | `{"Building": "building"}` | `run_uavid_building.py:28` |

The `get_xview_crops` docstring states the streamed images are “already-decoded 512x512 chips with parsed bbox/category lists” **[REPO]** (author's observation; not independently verified here).

**Sample counts** — **[DERIVED]** from the cached label arrays `embeddings/xview_vs_visdrone/xview_{clip,remoteclip}_labels.npy` (925 rows): `car` 300, `truck` 300, `bus` 300, `van` **25**. Because `subsample_balanced` keeps every crop when a class has ≤300, the 25 `van` crops are *all* the van-mapped (`Pickup Truck`) crops in the 5,000-image scan. The raw (pre-subsample) per-class counts were printed by the script but **not saved** anywhere (no log file); the docs record only `van` = 25 (`PROJECT_CONTEXT.md` §7d). `van` therefore falls below `MIN_SAMPLES_FOR_DRIFT = 30` and is excluded from the drift JSON, although it was embedded and cached. Building: 300 xView crops embedded (`xview_building_*_feats.npy`, shape 300×512 / 300×768); raw building count from the 500-image scan not logged.

`XVIEW_TO_EXPANDED` mapping (verbatim from `src/dataset.py`):

`src/dataset.py:44-59`

```python
XVIEW_TO_EXPANDED = {
    "Small Car": "car",
    "Passenger Vehicle": "car",
    "Bus": "bus",
    "Pickup Truck": "van",
    "Truck": "truck",
    "Cargo Truck": "truck",
    "Truck w/Box": "truck",
    "Truck w/Flatbed": "truck",
    "Truck w/Liquid": "truck",
    "Truck Tractor": "truck",
    "Utility Truck": "truck",
    "Dump Truck": "truck",
    "Haul Truck": "truck",
    "Building": "building",
}
```

### 2.4 UAVid (UAV domain; used for the `building` class only)

**Source** — `run_uavid_building.py` docstring **[REPO]**: “UAVid crops come from local files (data/raw/UAVid, downloaded from Kaggle -- the official DLR/UTwente direct-download links were unreachable from this environment)”. `PROJECT_CONTEXT.md` §4 gives the Kaggle dataset as `giavuongnguyen/modified-uavid-dataset` (422 MB). Local layout **[DERIVED]**:

| Folder | Files on disk | Used? |
|---|---|---|
| `data/raw/UAVid/modified_uavid_dataset/train_data/{Images,Labels}` | 200 + 200 | yes (`run_uavid_building.py`) |
| `.../val_data/{Images,Labels}` | 70 + 70 | yes |
| `.../test_data/Images` | **150** (no labels) | no. (`PROJECT_CONTEXT.md` §4 says 30 — discrepancy, see §10) |

Image size **[DERIVED]**: **960×540** RGB. **[ANALYSIS]** Native UAVid frames are 4K according to the card quoted next, so this “modified” copy appears to be down-scaled ≈4× linearly; the repo does not document how it was modified. The HF dataset card for `lironui/UAVid-RIS` **[EXTERNAL]** describes native UAVid as “oblique-view aerial imagery with a camera angle of approximately 45 degrees … UAVs flying at an altitude of around 50 meters, with high-resolution frames extracted from 4K video recordings”.

Because UAVid provides **pixel masks, not boxes**, crops are derived by connected-component analysis:

`src/dataset.py:161-205`

```python
def get_uavid_crops(img_dir, label_dir, class_map=None, min_crop_size=MIN_CROP_SIZE, min_area=800, log=print):
    """Ground-truth object crops from UAVid, derived from pixel-level segmentation masks via
    connected-component analysis -- UAVid provides per-pixel RGB-coded masks, not bounding boxes
    like DOTA/VisDrone/xView, so each contiguous same-color blob above min_area becomes one crop.

    class_map defaults to UAVID_TO_EXPANDED ({(128,0,0): "building"}). img_dir and label_dir
    must contain identically-named PNG files (UAVid's own train_data/Images + train_data/Labels
    layout already satisfies this).
    """
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    if class_map is None:
        class_map = UAVID_TO_EXPANDED

    crops = []
    n_images = 0
    for label_name in sorted(os.listdir(label_dir)):
        img_path = os.path.join(img_dir, label_name)
        if not os.path.exists(img_path):
            continue
        n_images += 1
        image = Image.open(img_path).convert("RGB")
        label_arr = np.array(Image.open(os.path.join(label_dir, label_name)).convert("RGB"))
        for color, cls in class_map.items():
            mask = np.all(label_arr == np.array(color), axis=-1)
            if not mask.any():
                continue
            labeled, num_components = ndimage.label(mask)
            for i in range(1, num_components + 1):
                ys, xs = np.where(labeled == i)
                if len(ys) < min_area:
                    continue
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
                    continue
                crops.append((image.crop((x1, y1, x2, y2)), cls))
        if log is not None and n_images % 50 == 0:
            log(f"  [uavid] scanned {n_images} images, {len(crops)} mapped crops so far", flush=True)

    if log is not None:
        log(f"  [uavid] done: scanned {n_images} images, {len(crops)} mapped crops", flush=True)
    return crops
```

Mapping (verbatim): 

`src/dataset.py:68-74`

```python
# UAVid ground-truth mask RGB colors -> expanded vocabulary. UAVid provides per-pixel semantic
# masks, not bounding boxes, so crops are derived via connected-component analysis on the mask
# (see get_uavid_crops). Only "building" is mapped here -- UAVid's car classes are redundant
# with what VisDrone already gives us, and UAVid has no bus/truck/van class at all.
UAVID_TO_EXPANDED = {
    (128, 0, 0): "building",  # Building
}
```

**Counts** — **[DERIVED]** (replaying `get_uavid_crops` on the masks): `train_data` 200 images → 872 building crops; `val_data` 70 images → 327; **total 1,199 building crops**, of which 300 were sampled (`subsample_balanced`, seed 42). Crop size (px, on the 960×540 frame) — all 1,199: width min/median/max 22/165/959, height 16/110/539. The raw count was not printed to any saved log.

**Caveat found while verifying [DERIVED]:** the masks in this Kaggle copy are *not* pure-palette images — the train+val masks contain **76,995 distinct RGB values** (anti-aliased boundaries). `get_uavid_crops` matches the exact colour `(128, 0, 0)` (28.70% of all mask pixels), so boundary pixels with blended colours (e.g. `(136, 0, 0)`) are excluded from the building mask.

### 2.5 Shared category vocabularies (all mapping dicts, verbatim)

`src/dataset.py` lines 4–74 (quoted directly so they are not reconstructed from memory):

`src/dataset.py:4-74`

```python
DOTA_TO_COMMON = {
    "small-vehicle": "small-vehicle",
    "large-vehicle": "large-vehicle",
}

VISDRONE_CLASSES = {
    1: "pedestrian", 2: "person", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor",
}

# Only classes with real DOTA/VisDrone counterparts (PROJECT_CONTEXT.md Section 5 / 14a) --
# DOTA has no person/motorcycle equivalent, so those are intentionally excluded here.
# NOTE: bicycle is deliberately excluded from small-vehicle (2026-08-19 fix) -- DOTA's
# "small-vehicle" label is satellite-altitude, which can only resolve cars, not bicycles.
# Including bicycle mixed two visually distinct object types into one bucket (~5.8% of the
# bucket was bicycles), adding noise to the small-vehicle drift measurement.
VISDRONE_TO_COMMON = {
    "car": "small-vehicle",
    "van": "large-vehicle",
    "truck": "large-vehicle",
    "bus": "large-vehicle",
}

COMMON_VOCAB = ["small-vehicle", "large-vehicle"]

MIN_CROP_SIZE = 16

# --- Expanded vocabulary: xView <-> VisDrone (separate from COMMON_VOCAB above) ---
# xView's fine-grained vehicle taxonomy lets car/bus/truck/van map 1:1 onto VisDrone's own
# categories instead of being lumped into small-vehicle/large-vehicle. DOTA cannot use this
# scheme -- it only has the coarse small-vehicle/large-vehicle split, no finer subtypes -- so
# this is for xView<->VisDrone comparisons specifically, not a replacement for COMMON_VOCAB.
# xView has no motorcycle or person/pedestrian class (satellite altitude can't resolve them),
# so those VisDrone categories are excluded here too, same reasoning as COMMON_VOCAB above.
# "building" added 2026-08-20: xView has a real "Building" class, and UAVid (a UAV semantic
# segmentation dataset) has one too -- pairing them adds a genuinely new shared category beyond
# vehicles, and buildings are far more common per-image than any single vehicle type (unlike
# "van", which is bottlenecked by how rare xView's closest match, Pickup Truck, actually is).
EXPANDED_VOCAB = ["car", "bus", "truck", "van", "building"]

XVIEW_TO_EXPANDED = {
    "Small Car": "car",
    "Passenger Vehicle": "car",
    "Bus": "bus",
    "Pickup Truck": "van",
    "Truck": "truck",
    "Cargo Truck": "truck",
    "Truck w/Box": "truck",
    "Truck w/Flatbed": "truck",
    "Truck w/Liquid": "truck",
    "Truck Tractor": "truck",
    "Utility Truck": "truck",
    "Dump Truck": "truck",
    "Haul Truck": "truck",
    "Building": "building",
}

VISDRONE_TO_EXPANDED = {
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "van": "van",
}

# UAVid ground-truth mask RGB colors -> expanded vocabulary. UAVid provides per-pixel semantic
# masks, not bounding boxes, so crops are derived via connected-component analysis on the mask
# (see get_uavid_crops). Only "building" is mapped here -- UAVid's car classes are redundant
# with what VisDrone already gives us, and UAVid has no bus/truck/van class at all.
UAVID_TO_EXPANDED = {
    (128, 0, 0): "building",  # Building
}
```

Summary of which vocabulary is used where:

| Comparison | Satellite side | UAV side | Shared classes actually measured |
|---|---|---|---|
| DOTA ↔ VisDrone | DOTA (`DOTA_TO_COMMON`) | VisDrone held-out pool (`VISDRONE_TO_COMMON`) | `small-vehicle`, `large-vehicle` (`COMMON_VOCAB`) |
| xView ↔ VisDrone | xView (`XVIEW_TO_EXPANDED`) | VisDrone all images (`VISDRONE_TO_EXPANDED`) | `car`, `bus`, `truck` (`van` excluded: 25 xView samples < 30) |
| xView ↔ UAVid | xView `Building` | UAVid `(128,0,0)` masks | `building` (VisDrone has no building class) |

**Phase-1 (whole-image) vocabulary** — a different, 7-class scheme mapping DOTA's 15 classes; verbatim from `notebooks/01_phase1_reproduction.ipynb` cell 12 (identical to `results/phase1_lightning/Capstone_v1.ipynb` cell 56):

```python
# Shared vocabulary mapping (identical to Capstone_v1.ipynb cells 55-56)
dota_classes = [
    "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court",
    "basketball-court", "ground-track-field", "harbor", "bridge",
    "large-vehicle", "small-vehicle", "helicopter", "roundabout",
    "soccer-ball-field", "swimming-pool",
]

class_mapping = {
    "plane": "aircraft", "helicopter": "aircraft",
    "small-vehicle": "vehicle", "large-vehicle": "vehicle",
    "ship": "watercraft",
    "harbor": "infrastructure", "bridge": "infrastructure", "roundabout": "infrastructure",
    "storage-tank": "industrial",
    "baseball-diamond": "sports-field", "tennis-court": "sports-field",
    "basketball-court": "sports-field", "soccer-ball-field": "sports-field",
    "ground-track-field": "sports-field",
    "swimming-pool": "recreational",
}

common_classes = sorted(set(class_mapping.values()))
print(common_classes)
```

Zero-shot prompts in Phase-1 are the bare class strings: `dota_class_names = ["plane", "ship", "vehicle", "building", "harbor"]` (cell 9) and then `common_classes` (7 strings; cell 13); **no prompt template** is used there. Whole-image predicted-label counts (**[DERIVED]** from the cached arrays): DOTA 300 images → sports-field 122, infrastructure 61, aircraft 50, watercraft 37, vehicle 17, industrial 12, recreational 1; VisDrone 300 images → infrastructure 146, sports-field 100, vehicle 36, recreational 9, industrial 6, aircraft 3. (VisDrone whole images being assigned `aircraft`, `industrial`, `sports-field`, … shows these “classes” are zero-shot guesses, not ground truth — the repo itself says so, see §8.)

`configs/experiment.yaml` (the only config file; records the Phase-1 reproduction only) is quoted in full:

`configs/experiment.yaml` (75 lines, verbatim)

```yaml
# Phase-1 reproduction (notebooks/01_phase1_reproduction.ipynb)
# Mirrors the pipeline documented in results/phase1_lightning/Capstone_v1.ipynb,
# run locally instead of on Lightning AI Studio.

phase1_reproduction:
  dota:
    images_dir: data/raw/DOTA/images
    labels_dir: data/raw/DOTA/labelTxt
    n_images: 300          # matches historical cell (image_files[:300])
    classes:                # raw DOTA vocabulary used for the initial 5-class pass
      - plane
      - ship
      - vehicle
      - building
      - harbor

  visdrone:
    base_dir: data/raw/VisDrone/VisDrone2019-DET-train
    images_dir: data/raw/VisDrone/VisDrone2019-DET-train/images
    annotations_dir: data/raw/VisDrone/VisDrone2019-DET-train/annotations
    n_images: 300          # matches historical cell (image_files[:300])
    class_id_map:            # VisDrone numeric id -> class name
      1: pedestrian
      2: person
      3: bicycle
      4: car
      5: van
      6: truck
      7: tricycle
      8: awning-tricycle
      9: bus
      10: motor

  shared_vocabulary:        # class_mapping used to compare DOTA vs VisDrone in a common space
    plane: aircraft
    helicopter: aircraft
    small-vehicle: vehicle
    large-vehicle: vehicle
    ship: watercraft
    harbor: infrastructure
    bridge: infrastructure
    storage-tank: industrial
    roundabout: infrastructure
    baseball-diamond: sports-field
    tennis-court: sports-field
    basketball-court: sports-field
    soccer-ball-field: sports-field
    ground-track-field: sports-field
    swimming-pool: recreational

  model:
    name: openai/clip-vit-base-patch32
    source: huggingface-transformers   # CLIPModel + CLIPProcessor
    embedding_dim: 512
    normalize: true

  crop_test:                # small object-crop sanity check (openai/CLIP package, not HF)
    n_images_per_dataset: 5
    model_name: ViT-B/32
    source: openai-clip

  device: cuda               # RTX 3050 Laptop, ~4GB VRAM
  batch_size: 1               # images processed one at a time, matches historical loop
  random_seed: 42             # used for t-SNE (random_state=42)

  outputs:
    embeddings_dir: embeddings/phase1_local
    results_dir: results/phase1_local

  historical_reference:
    notebook: results/phase1_lightning/Capstone_v1.ipynb
    embeddings_dir: embeddings/phase1_lightning
    figures:
      - results/phase1_lightning/drift_scores.png
      - results/phase1_lightning/tsne_drift.png
```

### 2.6 Preprocessing, filtering and sampling — consolidated

| Step | Exact behaviour | Where |
|---|---|---|
| Box source | Ground-truth boxes only (DOTA: oriented quad → axis-aligned min/max; VisDrone: `x,y,w,h`; xView: `bbox` from HF; UAVid: connected-component bbox of a colour mask). **No detector is involved anywhere.** | `src/dataset.py` |
| Minimum size | `MIN_CROP_SIZE = 16` px. DOTA/xView/UAVid: applied to the *clipped* box. **VisDrone: applied to the raw `w`,`h` before clipping**, and not re-checked after clipping. | `src/dataset.py:29`, functions above |
| Image conversion | `Image.open(...).convert("RGB")`; crops are PIL images (no resizing at crop time). | `get_*_crops` |
| Class-balanced sampling | `subsample_balanced(crops, max_per_class, seed)`: for each class with more than `max_per_class` items, `rng.sample` without replacement (`random.Random(seed)`), then `rng.shuffle` of the combined list. **No per-image cap** — many crops can come from the same source image. `max_per_class = 300`, `seed = 42` for every reported run. | `src/dataset.py` (function quoted below) |
| Encoder-side preprocessing (CLIP, HF `CLIPProcessor`) | `crop_size 224`, `do_resize` with `size 224` (shortest edge) and bicubic (`resample: 3`), `do_center_crop`, `do_normalize` with OpenAI mean/std. **[ANALYSIS]** → non-square crops are resized so the short side is 224 and then centre-cropped to 224×224, cutting off the ends of elongated objects. | HF cache `preprocessor_config.json` (§3.2) |
| Encoder-side preprocessing (RemoteCLIP, `open_clip`) | `Resize(224, bicubic, antialias)` → `CenterCrop(224)` → RGB → tensor → `Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))` | printed `preprocess` object (§3.3) |
| Normalisation of embeddings | All cached embeddings are L2-normalised (**[DERIVED]** check of the first 2,000 rows of every array: norms in [0.9995, 1.0005]; the two float16 Lightning arrays are the only ones not exactly 1.0000). | `src/embeddings.py`; §5.1 inventory |
| Augmentation | None (neither at fine-tuning nor at evaluation). | `CropPromptDataset` |

`src/dataset.py:269-281`

```python
def subsample_balanced(crops, max_per_class, seed):
    """Random, class-balanced subsample -- representative across the whole scan, not a filename prefix."""
    rng = random.Random(seed)
    by_class = {}
    for crop, cls in crops:
        by_class.setdefault(cls, []).append((crop, cls))
    out = []
    for cls, items in by_class.items():
        if len(items) > max_per_class:
            items = rng.sample(items, max_per_class)
        out.extend(items)
    rng.shuffle(out)
    return out
```

**Crop-size statistics (px)** — **[DERIVED]**: crop extents from the annotation files under the loader's own clipping/filter logic. Rows marked *eval sample* replay `subsample_balanced` with the run's seed on the crop list (same list order and lengths ⇒ same random draws), so they describe the 300 crops per class that were embedded. **Reconstruction caveat:** this is exact only if the crop-list order matched the original runs; it was validated indirectly (the total crop counts 9,570 / 153,406 / 167,699 match the repo's printed numbers), not by comparing against the embedded crops themselves. Note: `DOTA eval sample` is unaffected by any split; `VisDrone eval sample (held-out)` is the DOTA-comparison sample; `VisDrone expanded eval sample` is the xView-comparison sample (all images).

| Set | Class | n | width min / median / max | height min / median / max |
|---|---|---:|---:|---:|
| DOTA, full scan (469 images) | small-vehicle | 6,537 | 16 / 34.0 / 92 | 16 / 34.0 / 80 |
| DOTA, full scan (469 images) | large-vehicle | 3,033 | 16 / 75.0 / 203 | 16 / 65.0 / 199 |
| DOTA, eval sample (seed 42, 300/class) | small-vehicle | 300 | 16 / 34.0 / 72 | 16 / 33.5 / 62 |
| DOTA, eval sample (seed 42, 300/class) | large-vehicle | 300 | 17 / 77.0 / 183 | 17 / 66.0 / 180 |
| VisDrone, all images, current mapping | small-vehicle | 109,567 | 16 / 50.0 / 717 | 16 / 38.0 / 790 |
| VisDrone, all images, current mapping | large-vehicle | 37,064 | 16 / 51.0 / 983 | 16 / 45.0 / 704 |
| VisDrone, eval sample (held-out pool, seed 42, 300/class) | small-vehicle | 300 | 16 / 50.5 / 374 | 16 / 39.5 / 275 |
| VisDrone, eval sample (held-out pool, seed 42, 300/class) | large-vehicle | 300 | 16 / 51.5 / 347 | 16 / 44.5 / 299 |
| VisDrone, expanded eval sample (all images, seed 42, 300/class) | car | 300 | 16 / 48.5 / 328 | 16 / 38.0 / 310 |
| VisDrone, expanded eval sample (all images, seed 42, 300/class) | bus | 300 | 16 / 54.0 / 597 | 17 / 54.0 / 663 |
| VisDrone, expanded eval sample (all images, seed 42, 300/class) | truck | 300 | 16 / 52.0 / 983 | 16 / 44.5 / 384 |
| VisDrone, expanded eval sample (all images, seed 42, 300/class) | van | 300 | 16 / 47.0 / 230 | 16 / 38.0 / 224 |
| UAVid building, eval sample (seed 42, 300) | building | 300 | 32 / 171.5 / 959 | 18 / 101.5 / 539 |

xView crop sizes could not be computed offline (the stream is not cached locally); no size statistic for xView exists in the repo.

### 2.7 `src/dataset.py` — complete file

Function/constant index (line numbers):

| Name | Lines | Kind |
|---|---|---|
| `DOTA_TO_COMMON` | 4–7 | constant |
| `VISDRONE_CLASSES` | 9–12 | constant |
| `VISDRONE_TO_COMMON` | 20–25 | constant |
| `COMMON_VOCAB` | 27–27 | constant |
| `MIN_CROP_SIZE` | 29–29 | constant |
| `EXPANDED_VOCAB` | 42–42 | constant |
| `XVIEW_TO_EXPANDED` | 44–59 | constant |
| `VISDRONE_TO_EXPANDED` | 61–66 | constant |
| `UAVID_TO_EXPANDED` | 72–74 | constant |
| `parse_dota_label` | 77–93 | function |
| `get_dota_crops` | 96–116 | function |
| `get_visdrone_crops` | 119–158 | function |
| `get_uavid_crops` | 161–205 | function |
| `get_xview_crops` | 208–249 | function |
| `split_visdrone_images` | 252–266 | function |
| `subsample_balanced` | 269–281 | function |

`src/dataset.py` (281 lines, verbatim)

```python
import os
import random

DOTA_TO_COMMON = {
    "small-vehicle": "small-vehicle",
    "large-vehicle": "large-vehicle",
}

VISDRONE_CLASSES = {
    1: "pedestrian", 2: "person", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor",
}

# Only classes with real DOTA/VisDrone counterparts (PROJECT_CONTEXT.md Section 5 / 14a) --
# DOTA has no person/motorcycle equivalent, so those are intentionally excluded here.
# NOTE: bicycle is deliberately excluded from small-vehicle (2026-08-19 fix) -- DOTA's
# "small-vehicle" label is satellite-altitude, which can only resolve cars, not bicycles.
# Including bicycle mixed two visually distinct object types into one bucket (~5.8% of the
# bucket was bicycles), adding noise to the small-vehicle drift measurement.
VISDRONE_TO_COMMON = {
    "car": "small-vehicle",
    "van": "large-vehicle",
    "truck": "large-vehicle",
    "bus": "large-vehicle",
}

COMMON_VOCAB = ["small-vehicle", "large-vehicle"]

MIN_CROP_SIZE = 16

# --- Expanded vocabulary: xView <-> VisDrone (separate from COMMON_VOCAB above) ---
# xView's fine-grained vehicle taxonomy lets car/bus/truck/van map 1:1 onto VisDrone's own
# categories instead of being lumped into small-vehicle/large-vehicle. DOTA cannot use this
# scheme -- it only has the coarse small-vehicle/large-vehicle split, no finer subtypes -- so
# this is for xView<->VisDrone comparisons specifically, not a replacement for COMMON_VOCAB.
# xView has no motorcycle or person/pedestrian class (satellite altitude can't resolve them),
# so those VisDrone categories are excluded here too, same reasoning as COMMON_VOCAB above.
# "building" added 2026-08-20: xView has a real "Building" class, and UAVid (a UAV semantic
# segmentation dataset) has one too -- pairing them adds a genuinely new shared category beyond
# vehicles, and buildings are far more common per-image than any single vehicle type (unlike
# "van", which is bottlenecked by how rare xView's closest match, Pickup Truck, actually is).
EXPANDED_VOCAB = ["car", "bus", "truck", "van", "building"]

XVIEW_TO_EXPANDED = {
    "Small Car": "car",
    "Passenger Vehicle": "car",
    "Bus": "bus",
    "Pickup Truck": "van",
    "Truck": "truck",
    "Cargo Truck": "truck",
    "Truck w/Box": "truck",
    "Truck w/Flatbed": "truck",
    "Truck w/Liquid": "truck",
    "Truck Tractor": "truck",
    "Utility Truck": "truck",
    "Dump Truck": "truck",
    "Haul Truck": "truck",
    "Building": "building",
}

VISDRONE_TO_EXPANDED = {
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "van": "van",
}

# UAVid ground-truth mask RGB colors -> expanded vocabulary. UAVid provides per-pixel semantic
# masks, not bounding boxes, so crops are derived via connected-component analysis on the mask
# (see get_uavid_crops). Only "building" is mapped here -- UAVid's car classes are redundant
# with what VisDrone already gives us, and UAVid has no bus/truck/van class at all.
UAVID_TO_EXPANDED = {
    (128, 0, 0): "building",  # Building
}


def parse_dota_label(label_path):
    """Returns list of (x1, y1, x2, y2, classname) from a DOTA OBB label file."""
    boxes = []
    with open(label_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        try:
            coords = list(map(float, parts[:8]))
        except ValueError:
            continue
        cls = parts[8]
        xs, ys = coords[0::2], coords[1::2]
        boxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)), cls))
    return boxes


def get_dota_crops(img_dir, label_dir, class_map=DOTA_TO_COMMON, min_crop_size=MIN_CROP_SIZE):
    """Ground-truth object crops from DOTA, filtered/mapped to the shared vocabulary."""
    from PIL import Image

    crops = []
    for img_name in sorted(os.listdir(img_dir)):
        label_path = os.path.join(label_dir, img_name.replace(".png", ".txt"))
        if not os.path.exists(label_path):
            continue
        image = Image.open(os.path.join(img_dir, img_name)).convert("RGB")
        W, H = image.size
        for (x1, y1, x2, y2, cls) in parse_dota_label(label_path):
            mapped = class_map.get(cls)
            if mapped is None:
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
                continue
            crops.append((image.crop((x1, y1, x2, y2)), mapped))
    return crops


def get_visdrone_crops(img_dir, ann_dir, class_map=VISDRONE_TO_COMMON, min_crop_size=MIN_CROP_SIZE, allowed_files=None):
    """Ground-truth object crops from VisDrone, filtered/mapped to the shared vocabulary.

    allowed_files: optional set of annotation filenames to restrict extraction to (e.g. a
    train-only or held-out-only image split, to prevent train/eval leakage when a model is
    fine-tuned on VisDrone and then evaluated on it too). Default None = no restriction.
    """
    from PIL import Image

    crops = []
    ann_files = sorted(os.listdir(ann_dir))
    if allowed_files is not None:
        ann_files = [f for f in ann_files if f in allowed_files]
    for ann_file in ann_files:
        img_name = ann_file.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue
        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        with open(os.path.join(ann_dir, ann_file), "r") as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                cls_id = int(parts[5])
            except ValueError:
                continue
            if cls_id == 0 or cls_id not in VISDRONE_CLASSES or w < min_crop_size or h < min_crop_size:
                continue
            mapped = class_map.get(VISDRONE_CLASSES[cls_id])
            if mapped is None:
                continue
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            crops.append((image.crop((x1, y1, x2, y2)), mapped))
    return crops


def get_uavid_crops(img_dir, label_dir, class_map=None, min_crop_size=MIN_CROP_SIZE, min_area=800, log=print):
    """Ground-truth object crops from UAVid, derived from pixel-level segmentation masks via
    connected-component analysis -- UAVid provides per-pixel RGB-coded masks, not bounding boxes
    like DOTA/VisDrone/xView, so each contiguous same-color blob above min_area becomes one crop.

    class_map defaults to UAVID_TO_EXPANDED ({(128,0,0): "building"}). img_dir and label_dir
    must contain identically-named PNG files (UAVid's own train_data/Images + train_data/Labels
    layout already satisfies this).
    """
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    if class_map is None:
        class_map = UAVID_TO_EXPANDED

    crops = []
    n_images = 0
    for label_name in sorted(os.listdir(label_dir)):
        img_path = os.path.join(img_dir, label_name)
        if not os.path.exists(img_path):
            continue
        n_images += 1
        image = Image.open(img_path).convert("RGB")
        label_arr = np.array(Image.open(os.path.join(label_dir, label_name)).convert("RGB"))
        for color, cls in class_map.items():
            mask = np.all(label_arr == np.array(color), axis=-1)
            if not mask.any():
                continue
            labeled, num_components = ndimage.label(mask)
            for i in range(1, num_components + 1):
                ys, xs = np.where(labeled == i)
                if len(ys) < min_area:
                    continue
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
                    continue
                crops.append((image.crop((x1, y1, x2, y2)), cls))
        if log is not None and n_images % 50 == 0:
            log(f"  [uavid] scanned {n_images} images, {len(crops)} mapped crops so far", flush=True)

    if log is not None:
        log(f"  [uavid] done: scanned {n_images} images, {len(crops)} mapped crops", flush=True)
    return crops


def get_xview_crops(class_map=None, split="train", min_crop_size=MIN_CROP_SIZE, max_images=None, log=print):
    """Ground-truth object crops from xView, streamed from the HichTala/xview HF mirror --
    no local download needed. Images are already-decoded 512x512 chips with parsed bbox/category
    lists, so unlike get_dota_crops/get_visdrone_crops there's no raw label-file parsing step.

    class_map defaults to XVIEW_TO_EXPANDED (car/bus/truck/van). NOTE: this expanded vocabulary
    is for xView<->VisDrone comparisons specifically -- DOTA only distinguishes small-vehicle/
    large-vehicle, not car/bus/truck/van, so it can't participate in this finer scheme.
    """
    from datasets import load_dataset

    if class_map is None:
        class_map = XVIEW_TO_EXPANDED

    ds = load_dataset("HichTala/xview", split=split, streaming=True)
    category_names = ds.features["objects"]["category"].feature.names

    crops = []
    n_images = 0
    for example in ds:
        if max_images is not None and n_images >= max_images:
            break
        n_images += 1
        image = example["image"]
        W, H = image.size
        objects = example["objects"]
        for cat_id, bbox in zip(objects["category"], objects["bbox"]):
            mapped = class_map.get(category_names[cat_id])
            if mapped is None:
                continue
            x, y, w, h = bbox
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
                continue
            crops.append((image.crop((x1, y1, x2, y2)), mapped))
        if log is not None and n_images % 200 == 0:
            log(f"  [xview] scanned {n_images} images, {len(crops)} mapped crops so far", flush=True)

    if log is not None:
        log(f"  [xview] done: scanned {n_images} images, {len(crops)} mapped crops", flush=True)
    return crops


def split_visdrone_images(ann_dir, holdout_fraction, seed):
    """Split VisDrone annotation filenames into a training pool and a held-out pool, at the
    image level (so no image contributes crops to both). Use this whenever a model is (or might
    be) fine-tuned on VisDrone and then also evaluated on it, to prevent train/eval leakage --
    fine-tuning should only ever draw from the training pool, evaluation only from the held-out
    pool. Returns (train_files, holdout_files), both sets of annotation filenames.
    """
    all_ann_files = sorted(os.listdir(ann_dir))
    rng = random.Random(seed)
    shuffled = all_ann_files[:]
    rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * holdout_fraction)
    holdout_files = set(shuffled[:n_holdout])
    train_files = set(shuffled[n_holdout:])
    return train_files, holdout_files


def subsample_balanced(crops, max_per_class, seed):
    """Random, class-balanced subsample -- representative across the whole scan, not a filename prefix."""
    rng = random.Random(seed)
    by_class = {}
    for crop, cls in crops:
        by_class.setdefault(cls, []).append((crop, cls))
    out = []
    for cls, items in by_class.items():
        if len(items) > max_per_class:
            items = rng.sample(items, max_per_class)
        out.extend(items)
    rng.shuffle(out)
    return out
```

---

## 3. Models

### 3.1 Encoders at a glance

| Arm (name used in results) | Architecture | Weights / load call | Embedding dim | Parameters | Used in |
|---|---|---|---|---|---|
| `clip_vit_b32` — CLIP ViT-B/32 | HF `CLIPModel` (vision ViT-B/32 + text transformer) | `CLIPModel.from_pretrained("openai/clip-vit-base-patch32")` | **512** (`projection_dim`) | 151,277,313 (both towers) | all comparisons (baseline) |
| `remoteclip_vit_l14` — RemoteCLIP ViT-L/14 | `open_clip` `ViT-L-14` | `open_clip.create_model_and_transforms("ViT-L-14", pretrained=None)`, then the local file `models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt` is `torch.load`-ed and passed to `model.load_state_dict(checkpoint, strict=False)`, raising if any key is missing/unexpected (exact code in §3.3) | **768** | 427,616,513 (both towers) | all comparisons |
| `finetuned_clip_visdrone_heldout` — CLIP fine-tuned on VisDrone | = CLIP ViT-B/32 with 14,570,496 parameters updated | base CLIP + `models/finetuned_clip_visdrone_heldout/finetuned_clip_visdrone_heldout_epoch3.pt` state dict (in-memory model after epoch 3 was what the notebook evaluated) | **512** | 151,277,313 total / 14,570,496 trainable | DOTA↔VisDrone 3-way only (never evaluated on xView/UAVid) |
| `finetuned_clip_visdrone` — first (leaky) fine-tuned run | same architecture | `models/finetuned_clip_visdrone/finetuned_clip_visdrone_epoch3.pt` | 512 | same | superseded; `drift_comparison_3way.json` |
| OpenAI `clip` ViT-B/32 (historical, Lightning Phase-1 crop sanity check only) | OpenAI `clip` package | `clip.load("ViT-B/32", device=device)` (`results/phase1_lightning/Capstone_v1.ipynb` cell 38) | 512 (arrays stored as float16) | — | `embeddings/phase1_lightning/{dota,vis}_feats.npy` (893×512, 312×512) |
| Historical Lightning models — **code not in repo** | RemoteCLIP ViT-L/14 (frozen); CLIP ViT-B/32 fine-tuned on VisDrone; attempted CLIP ViT-L/14 fine-tune | described only in `PROJECT_CONTEXT.md` §7a (quoted in §3.5) | 768 / 512 | — | `embeddings/phase1_lightning/{remoteclip_dota,finetuned_visdrone}_*.npy`; the two Lightning PNGs |

### 3.2 CLIP ViT-B/32 (baseline)

**Load call and encoding (verbatim from `src/embeddings.py`):**

`src/embeddings.py:20-25`

```python
def load_clip(device, model_name="openai/clip-vit-base-patch32"):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor
```

`src/embeddings.py:59-87`

```python
def encode_with_clip(crops, model, processor, device, batch_size=16, progress_every=10, log=print):
    """crops: list of (PIL.Image, label). Returns (features [N, D] float32, labels [N])."""
    feats, labels = [], []
    batch_imgs, batch_labels = [], []
    n_done, t0 = 0, time.time()

    def flush():
        nonlocal n_done
        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            f = model.get_image_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
        labels.extend(batch_labels)
        n_done += len(batch_imgs)
        if log is not None and (n_done // batch_size) % progress_every == 0:
            elapsed = time.time() - t0
            log(f"  [clip] {n_done}/{len(crops)} crops, {n_done/elapsed:.1f} crops/s, {elapsed:.0f}s elapsed", flush=True)

    for crop, label in crops:
        batch_imgs.append(crop)
        batch_labels.append(label)
        if len(batch_imgs) == batch_size:
            flush()
            batch_imgs, batch_labels = [], []
    if batch_imgs:
        flush()

    return np.vstack(feats), np.array(labels)
```

Checkpoint identity — **[EXTERNAL]** local Hugging Face cache `~/.cache/huggingface/hub/models--openai--clip-vit-base-patch32`: `refs/main` = **`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`** (snapshot files: `config.json`, `merges.txt`, `preprocessor_config.json`, `pytorch_model.bin` 605,247,071 B, `special_tokens_map.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`; downloaded Aug 13 18:42–18:51). A second snapshot `c237dc49a33fc61debc9276459120b7eac67e7ef` holds only `model.safetensors` (605,157,884 B, Aug 13 18:58). The repo pins `transformers==4.49.0` precisely because newer versions refuse `pytorch_model.bin` under torch<2.6 (`README.md`, `PROJECT_CONTEXT.md` §8), so `refs/main`'s `pytorch_model.bin` is the file the pinned environment loads. **[ANALYSIS, speculative]** The extra safetensors-only snapshot (written 9 minutes after the `.bin`) may be a side effect of loading under a newer `transformers`, which fetches a safetensors conversion; the repo does not record this.

Model config **[EXTERNAL/DERIVED]** (HF cache `config.json`): `projection_dim` = 512, `logit_scale_init_value` = 2.6592, `vision_image_size` = 224, `vision_patch_size` = 32, `vision_hidden_size` = 768, `vision_layers` = 12, `text_hidden_size` = 512, `text_layers` = 12, `text_max_position_embeddings` = 77. Total parameters (`sum(p.numel())`): **151,277,313**; `logit_scale.exp()` = 100.0000.

Image preprocessing config (HF cache `preprocessor_config.json`):

```json
{
  "crop_size": 224,
  "do_center_crop": true,
  "do_normalize": true,
  "do_resize": true,
  "feature_extractor_type": "CLIPFeatureExtractor",
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "resample": 3,
  "size": 224
}
```

Output: `model.get_image_features(**inputs)` (the projected image embedding, 512-d) divided by its L2 norm — no text tower involved in evaluation.

### 3.3 RemoteCLIP ViT-L/14

`src/embeddings.py:8-17`

```python
def load_remoteclip(checkpoint_path, device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=None)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"RemoteCLIP checkpoint did not load cleanly: missing={missing}, unexpected={unexpected}")
    model = model.to(device).eval()
    return model, preprocess
```

`src/embeddings.py:28-56`

```python
def encode_with_remoteclip(crops, model, preprocess, device, batch_size=16, progress_every=10, log=print):
    """crops: list of (PIL.Image, label). Returns (features [N, D] float32, labels [N])."""
    feats, labels = [], []
    batch_imgs, batch_labels = [], []
    n_done, t0 = 0, time.time()

    def flush():
        nonlocal n_done
        with torch.no_grad():
            t = torch.stack(batch_imgs).to(device)
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
        labels.extend(batch_labels)
        n_done += len(batch_imgs)
        if log is not None and (n_done // batch_size) % progress_every == 0:
            elapsed = time.time() - t0
            log(f"  [remoteclip] {n_done}/{len(crops)} crops, {n_done/elapsed:.1f} crops/s, {elapsed:.0f}s elapsed", flush=True)

    for crop, label in crops:
        batch_imgs.append(preprocess(crop))
        batch_labels.append(label)
        if len(batch_imgs) == batch_size:
            flush()
            batch_imgs, batch_labels = [], []
    if batch_imgs:
        flush()

    return np.vstack(feats), np.array(labels)
```

Checkpoint facts — **[DERIVED]/[EXTERNAL]** (file `models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt`): size **1,710,613,765 B**, file mtime 2026-04-28 22:31:09; `torch.load` returns a `dict` (plain state dict) with **446 tensors**; `visual.proj` shape [1024, 768] (⇒ 768-d embedding); `logit_scale.exp()` = 99.7623. `open_clip` version **3.3.0**; `open_clip.get_model_config("ViT-L-14")` = `{"embed_dim": 768, "vision_cfg": {"image_size": 224, "layers": 24, "width": 1024, "patch_size": 14}, "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 768, "heads": 12, "layers": 12}}`; total parameters 427,616,513.

`preprocess` object returned by `create_model_and_transforms` (printed):

```text
Compose(
    Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
    CenterCrop(size=(224, 224))
    MaybeConvertMode()
    MaybeToTensor()
    Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
)
```

Load behaviour **[LOG]**: in `notebooks/04_finetune_visdrone_clip.ipynb` cell 13 stderr, `WARNING:root:No pretrained weights loaded for model 'ViT-L-14'. Model initialized randomly.` is printed at construction (expected — `pretrained=None`), followed by a `torch.load ... weights_only=False` FutureWarning. The loader then raises if `load_state_dict(strict=False)` reports any missing/unexpected key; `PROJECT_CONTEXT.md` §21 records “loads cleanly, 0 missing/unexpected keys, ~3.4GB VRAM”. Output is `model.encode_image(...)` L2-normalised (768-d).

**Source of the checkpoint is not recorded in the repo** (no URL, no HF repo id, no download command anywhere — searched code, notebooks and all four `.md` files). **[EXTERNAL knowledge, unverified against this repo]** The publicly released RemoteCLIP weights are distributed as `RemoteCLIP-ViT-L-14.pt` in the Hugging Face repo `chendelong/RemoteCLIP`; the filename matches, but the repo never states that this is where the file came from. **[ANALYSIS]** The file's mtime (2026-04-28 22:31) predates the local project setup (2026-08-13); the circumstances of that earlier download/copy are not recorded (a `RemoteClip` directory also existed in the Lightning workspace, `Capstone_v1.ipynb` cell 17 output). Cite only after confirming the origin.

### 3.4 Fine-tuned CLIP (UAV-adapted)

#### 3.4.1 Design intent [REPO]

From the notebook's own header (`notebooks/04_finetune_visdrone_clip.ipynb` cell 0, “Notebook 04 (v2) - Third arm: CLIP fine-tuned on VisDrone, with a proper held-out split”): the arm exists “so the comparison would be fair in both directions — not just testing satellite specialization, but drone specialization too” (`PRESENTATION_SCRIPT.md` §5). It is trained only on VisDrone and evaluated on both DOTA and VisDrone with the same frozen-encoder drift protocol as the other arms.

#### 3.4.2 Complete training configuration

| Setting | Value | Source |
|---|---|---|
| Base model | `openai/clip-vit-base-patch32` (HF `CLIPModel`), fp32 (no autocast / AMP anywhere) | `src/embeddings.py:20`; notebook 04 cell 7 `base_model, processor = load_clip(device)` |
| Trainable parameters | **14,570,496 / 151,277,313** (9.63%) — printed as `trainable params: 14,570,496 / 151,277,313` | notebook 04 cell 7 [LOG]; `src/embeddings.py:145` |
| Unfrozen (default `unfreeze_patterns`) | `vision_model.encoder.layers.11`, `vision_model.encoder.layers.10`, `vision_model.post_layernorm`, `visual_projection` — **everything else frozen, including the entire text tower, `logit_scale`, the vision patch/position embeddings, `pre_layrnorm` and vision layers 0–9** | `src/embeddings.py:128-131`; notebook 04 does not override it |
| Loss | symmetric InfoNCE: mean of cross-entropy(image→text) and cross-entropy(text→image) on `logits_per_image`, targets = diagonal. Temperature = the model's own frozen `logit_scale.exp() = 100.0` | `src/embeddings.py:115` (quoted below) |
| Optimizer | `torch.optim.AdamW(filter(requires_grad), lr=1e-5, weight_decay=0.01)`; β/ε are PyTorch defaults (0.9, 0.999, 1e-8) — not set in code | `src/embeddings.py:153` |
| Learning rate | `LEARNING_RATE = 1e-5` | notebook 04 cell 1 |
| Schedule | `CosineAnnealingLR(optimizer, T_max=max(1, len(dataloader) * num_epochs))` stepped **every optimizer step**; no warm-up; `eta_min` = default 0 → T_max = 20,963 × 3 = **62,889 steps** | `src/embeddings.py:154` |
| Gradient clipping | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` | `src/embeddings.py:170` |
| Batch size | `TRAIN_BATCH_SIZE = 8` | notebook 04 cell 1 |
| Epochs | `NUM_EPOCHS = 3` (checkpoint saved after every epoch) | notebook 04 cell 1 |
| Steps | 20,963 per epoch (= ⌈167,699 / 8⌉), 62,889 total; 503,097 crop presentations | [LOG] `epoch 1/3 step 20/20963` |
| Prompt template | `PROMPT_TEMPLATE = "a drone photo of a {}"`, formatted with the class string (`small-vehicle`, `large-vehicle`, `motorcycle`, `person`) → e.g. `a drone photo of a small-vehicle`. Tokenised with `padding="max_length", truncation=True, max_length=77` | notebook 04 cell 1; `CropPromptDataset.__getitem__` |
| Image input | `CLIPProcessor` defaults (bicubic resize short side 224 → centre-crop 224 → normalise). No augmentation. | `CropPromptDataset` |
| Data loading | `torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)`; `num_workers` default 0; **no `torch.manual_seed` anywhere in the repo ⇒ shuffle order and CUDA nondeterminism are unseeded** | `src/embeddings.py:151`; repo-wide grep §5.3 |
| Train data | 167,699 VisDrone crops from the 5,177-image train pool, 4 classes, **no per-class cap** (`MAX_TRAIN_CROPS_PER_CLASS = None`) and no re-weighting (small-vehicle = 52.0% of crops) | notebook 04 cells 1, 5 |
| Device | `cuda` (RTX 3050 Laptop, 4 GB) | notebook 04 cell 1 |
| Model mode at the end | `model.eval()`; the in-memory epoch-3 model is what was evaluated (`encode_with_clip(..., finetuned_model, processor, ...)`) | `src/embeddings.py:201`; notebook 04 cell 11 |

#### 3.4.3 Code (verbatim)

`src/embeddings.py:90-112`

```python
class CropPromptDataset(torch.utils.data.Dataset):
    """Wraps (PIL.Image, label) crops as (image, text-prompt) pairs for CLIP contrastive fine-tuning."""

    def __init__(self, crops, processor, prompt_template="a photo of a {}"):
        self.crops = crops
        self.processor = processor
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop, label = self.crops[idx]
        prompt = self.prompt_template.format(label)
        inputs = self.processor(
            text=[prompt], images=crop, return_tensors="pt",
            padding="max_length", truncation=True, max_length=77,
        )
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }
```

`src/embeddings.py:115-122`

```python
def clip_contrastive_loss(logits_per_image):
    """Standard CLIP image/text contrastive loss -- diagonal entries are the correct pairs."""
    import torch.nn.functional as F

    labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_image.T, labels)
    return (loss_i + loss_t) / 2
```

`src/embeddings.py:125-202`

```python
def finetune_clip_on_crops(
    crops, model, processor, device,
    num_epochs=3, batch_size=8, lr=1e-5,
    unfreeze_patterns=(
        "vision_model.encoder.layers.11", "vision_model.encoder.layers.10",
        "vision_model.post_layernorm", "visual_projection",
    ),
    prompt_template="a photo of a {}",
    checkpoint_dir=None, checkpoint_prefix="finetuned_clip",
    progress_every=20, log=print,
):
    """Partial contrastive fine-tune of a CLIPModel on (crop, label) pairs.

    Only parameters whose name contains one of unfreeze_patterns are trained (default: last
    2 vision-encoder layers + post-layernorm + visual projection). Saves a checkpoint after
    every epoch if checkpoint_dir is given. Returns the model in eval() mode.
    """
    for name, param in model.named_parameters():
        param.requires_grad = any(p in name for p in unfreeze_patterns)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if log is not None:
        log(f"trainable params: {trainable:,} / {total:,}", flush=True)

    dataset = CropPromptDataset(crops, processor, prompt_template)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, len(dataloader) * num_epochs))

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            loss = clip_contrastive_loss(outputs.logits_per_image)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if log is not None and (step + 1) % progress_every == 0:
                elapsed = time.time() - t0
                done = (step + 1) * batch_size
                rate = done / elapsed
                remaining_steps = len(dataloader) - (step + 1)
                eta_this_epoch = remaining_steps * batch_size / rate if rate > 0 else float("nan")
                log(
                    f"  epoch {epoch+1}/{num_epochs} step {step+1}/{len(dataloader)} "
                    f"loss={loss.item():.4f} avg_loss={total_loss/(step+1):.4f} "
                    f"{rate:.1f} crops/s elapsed={elapsed:.0f}s eta_this_epoch={eta_this_epoch:.0f}s",
                    flush=True,
                )

        avg_loss = total_loss / max(1, len(dataloader))
        epoch_time = time.time() - t0
        if log is not None:
            log(f"epoch {epoch+1}/{num_epochs} done. avg_loss={avg_loss:.4f} time={epoch_time:.0f}s", flush=True)

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            if log is not None:
                log(f"  saved checkpoint: {ckpt_path}", flush=True)

    model.eval()
    return model
```

Notebook configuration cell and call (`notebooks/04_finetune_visdrone_clip.ipynb` cells 1 and 7, verbatim):

```python
import os
import sys
import json
import time
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from numpy.linalg import norm

sys.path.insert(0, os.path.abspath(".."))
from src.dataset import COMMON_VOCAB, VISDRONE_TO_COMMON, get_dota_crops, get_visdrone_crops, subsample_balanced
from src.embeddings import load_clip, load_remoteclip, encode_with_clip, encode_with_remoteclip, finetune_clip_on_crops

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

DOTA_IMG_DIR = "../data/raw/DOTA/images"
DOTA_LABEL_DIR = "../data/raw/DOTA/labelTxt"
VISDRONE_BASE = "../data/raw/VisDrone/VisDrone2019-DET-train"
VISDRONE_IMG_DIR = os.path.join(VISDRONE_BASE, "images")
VISDRONE_ANN_DIR = os.path.join(VISDRONE_BASE, "annotations")

# Evaluation scope
EVAL_MAX_CROPS_PER_CLASS = 300
EVAL_SEED = 42

# Train/held-out split (image-level, so no image contributes crops to both sets)
HOLDOUT_FRACTION = 0.2
SPLIT_SEED = 1234

# Training scope
MAX_TRAIN_CROPS_PER_CLASS = None  # None = use every mapped crop from the training-pool images
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 8
LEARNING_RATE = 1e-5
PROMPT_TEMPLATE = "a drone photo of a {}"

CHECKPOINT_DIR = "../models/finetuned_clip_visdrone_heldout"
EMBED_OUT = "../embeddings/remoteclip_vs_clip"
RESULTS_OUT = "../results/remoteclip_vs_clip"
os.makedirs(EMBED_OUT, exist_ok=True)
os.makedirs(RESULTS_OUT, exist_ok=True)
```

```python
base_model, processor = load_clip(device)

finetuned_model = finetune_clip_on_crops(
    visdrone_train_crops, base_model, processor, device,
    num_epochs=NUM_EPOCHS, batch_size=TRAIN_BATCH_SIZE, lr=LEARNING_RATE,
    prompt_template=PROMPT_TEMPLATE,
    checkpoint_dir=CHECKPOINT_DIR, checkpoint_prefix="finetuned_clip_visdrone_heldout",
    progress_every=20, log=print,
)
print("Fine-tuning complete.")
```

#### 3.4.4 Training data

VisDrone train pool only (5,177 images, seed-1234 split — §2.2). Class distribution **[LOG]** (`notebooks/04_finetune_visdrone_clip.ipynb` cell 5): `{'small-vehicle': 87199, 'large-vehicle': 29723, 'motorcycle': 19003, 'person': 31774}` = **167,699** crops (extraction took 103.8 s). Text side: one fixed prompt per class. **DOTA, xView, UAVid and the 1,294 held-out VisDrone images never enter training.** `bicycle` is not in `VISDRONE_TRAIN_MAP` and is therefore excluded from training too.

#### 3.4.5 What actually changed — tensor-level check of the saved checkpoints [DERIVED]

| Checkpoint | File bytes | Tensors in file | Tensors ≠ base CLIP | Parameters ≠ base | Changed tensor groups (count) | `logit_scale.exp()` |
|---|---|---|---|---|---|---|
| `heldout_epoch1` | 605,275,780 | 398 | 35 | 14,570,496 | vision_model.encoder.layers.10: 16; vision_model.encoder.layers.11: 16; vision_model.post_layernorm: 2; visual_projection.weight: 1 | 100.0000 |
| `heldout_epoch2` | 605,275,780 | 398 | 35 | 14,570,496 | vision_model.encoder.layers.10: 16; vision_model.encoder.layers.11: 16; vision_model.post_layernorm: 2; visual_projection.weight: 1 | 100.0000 |
| `heldout_epoch3` | 605,275,780 | 398 | 35 | 14,570,496 | vision_model.encoder.layers.10: 16; vision_model.encoder.layers.11: 16; vision_model.post_layernorm: 2; visual_projection.weight: 1 | 100.0000 |
| `leaky_epoch3` | 605,272,564 | 398 | 35 | 14,570,496 | vision_model.encoder.layers.10: 16; vision_model.encoder.layers.11: 16; vision_model.post_layernorm: 2; visual_projection.weight: 1 | 100.0000 |

Every checkpoint differs from the pretrained weights in **exactly** the 35 tensors of the four unfrozen modules (16 in each of vision layers 10 and 11, 2 in `post_layernorm`, 1 in `visual_projection.weight`), totalling 14,570,496 parameters — matching the trainable count printed during training. The text tower is bit-identical to the base model, and `logit_scale` (temperature) is unchanged.

#### 3.4.6 Training curves and logs [LOG]

The only surviving training log is the stdout stored in `notebooks/04_finetune_visdrone_clip.ipynb` cell 7 (held-out run): 3,144 logged lines (every 20 steps) plus three epoch summaries. No TensorBoard/CSV/`.log` exists. Epoch summaries:

| Epoch | avg_loss (mean of per-step loss) | Wall time (s) | Checkpoint mtime |
|---|---|---|---|
| 1 | 1.4800 | 2146 | 2026-08-20 20:12:28 |
| 2 | 1.4040 | 2154 | 2026-08-20 20:48:26 |
| 3 | 1.3525 | 2183 | 2026-08-20 21:24:54 |

Loss curve, condensed to 1,000-step bins. “mean” = mean of the logged instantaneous losses in the bin (≈50 logged points); “run.avg” = the notebook's `avg_loss` running average at the bin's last logged step (this is a running mean over the *epoch so far*, so it lags). Bin label = epoch step range. **[DERIVED]** from the 3,144 logged lines:

| Steps in epoch | E1 mean | E1 run.avg | E2 mean | E2 run.avg | E3 mean | E3 run.avg |
|---|---:|---:|---:|---:|---:|---:|
| 1–1,000 | 1.5283 | 1.6192 | 1.3457 | 1.4036 | 1.4006 | 1.3438 |
| 1,001–2,000 | 1.5398 | 1.5747 | 1.4451 | 1.4026 | 1.3288 | 1.3523 |
| 2,001–3,000 | 1.5681 | 1.5562 | 1.5111 | 1.4103 | 1.2871 | 1.3559 |
| 3,001–4,000 | 1.5059 | 1.5380 | 1.3254 | 1.4128 | 1.3299 | 1.3583 |
| 4,001–5,000 | 1.4840 | 1.5291 | 1.4194 | 1.4129 | 1.4232 | 1.3608 |
| 5,001–6,000 | 1.5403 | 1.5231 | 1.3679 | 1.4123 | 1.4380 | 1.3605 |
| 6,001–7,000 | 1.4560 | 1.5179 | 1.4508 | 1.4126 | 1.2895 | 1.3601 |
| 7,001–8,000 | 1.5072 | 1.5127 | 1.4418 | 1.4126 | 1.3928 | 1.3584 |
| 8,001–9,000 | 1.5059 | 1.5088 | 1.3994 | 1.4126 | 1.3822 | 1.3580 |
| 9,001–10,000 | 1.5210 | 1.5059 | 1.4239 | 1.4121 | 1.3646 | 1.3576 |
| 10,001–11,000 | 1.6061 | 1.5040 | 1.4238 | 1.4109 | 1.4699 | 1.3574 |
| 11,001–12,000 | 1.4747 | 1.5005 | 1.4123 | 1.4113 | 1.3142 | 1.3554 |
| 12,001–13,000 | 1.4666 | 1.4973 | 1.4277 | 1.4106 | 1.4677 | 1.3565 |
| 13,001–14,000 | 1.4771 | 1.4954 | 1.3923 | 1.4104 | 1.3943 | 1.3552 |
| 14,001–15,000 | 1.4409 | 1.4922 | 1.4186 | 1.4092 | 1.2671 | 1.3547 |
| 15,001–16,000 | 1.4773 | 1.4897 | 1.3056 | 1.4092 | 1.3257 | 1.3544 |
| 16,001–17,000 | 1.4487 | 1.4875 | 1.4544 | 1.4082 | 1.3497 | 1.3533 |
| 17,001–18,000 | 1.4730 | 1.4854 | 1.4106 | 1.4072 | 1.3472 | 1.3525 |
| 18,001–19,000 | 1.4218 | 1.4833 | 1.3808 | 1.4068 | 1.3730 | 1.3518 |
| 19,001–20,000 | 1.4600 | 1.4816 | 1.4498 | 1.4060 | 1.3574 | 1.3525 |
| 20,001–20,960 | 1.4587 | 1.4801 | 1.3541 | 1.4041 | 1.2270 | 1.3526 |

Selected raw lines (first, a mid-epoch-2 line, last) **[LOG]**:

```text
trainable params: 14,570,496 / 151,277,313
  epoch 1/3 step 20/20963 loss=1.5860 avg_loss=2.2003 47.9 crops/s elapsed=3s eta_this_epoch=3494s
  epoch 1/3 step 480/20963 loss=1.3833 avg_loss=1.6673 76.2 crops/s elapsed=50s eta_this_epoch=2151s
  epoch 3/3 step 20960/20963 loss=1.4083 avg_loss=1.3526 76.8 crops/s elapsed=2183s eta_this_epoch=0s
epoch 1/3 done. avg_loss=1.4800 time=2146s
epoch 2/3 done. avg_loss=1.4040 time=2154s
epoch 3/3 done. avg_loss=1.3525 time=2183s
```

Per-epoch range of the instantaneous logged loss **[DERIVED]**: epoch 1 min 0.7847 / max 3.1052; epoch 2 min 0.7164 / max 2.8975; epoch 3 min 0.7434 / max 2.8634. There is **no validation loss, no held-out loss curve and no accuracy metric** for the fine-tuning; the only downstream check is the drift measurement itself.

**[ANALYSIS] How to read the loss values.** Each batch has 8 crops but only 4 distinct prompt strings, so identical prompts are guaranteed within a batch (pigeonhole); `clip_contrastive_loss` treats only the diagonal as positive, so same-class pairs act as negatives. For a model that separates classes perfectly, the best achievable per-batch loss is `mean_i ln(m_i)` where `m_i` is the number of batch items sharing item *i*'s class. With the train-set class frequencies above and B = 8 (binomial approximation), the expected floor is **1.0918**; the loss of an untrained uniform-logit model is `ln 8 = 2.0794`. The observed epoch averages (1.4800 → 1.4040 → 1.3525) therefore lie between the floor and chance; the loss can never approach 0 and its absolute value is not a measure of fit quality. (Calculation done for this report; it appears nowhere in the repo.)

#### 3.4.7 The two fine-tuning runs and what survives of each

|  | First run (leaky — superseded) | Second run (held-out — used for headline numbers) |
|---|---|---|
| Dates (checkpoint mtimes, local time IST) | 2026-08-19 15:11:13 / 2026-08-19 15:57:59 / 2026-08-19 16:44:23 | 2026-08-20 20:12:28 / 2026-08-20 20:48:26 / 2026-08-20 21:24:54 |
| Epoch wall time from mtime differences | ≈46 min 46 s, ≈46 min 24 s | ≈35 min 58 s, ≈36 min 28 s (matches logged 2,146 s / 2,154 s / 2,183 s) |
| Training crops | **every** VisDrone image, pre-fix mapping (`train_max_crops_per_class: null` in `drift_comparison_3way.json`). Count not logged; **[ANALYSIS]** ≈217,153 (= the historical Lightning count), consistent with ≈46.5-min epochs at ≈78 crops/s | 167,699 (5,177 train-pool images) |
| Evaluation pool | sample drawn from the same complete pool → **train/eval leakage** | 1,294 held-out images, never trained on; all three arms evaluated on it |
| Hyper-parameters | `train_num_epochs 3`, `train_batch_size 8`, `train_lr 1e-05`, `prompt_template "a drone photo of a {}"` (from the JSON) | same values (from the JSON) |
| Checkpoints on disk | `models/finetuned_clip_visdrone/finetuned_clip_visdrone_epoch{1,2,3}.pt`, 605,272,564 B each | `models/finetuned_clip_visdrone_heldout/finetuned_clip_visdrone_heldout_epoch{1,2,3}.pt`, 605,275,780 B each |
| Notebook / log | **v1 notebook was overwritten in place** (only v2 was ever committed, in `5791ee2`); its training log is **not preserved** | `notebooks/04_finetune_visdrone_clip.ipynb` (v2) with full stdout |
| Result files | `results/remoteclip_vs_clip/drift_comparison_3way.json`, `three_way_drift_comparison.png`, arrays `embeddings/remoteclip_vs_clip/{dota,visdrone}_finetuned_clip_{feats,labels}.npy` | `drift_comparison_3way_heldout.json`, `three_way_drift_comparison_heldout.png`, arrays `*_finetuned_clip_heldout_*.npy` |
| Reported drift (L2) | small-vehicle 0.1412, large-vehicle 0.0964 | small-vehicle 0.1585, large-vehicle 0.0955 |

Only the epoch-3 checkpoints were used for reported numbers (`verify_drift_seeds.py:32` also points at the *leaky* epoch-3 file). Epoch-1 and epoch-2 checkpoints have never been evaluated.

### 3.5 Historical Lightning models (code and checkpoints not in the repo)

The notebook that produced `remoteclip_dota_features.npy` and `finetuned_visdrone_features.npy` (`model.ipynb`) is **not in the repo directory, not in `capstone_phase1_export.tar.gz` (checked: the tarball lists only the 8 `.npy`, 2 `.png` and `Capstone_v1.ipynb`), and was not found on the Desktop (searched to depth 4 for `model.ipynb`)**. Its content survives only as the description below. Note one difference it records: the historical fine-tune used prompts like `"an aerial photo of a {class}"` and trained on all 217,153 VisDrone crops, whereas the local runs use `"a drone photo of a {}"` (§3.4.2). Quoted verbatim:

`PROJECT_CONTEXT.md:222-256` (verbatim)

````markdown
# 7. Existing Lightning Artifacts

The extracted export has been organized into the recommended structure (Section 11/17):

```text
embeddings/phase1_lightning/
├── remoteclip_dota_features.npy        (28,767 × 768)
├── remoteclip_dota_labels.npy
├── finetuned_visdrone_features.npy     (217,153 × 512)
├── finetuned_visdrone_labels.npy
├── dota_feats.npy                      (893 × 512, crop-level sanity check)
├── dota_labels.npy
├── vis_feats.npy                       (312 × 512, crop-level sanity check)
└── vis_labels.npy

results/phase1_lightning/
├── Capstone_v1.ipynb    (documents the small-scale CLIP ViT-B/32 pipeline only)
├── drift_scores.png
└── tsne_drift.png
```

There were also model checkpoints such as:

```text
visdrone_clip_epoch1.pt
visdrone_clip_epoch2.pt
visdrone_clip_epoch3.pt

visdrone_vit14_epoch1.pt
visdrone_vit14_epoch2.pt
```

The large `.pt` checkpoints were intentionally NOT included in the reduced 525 MB export. `visdrone_vit14_epoch3.pt` was never produced at all — the fine-tuning run that would have created it was cut off at epoch 3, 70% through (see Section 7a).

Do not assume these checkpoints are required for the next experiment. Retrieve them only if later analysis proves they are necessary.
````

`PROJECT_CONTEXT.md:258-268` (verbatim)

```markdown
## 7a. `model.ipynb` — the notebook that explains the large artifacts

`Capstone_v1.ipynb` only documents the small-scale CLIP ViT-B/32 pipeline. A second Lightning notebook, `model.ipynb`, was recovered separately and explains `remoteclip_dota_features.npy` and `finetuned_visdrone_features.npy`:

1. **RemoteCLIP ViT-L/14** (frozen, off-the-shelf, remote-sensing-pretrained) embeds every object crop from all 469 local DOTA images → `remoteclip_dota_features.npy`, labeled with the raw 15-class DOTA taxonomy.
2. A **separate, different model** — vanilla `openai/clip-vit-base-patch32` — is fine-tuned on VisDrone crops (contrastive loss, last-2 vision-encoder layers + projection unfrozen, prompts like `"an aerial photo of a {class}"`) over all 217,153 VisDrone crops, saving `visdrone_clip_epoch{1,2,3}.pt`.
3. The fine-tuned checkpoint is used to re-embed all 217,153 VisDrone crops → `finetuned_visdrone_features.npy`.
4. **The drift measurement step fits two independent PCA models — one on DOTA, one on VisDrone — each reducing separately to 256-dim**, then takes centroid distances across those two independently-fit spaces (large-vehicle: L2=0.20/cosine=0.36; small-vehicle: L2=0.125/cosine=0.27).
5. A later cell attempts a real fix — fine-tuning CLIP ViT-L/14 (matching RemoteCLIP's dimensionality, avoiding the need for PCA) — but is cut off at epoch 3, 70% through. No checkpoint or embeddings were produced from this attempt.

**This comparison is methodologically invalid as-is** (see Section 14) and its numbers should not be cited in the report until redone properly.
```

The historical checkpoints named there (`visdrone_clip_epoch{1,2,3}.pt`, `visdrone_vit14_epoch{1,2}.pt`) are **not present locally**; `visdrone_vit14_epoch3.pt` never existed. What *is* available from that run: `embeddings/phase1_lightning/remoteclip_dota_features.npy` (28,767×768 float32) with labels over all 15 DOTA classes, and `finetuned_visdrone_features.npy` (217,153×512 float32) with labels `small-vehicle` 116,342 / `person` 39,848 / `large-vehicle` 37,064 / `motorcycle` 23,899 (**[DERIVED]** from the label arrays). The two PNGs in `results/phase1_lightning/` come from this invalid comparison (title of `drift_scores.png`, read from the image: “Semantic Drift per Class (RemoteCLIP vs Fine-tuned VisDrone CLIP)”; bars ≈ L2 0.20 / cosine 0.36 for large-vehicle and ≈ 0.125 / 0.27 for small-vehicle — the same values `PROJECT_CONTEXT.md` §7a records).

### 3.6 Model artifacts on disk (all git-ignored)

| File | Bytes | mtime |
|---|---|---|
| `models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt` | 1,710,613,765 | 2026-04-28 22:31:09 |
| `models/finetuned_clip_visdrone/finetuned_clip_visdrone_epoch1.pt` | 605,272,564 | 2026-08-19 15:11:13 |
| `models/finetuned_clip_visdrone/finetuned_clip_visdrone_epoch2.pt` | 605,272,564 | 2026-08-19 15:57:59 |
| `models/finetuned_clip_visdrone/finetuned_clip_visdrone_epoch3.pt` | 605,272,564 | 2026-08-19 16:44:23 |
| `models/finetuned_clip_visdrone_heldout/finetuned_clip_visdrone_heldout_epoch1.pt` | 605,275,780 | 2026-08-20 20:12:28 |
| `models/finetuned_clip_visdrone_heldout/finetuned_clip_visdrone_heldout_epoch2.pt` | 605,275,780 | 2026-08-20 20:48:26 |
| `models/finetuned_clip_visdrone_heldout/finetuned_clip_visdrone_heldout_epoch3.pt` | 605,275,780 | 2026-08-20 21:24:54 |

### 3.7 `src/embeddings.py` — complete file

| Name | Lines | Kind |
|---|---|---|
| `load_remoteclip` | 8–17 | function |
| `load_clip` | 20–25 | function |
| `encode_with_remoteclip` | 28–56 | function |
| `encode_with_clip` | 59–87 | function |
| `CropPromptDataset` | 90–112 | class |
| `clip_contrastive_loss` | 115–122 | function |
| `finetune_clip_on_crops` | 125–202 | function |

`src/embeddings.py` (202 lines, verbatim)

```python
import os
import time

import numpy as np
import torch


def load_remoteclip(checkpoint_path, device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained=None)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"RemoteCLIP checkpoint did not load cleanly: missing={missing}, unexpected={unexpected}")
    model = model.to(device).eval()
    return model, preprocess


def load_clip(device, model_name="openai/clip-vit-base-patch32"):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def encode_with_remoteclip(crops, model, preprocess, device, batch_size=16, progress_every=10, log=print):
    """crops: list of (PIL.Image, label). Returns (features [N, D] float32, labels [N])."""
    feats, labels = [], []
    batch_imgs, batch_labels = [], []
    n_done, t0 = 0, time.time()

    def flush():
        nonlocal n_done
        with torch.no_grad():
            t = torch.stack(batch_imgs).to(device)
            f = model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
        labels.extend(batch_labels)
        n_done += len(batch_imgs)
        if log is not None and (n_done // batch_size) % progress_every == 0:
            elapsed = time.time() - t0
            log(f"  [remoteclip] {n_done}/{len(crops)} crops, {n_done/elapsed:.1f} crops/s, {elapsed:.0f}s elapsed", flush=True)

    for crop, label in crops:
        batch_imgs.append(preprocess(crop))
        batch_labels.append(label)
        if len(batch_imgs) == batch_size:
            flush()
            batch_imgs, batch_labels = [], []
    if batch_imgs:
        flush()

    return np.vstack(feats), np.array(labels)


def encode_with_clip(crops, model, processor, device, batch_size=16, progress_every=10, log=print):
    """crops: list of (PIL.Image, label). Returns (features [N, D] float32, labels [N])."""
    feats, labels = [], []
    batch_imgs, batch_labels = [], []
    n_done, t0 = 0, time.time()

    def flush():
        nonlocal n_done
        inputs = processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            f = model.get_image_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu().numpy())
        labels.extend(batch_labels)
        n_done += len(batch_imgs)
        if log is not None and (n_done // batch_size) % progress_every == 0:
            elapsed = time.time() - t0
            log(f"  [clip] {n_done}/{len(crops)} crops, {n_done/elapsed:.1f} crops/s, {elapsed:.0f}s elapsed", flush=True)

    for crop, label in crops:
        batch_imgs.append(crop)
        batch_labels.append(label)
        if len(batch_imgs) == batch_size:
            flush()
            batch_imgs, batch_labels = [], []
    if batch_imgs:
        flush()

    return np.vstack(feats), np.array(labels)


class CropPromptDataset(torch.utils.data.Dataset):
    """Wraps (PIL.Image, label) crops as (image, text-prompt) pairs for CLIP contrastive fine-tuning."""

    def __init__(self, crops, processor, prompt_template="a photo of a {}"):
        self.crops = crops
        self.processor = processor
        self.prompt_template = prompt_template

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        crop, label = self.crops[idx]
        prompt = self.prompt_template.format(label)
        inputs = self.processor(
            text=[prompt], images=crop, return_tensors="pt",
            padding="max_length", truncation=True, max_length=77,
        )
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }


def clip_contrastive_loss(logits_per_image):
    """Standard CLIP image/text contrastive loss -- diagonal entries are the correct pairs."""
    import torch.nn.functional as F

    labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_image.T, labels)
    return (loss_i + loss_t) / 2


def finetune_clip_on_crops(
    crops, model, processor, device,
    num_epochs=3, batch_size=8, lr=1e-5,
    unfreeze_patterns=(
        "vision_model.encoder.layers.11", "vision_model.encoder.layers.10",
        "vision_model.post_layernorm", "visual_projection",
    ),
    prompt_template="a photo of a {}",
    checkpoint_dir=None, checkpoint_prefix="finetuned_clip",
    progress_every=20, log=print,
):
    """Partial contrastive fine-tune of a CLIPModel on (crop, label) pairs.

    Only parameters whose name contains one of unfreeze_patterns are trained (default: last
    2 vision-encoder layers + post-layernorm + visual projection). Saves a checkpoint after
    every epoch if checkpoint_dir is given. Returns the model in eval() mode.
    """
    for name, param in model.named_parameters():
        param.requires_grad = any(p in name for p in unfreeze_patterns)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if log is not None:
        log(f"trainable params: {trainable:,} / {total:,}", flush=True)

    dataset = CropPromptDataset(crops, processor, prompt_template)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, len(dataloader) * num_epochs))

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        t0 = time.time()
        for step, batch in enumerate(dataloader):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            loss = clip_contrastive_loss(outputs.logits_per_image)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if log is not None and (step + 1) % progress_every == 0:
                elapsed = time.time() - t0
                done = (step + 1) * batch_size
                rate = done / elapsed
                remaining_steps = len(dataloader) - (step + 1)
                eta_this_epoch = remaining_steps * batch_size / rate if rate > 0 else float("nan")
                log(
                    f"  epoch {epoch+1}/{num_epochs} step {step+1}/{len(dataloader)} "
                    f"loss={loss.item():.4f} avg_loss={total_loss/(step+1):.4f} "
                    f"{rate:.1f} crops/s elapsed={elapsed:.0f}s eta_this_epoch={eta_this_epoch:.0f}s",
                    flush=True,
                )

        avg_loss = total_loss / max(1, len(dataloader))
        epoch_time = time.time() - t0
        if log is not None:
            log(f"epoch {epoch+1}/{num_epochs} done. avg_loss={avg_loss:.4f} time={epoch_time:.0f}s", flush=True)

        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            if log is not None:
                log(f"  saved checkpoint: {ckpt_path}", flush=True)

    model.eval()
    return model
```

---

## 4. Metric implementations

### 4.1 Where the metric code lives

`src/metrics.py`, `src/centroids.py` and `src/visualization.py` are **empty (0 bytes; committed empty in `36209f7`)**. `notebooks/02_full_dataset.ipynb` and `notebooks/04_drift_analysis.ipynb` are empty notebooks (0 cells, 240 bytes each). So there is **no shared metrics module**: the drift metric is re-implemented inline, with small differences, in each driver script / notebook. Every implementation is quoted below.

| Where | What it computes | Location |
|---|---|---|
| `run_remoteclip_vs_clip.py` | L2, cosine, drift_ratio, intra radii (+ sample counts) — DOTA↔VisDrone, CLIP & RemoteCLIP | `compute_drift`, lines 49–71 |
| `notebooks/03_remoteclip.ipynb` | same metrics, recomputed from cached embeddings as a sanity check; % reduction; bar plot | cells 3, 5, 8 |
| `notebooks/04_finetune_visdrone_clip.ipynb` | same metrics for the 3 arms (CLIP, RemoteCLIP, fine-tuned) | cell 15 |
| `run_xview_vs_visdrone.py` | same metrics, per `EXPANDED_VOCAB` class, gated by `MIN_SAMPLES_FOR_DRIFT = 30` | `compute_drift`, lines 49–69 |
| `run_uavid_building.py` | same metrics for the single `building` class (xView vs UAVid) | `compute_drift`, lines 46–58 |
| `verify_drift_seeds.py` | **L2 only** (no cosine, no ratio) — multi-seed check (never produced output) | `compute_drift`, lines 40–49 |
| `notebooks/01_phase1_reproduction.ipynb` / `results/phase1_lightning/Capstone_v1.ipynb` | Phase-1 whole-image drift (**L2 only, on zero-shot-predicted labels**) + cross-class distance heat-map | cells 18, 22 / cells 82–84, 91 |
| `make_category_yardstick_plots.py` | cross-altitude cosine vs cross-category cosine and their ratio | `cosine_dist`, `centroids`, `compute_table` |
| `make_normalized_drift_plot.py` | drift as % of the CLIP baseline | `pct_of_baseline` |
| `make_drift_ratio_plot.py` | plots stored `drift_ratio`; computes nothing new | — |

### 4.2 The core metrics — reference implementation (verbatim)

`run_remoteclip_vs_clip.py` (the script that produced `drift_comparison.json`):

`run_remoteclip_vs_clip.py:49-71`

```python
def compute_drift(dota_feats, dota_labels, vis_feats, vis_labels):
    scores = {}
    for cls in COMMON_VOCAB:
        sat = dota_feats[dota_labels == cls]
        uav = vis_feats[vis_labels == cls]
        if len(sat) == 0 or len(uav) == 0:
            log(f"  skipping {cls}: sat={len(sat)} uav={len(uav)}")
            continue
        sat_c, uav_c = sat.mean(axis=0), uav.mean(axis=0)
        l2 = float(norm(sat_c - uav_c))
        cos = float(1 - np.dot(sat_c, uav_c) / (norm(sat_c) * norm(uav_c)))
        sat_intra = float(np.mean(norm(sat - sat_c, axis=1)))
        uav_intra = float(np.mean(norm(uav - uav_c, axis=1)))
        scores[cls] = {
            "l2_drift": round(l2, 4),
            "cosine_drift": round(cos, 4),
            "drift_ratio": round(l2 / ((sat_intra + uav_intra) / 2), 4),
            "sat_samples": int(len(sat)),
            "uav_samples": int(len(uav)),
            "sat_intra_var": round(sat_intra, 4),
            "uav_intra_var": round(uav_intra, 4),
        }
    return scores
```

**Formulas exactly as implemented.** For a class *c*, let `sat` and `uav` be the matrices of L2-normalised embeddings (one row per crop) of that class in the two domains, `sat_c = sat.mean(0)`, `uav_c = uav.mean(0)` (plain arithmetic means; **the centroids are not re-normalised**):

```text
l2_drift(c)      = || sat_c - uav_c ||_2
cosine_drift(c)  = 1 - (sat_c . uav_c) / ( ||sat_c|| * ||uav_c|| )
sat_intra(c)     = mean_i || sat[i] - sat_c ||_2        # mean Euclidean distance to the domain centroid ("radius"); not a variance
uav_intra(c)     = mean_i || uav[i] - uav_c ||_2
drift_ratio(c)   = l2_drift(c) / ( (sat_intra(c) + uav_intra(c)) / 2 )
```

Every value is `float(...)` then `round(..., 4)`. In the code the intra-domain quantities are stored under the keys `sat_intra_var` / `uav_intra_var` (only in the script/`drift_comparison.json`; the notebook-4 and xView/UAVid JSONs store just the ratio) although they are mean distances, not variances. Nothing is standardised, whitened or PCA-reduced: each encoder's drift is computed in its native space (512-d CLIP, 768-d RemoteCLIP).

### 4.3 Every other implementation of the drift metric (verbatim)

**Notebook 03, cell 3** (identical logic; stores no `*_intra_var` keys) and **cell 5** (the “% reduction” calculation):

```python
from numpy.linalg import norm

COMMON_VOCAB = ["small-vehicle", "large-vehicle"]

def compute_drift(dota_feats, dota_labels, vis_feats, vis_labels):
    scores = {}
    for cls in COMMON_VOCAB:
        sat = dota_feats[dota_labels == cls]
        uav = vis_feats[vis_labels == cls]
        if len(sat) == 0 or len(uav) == 0:
            continue
        sat_c, uav_c = sat.mean(axis=0), uav.mean(axis=0)
        l2 = float(norm(sat_c - uav_c))
        cos = float(1 - np.dot(sat_c, uav_c) / (norm(sat_c) * norm(uav_c)))
        sat_intra = float(np.mean(norm(sat - sat_c, axis=1)))
        uav_intra = float(np.mean(norm(uav - uav_c, axis=1)))
        scores[cls] = {
            "l2_drift": round(l2, 4),
            "cosine_drift": round(cos, 4),
            "drift_ratio": round(l2 / ((sat_intra + uav_intra) / 2), 4),
            "sat_samples": int(len(sat)),
            "uav_samples": int(len(uav)),
        }
    return scores


remoteclip_drift = compute_drift(dota_rc_feats, dota_rc_labels, vis_rc_feats, vis_rc_labels)
clip_drift = compute_drift(dota_clip_feats, dota_clip_labels, vis_clip_feats, vis_clip_labels)

print("=== RemoteCLIP drift (native 768-dim space, both domains) ===")
for cls, s in remoteclip_drift.items():
    print(cls, s)

print("\n=== CLIP ViT-B/32 drift (native 512-dim space, both domains) ===")
for cls, s in clip_drift.items():
    print(cls, s)
```

```python
print(f"{'class':<16}{'CLIP L2':>10}{'RemoteCLIP L2':>16}{'reduction':>12}{'lower?':>12}")
for cls in COMMON_VOCAB:
    c, r = clip_drift[cls]["l2_drift"], remoteclip_drift[cls]["l2_drift"]
    pct = (c - r) / c * 100
    print(f"{cls:<16}{c:>10.4f}{r:>16.4f}{pct:>11.1f}%{'RemoteCLIP' if r < c else 'CLIP':>12}")
```

**Notebook 04 (v2), cell 15** — same function as notebook 03, plus the “lowest arm” printout:

```python
def compute_drift(dota_feats, dota_labels, vis_feats, vis_labels):
    scores = {}
    for cls in COMMON_VOCAB:
        sat = dota_feats[dota_labels == cls]
        uav = vis_feats[vis_labels == cls]
        if len(sat) == 0 or len(uav) == 0:
            continue
        sat_c, uav_c = sat.mean(axis=0), uav.mean(axis=0)
        l2 = float(norm(sat_c - uav_c))
        cos = float(1 - np.dot(sat_c, uav_c) / (norm(sat_c) * norm(uav_c)))
        sat_intra = float(np.mean(norm(sat - sat_c, axis=1)))
        uav_intra = float(np.mean(norm(uav - uav_c, axis=1)))
        scores[cls] = {
            "l2_drift": round(l2, 4),
            "cosine_drift": round(cos, 4),
            "drift_ratio": round(l2 / ((sat_intra + uav_intra) / 2), 4),
            "sat_samples": int(len(sat)),
            "uav_samples": int(len(uav)),
        }
    return scores


clip_drift = compute_drift(dota_clip_feats, dota_clip_labels, visdrone_clip_feats, visdrone_clip_labels)
remoteclip_drift = compute_drift(dota_rc_feats, dota_rc_labels, visdrone_rc_feats, visdrone_rc_labels)
finetuned_drift = compute_drift(dota_ft_feats, dota_ft_labels, visdrone_ft_feats, visdrone_ft_labels)

print(f"{'class':<16}{'CLIP':>10}{'RemoteCLIP':>12}{'FT-CLIP(VisDrone)':>20}{'lowest':>14}")
for cls in COMMON_VOCAB:
    vals = {"CLIP": clip_drift[cls]["l2_drift"], "RemoteCLIP": remoteclip_drift[cls]["l2_drift"], "FT-CLIP(VisDrone)": finetuned_drift[cls]["l2_drift"]}
    lowest = min(vals, key=vals.get)
    print(f"{cls:<16}{vals['CLIP']:>10.4f}{vals['RemoteCLIP']:>12.4f}{vals['FT-CLIP(VisDrone)']:>20.4f}{lowest:>14}")
```

**`run_xview_vs_visdrone.py`** — adds the sample-count gate; result keys are `xview_samples` / `visdrone_samples`:

`run_xview_vs_visdrone.py:31-31`

```python
MIN_SAMPLES_FOR_DRIFT = 30  # classes below this on either side are reported but excluded from drift
```

`run_xview_vs_visdrone.py:49-69`

```python
def compute_drift(xview_feats, xview_labels, vis_feats, vis_labels):
    scores = {}
    for cls in EXPANDED_VOCAB:
        xv = xview_feats[xview_labels == cls]
        uav = vis_feats[vis_labels == cls]
        if len(xv) < MIN_SAMPLES_FOR_DRIFT or len(uav) < MIN_SAMPLES_FOR_DRIFT:
            log(f"  skipping {cls}: xview={len(xv)} visdrone={len(uav)} (below MIN_SAMPLES_FOR_DRIFT={MIN_SAMPLES_FOR_DRIFT})")
            continue
        xv_c, uav_c = xv.mean(axis=0), uav.mean(axis=0)
        l2 = float(norm(xv_c - uav_c))
        cos = float(1 - np.dot(xv_c, uav_c) / (norm(xv_c) * norm(uav_c)))
        xv_intra = float(np.mean(norm(xv - xv_c, axis=1)))
        uav_intra = float(np.mean(norm(uav - uav_c, axis=1)))
        scores[cls] = {
            "l2_drift": round(l2, 4),
            "cosine_drift": round(cos, 4),
            "drift_ratio": round(l2 / ((xv_intra + uav_intra) / 2), 4),
            "xview_samples": int(len(xv)),
            "visdrone_samples": int(len(uav)),
        }
    return scores
```

**`run_uavid_building.py`** — single class, no gate inside the function (the gate is applied to the crop counts before encoding: `if len(uavid_crops) < MIN_SAMPLES_FOR_DRIFT or len(xview_crops) < MIN_SAMPLES_FOR_DRIFT: ... return`); note the result key `visdrone_samples` actually holds the UAVid count (the code comment says so):

`run_uavid_building.py:46-58`

```python
def compute_drift(xview_feats, uav_feats):
    xv_c, uav_c = xview_feats.mean(axis=0), uav_feats.mean(axis=0)
    l2 = float(norm(xv_c - uav_c))
    cos = float(1 - np.dot(xv_c, uav_c) / (norm(xv_c) * norm(uav_c)))
    xv_intra = float(np.mean(norm(xview_feats - xv_c, axis=1)))
    uav_intra = float(np.mean(norm(uav_feats - uav_c, axis=1)))
    return {
        "l2_drift": round(l2, 4),
        "cosine_drift": round(cos, 4),
        "drift_ratio": round(l2 / ((xv_intra + uav_intra) / 2), 4),
        "xview_samples": int(len(xview_feats)),
        "visdrone_samples": int(len(uav_feats)),  # keeping key name consistent with the other file (uav side)
    }
```

**`verify_drift_seeds.py`** — L2 only:

`verify_drift_seeds.py:40-49`

```python
def compute_drift(dota_feats, dota_labels, vis_feats, vis_labels):
    scores = {}
    for cls in COMMON_VOCAB:
        sat = dota_feats[dota_labels == cls]
        uav = vis_feats[vis_labels == cls]
        if len(sat) == 0 or len(uav) == 0:
            continue
        sat_c, uav_c = sat.mean(axis=0), uav.mean(axis=0)
        scores[cls] = float(norm(sat_c - uav_c))
    return scores
```

**Phase-1 whole-image drift** (`notebooks/01_phase1_reproduction.ipynb` cell 18; identical to `results/phase1_lightning/Capstone_v1.ipynb` cells 81–83) — L2 only, classes = the zero-shot **predicted** label, 300 whole images per domain:

```python
combined_features = np.vstack([image_features_array, visdrone_features_array])
combined_labels = final_predictions + visdrone_predictions
domain_labels = (["satellite"] * len(final_predictions)) + (["uav"] * len(visdrone_predictions))

class_embeddings = {}
for i in range(len(combined_features)):
    cls = combined_labels[i]
    domain = domain_labels[i]
    class_embeddings.setdefault(cls, {"satellite": [], "uav": []})
    class_embeddings[cls][domain].append(combined_features[i])

class_centroids = {}
for cls, d in class_embeddings.items():
    sat = np.array(d["satellite"])
    uav = np.array(d["uav"])
    if len(sat) == 0 or len(uav) == 0:
        continue
    class_centroids[cls] = (sat.mean(axis=0), uav.mean(axis=0))

from numpy.linalg import norm

drift_scores = {cls: float(norm(sat_c - uav_c)) for cls, (sat_c, uav_c) in class_centroids.items()}

for cls, score in sorted(drift_scores.items(), key=lambda kv: kv[1]):
    print(f"{cls}: Drift Score = {score:.4f}")
```

**Phase-1 cross-domain distance matrix** (`notebooks/01_phase1_reproduction.ipynb` cell 22, the “heat-map”; entry (i, j) = L2 between the *satellite centroid of class i* and the *UAV centroid of class j*):

```python
classes_h = list(class_centroids.keys())
n = len(classes_h)
distance_matrix = np.zeros((n, n))

for i, cls_sat in enumerate(classes_h):
    for j, cls_uav in enumerate(classes_h):
        sat_vec = class_centroids[cls_sat][0]
        uav_vec = class_centroids[cls_uav][1]
        distance_matrix[i, j] = norm(sat_vec - uav_vec)
```

### 4.4 Derived-metric code (verbatim)

**Cross-altitude vs cross-category yardstick** — `make_category_yardstick_plots.py` (docstring explains the intent):

`make_category_yardstick_plots.py:1-17`

```python
"""
Answers "is the cross-altitude cosine drift actually meaningful, given how small it looks in
absolute terms?" by comparing it against a natural yardstick: the cosine distance between two
genuinely DIFFERENT categories within the same domain (e.g. small-vehicle vs. large-vehicle,
both from DOTA). CLIP embeddings are unit-normalized, so cosine distance is the space's native
notion of "how different two things are" -- comparing the cross-altitude, same-category distance
against the same-altitude, different-category distance tells us whether crossing altitude moves
an object's embedding by roughly as much as swapping it for a different kind of object.

For each of the 3 comparisons, for each of CLIP/RemoteCLIP:
  cross_altitude[c]   = cosine_dist(centroid_domainA[c], centroid_domainB[c])           (drift, as reported)
  cross_category[c]   = mean over c' != c of cosine_dist to c' within domain A and domain B
  ratio[c]            = cross_altitude[c] / cross_category[c]   (~1 => altitude-crossing is as
                                                                    big a jump as a real category change)

Uses the same cached embeddings as make_pca_tsne_plots.py -- no re-encoding needed.
"""
```

`make_category_yardstick_plots.py:35-36`

```python
def cosine_dist(a, b):
    return 1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

`make_category_yardstick_plots.py:39-40`

```python
def centroids(feats, labels, classes):
    return {c: feats[labels == c].mean(axis=0) for c in classes}
```

`make_category_yardstick_plots.py:43-60`

```python
def compute_table(sat_feats, sat_labels, uav_feats, uav_labels, classes):
    sat_c = centroids(sat_feats, sat_labels, classes)
    uav_c = centroids(uav_feats, uav_labels, classes)
    rows = {}
    for c in classes:
        cross_altitude = cosine_dist(sat_c[c], uav_c[c])
        others = [c2 for c2 in classes if c2 != c]
        same_domain_dists = []
        for c2 in others:
            same_domain_dists.append(cosine_dist(sat_c[c], sat_c[c2]))
            same_domain_dists.append(cosine_dist(uav_c[c], uav_c[c2]))
        cross_category = float(np.mean(same_domain_dists)) if same_domain_dists else float("nan")
        rows[c] = {
            "cross_altitude_cosine": round(float(cross_altitude), 4),
            "cross_category_cosine": round(cross_category, 4),
            "ratio": round(float(cross_altitude) / cross_category, 4) if cross_category else float("nan"),
        }
    return rows
```

Note: for the 2-class DOTA comparison `cross_category_cosine` is identical for both classes (there is only one “other” class, so the mean is over the same two distances).

**Normalised drift reduction** — `make_normalized_drift_plot.py`:

`make_normalized_drift_plot.py:34-35`

```python
def pct_of_baseline(data, model, cls, metric):
    return 100 * data[model][cls][metric] / data["clip_vit_b32"][cls][metric]
```

The docstring of that file states “L2 ~ sqrt(2 * cosine_distance)” to explain why the cosine bars look small on a shared axis; see §4.5 for how well that approximation holds for these (non-unit) centroids.

**PCA / t-SNE settings** (`make_pca_tsne_plots.py`, verbatim):

`make_pca_tsne_plots.py:29-30`

```python
RNG_SEED = 42
MAX_POINTS_PER_SIDE = 900  # subsample for t-SNE speed/legibility if a side is larger
```

`make_pca_tsne_plots.py:49-60`

```python
def make_panel(ax, feats, labels, domains, class_order, method, title):
    cmap = plt.colormaps.get_cmap("tab10")
    class_to_color = {c: cmap(i / max(len(class_order), 1)) for i, c in enumerate(class_order)}

    if method == "pca":
        reducer = PCA(n_components=2, random_state=RNG_SEED)
    else:
        perplexity = min(30, max(5, (len(labels) - 1) // 3))
        reducer = TSNE(n_components=2, perplexity=perplexity, learning_rate=200,
                        init="pca", random_state=RNG_SEED)
    coords = reducer.fit_transform(feats)
```

Reading of those settings **[ANALYSIS]**: PCA (2 components) and t-SNE (`perplexity = min(30, max(5, (N-1)//3))`, `learning_rate=200`, `init="pca"`, `random_state=42`) are each fit **jointly on both domains and all classes of one encoder's panel** (so no per-domain independent projection). `MAX_POINTS_PER_SIDE = 900` subsamples a side *randomly and not class-balanced* only when it has more than 900 points — which happens for the 4-class figure (1,200 points per side), not for the 2-class (600) or 3-class (900) figures. The Phase-1 notebooks use `TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000 [local] / max_iter=1000 [Lightning], random_state=42)` and `PCA(n_components=2)` fit on the *combined* whole-image features.

### 4.5 Diagnostic table: every stored drift number re-derived, with the quantities behind it [DERIVED]

Recomputed from the cached embeddings with the formulas of §4.2 (script run for this report). **All 16 class×encoder entries in the table (20 comparisons against the stored JSONs, since the DOTA CLIP/RemoteCLIP entries appear in two files) match the stored values exactly** (largest absolute difference across l2/cos/ratio: 0.0). The extra columns are quantities the JSONs do not store for every arm: the intra-domain radii (the `drift_ratio` denominator) and the centroid norms.

| Comparison | Encoder | Class | n sat/xView | n UAV | L2 | cosine | √(2·cos) | sat intra | UAV intra | ‖sat centroid‖ | ‖UAV centroid‖ | drift_ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DOTA-VisDrone | CLIP | small-vehicle | 300 | 300 | 0.2460 | 0.0346 | 0.2631 | 0.3286 | 0.4286 | 0.9428 | 0.8992 | 0.6498 |
| DOTA-VisDrone | CLIP | large-vehicle | 300 | 300 | 0.2596 | 0.0406 | 0.2850 | 0.3855 | 0.4471 | 0.9196 | 0.8887 | 0.6235 |
| DOTA-VisDrone | RemoteCLIP | small-vehicle | 300 | 300 | 0.1892 | 0.0199 | 0.1995 | 0.3019 | 0.3707 | 0.9519 | 0.9248 | 0.5627 |
| DOTA-VisDrone | RemoteCLIP | large-vehicle | 300 | 300 | 0.2276 | 0.0300 | 0.2449 | 0.3444 | 0.3837 | 0.9357 | 0.9182 | 0.6252 |
| DOTA-VisDrone | FineTuned-heldout | small-vehicle | 300 | 300 | 0.1585 | 0.0129 | 0.1606 | 0.2523 | 0.136 | 0.9633 | 0.9876 | 0.8165 |
| DOTA-VisDrone | FineTuned-heldout | large-vehicle | 300 | 300 | 0.0955 | 0.0048 | 0.0980 | 0.2221 | 0.2399 | 0.972 | 0.9681 | 0.4132 |
| DOTA-VisDrone | FineTuned-LEAKY-first-run | small-vehicle | 300 | 300 | 0.1412 | 0.0102 | 0.1428 | 0.2349 | 0.1423 | 0.9685 | 0.9873 | 0.7486 |
| DOTA-VisDrone | FineTuned-LEAKY-first-run | large-vehicle | 300 | 300 | 0.0964 | 0.0049 | 0.0990 | 0.2258 | 0.206 | 0.9712 | 0.976 | 0.4467 |
| xView-VisDrone | clip | car | 300 | 300 | 0.3564 | 0.0718 | 0.3789 | 0.2795 | 0.4383 | 0.9578 | 0.8943 | 0.9931 |
| xView-VisDrone | clip | bus | 300 | 300 | 0.3256 | 0.0603 | 0.3473 | 0.3013 | 0.4701 | 0.9512 | 0.8776 | 0.8443 |
| xView-VisDrone | clip | truck | 300 | 300 | 0.2837 | 0.0451 | 0.3003 | 0.2985 | 0.4529 | 0.9518 | 0.8842 | 0.7550 |
| xView-UAVid | clip | building | 300 | 300 | 0.5417 | 0.1822 | 0.6037 | 0.3467 | 0.5401 | 0.9321 | 0.8382 | 1.2219 |
| xView-VisDrone | remoteclip | car | 300 | 300 | 0.3024 | 0.0503 | 0.3172 | 0.2496 | 0.3726 | 0.9647 | 0.9239 | 0.9719 |
| xView-VisDrone | remoteclip | bus | 300 | 300 | 0.2802 | 0.0434 | 0.2946 | 0.2682 | 0.3919 | 0.9617 | 0.9162 | 0.8491 |
| xView-VisDrone | remoteclip | truck | 300 | 300 | 0.2366 | 0.0303 | 0.2462 | 0.2538 | 0.3771 | 0.9655 | 0.9201 | 0.7500 |
| xView-UAVid | remoteclip | building | 300 | 300 | 0.3783 | 0.0847 | 0.4116 | 0.3296 | 0.4416 | 0.9344 | 0.8942 | 0.9809 |

Rows labelled `FineTuned-LEAKY-first-run` reproduce `drift_comparison_3way.json` (first run); `FineTuned-heldout` reproduces `drift_comparison_3way_heldout.json`; `xView-VisDrone` rows use xView vs *all* VisDrone images; `xView-UAVid` is the building comparison.

**[ANALYSIS]** Because the centroid of *n* unit vectors has norm < 1 (here 0.84–0.99 across all rows, see columns), `||μs − μu||² =(‖μs‖ − ‖μu‖)² + 2‖μs‖‖μu‖·(1 − cos θ)`; the “L2 ≈ √(2·cosine)” shorthand in `make_normalized_drift_plot.py` is therefore only approximate (compare the L2 and √(2·cos) columns). It also means `l2_drift` mixes centroid *direction* change with centroid *length* change, whereas `cosine_drift` is direction only.

### 4.6 Text-embedding-based metrics (anchor alignment, semantic margin, …) — thorough status check

**Conclusion: none of these metrics is implemented — not fully, not partially, not in a notebook, not in another branch. The status flagged by the prior paper version is unchanged.** Evidence:

1. **Search of all code/notebook/doc text.** A regex search over every `.py`, `.ipynb` (source cells *and* saved outputs), `.yaml`, `.md` and `.json` file for `get_text_features|encode_text|logits_per_text|text_features|text_embed|text_centroid|get_tokenizer|open_clip.tokenize|tokenizer(|CLIPTokenizer|text_model|text_projection|text_inputs|semantic margin|margin|anchor` returns only the following (the two `bbox_to_anchor` hits are matplotlib legend placement):

| File | Where | Matching line |
|---|---|---|
| `make_pca_tsne_plots.py` | line 83 | `bbox_to_anchor=(0.5, 0.02), ncol=min(len(class_order), 6), fontsize=9)` |
| `make_pca_tsne_plots.py` | line 86 | `bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9)` |
| `notebooks/01_phase1_reproduction.ipynb` | cell 9 (source) | `text_inputs = processor(text=dota_class_names, return_tensors="pt", padding=True).to(device)` |
| `notebooks/01_phase1_reproduction.ipynb` | cell 9 (source) | `text_features = model.get_text_features(**text_inputs)` |
| `notebooks/01_phase1_reproduction.ipynb` | cell 9 (source) | `text_features = text_features / text_features.norm(dim=-1, keepdim=True)` |
| `notebooks/01_phase1_reproduction.ipynb` | cell 11 (source) | `similarity = image_features_array @ text_features.cpu().numpy().T` |
| `notebooks/01_phase1_reproduction.ipynb` | cell 13 (source) | `text_inputs_common = processor(text=common_classes, return_tensors="pt", padding=True).to(device)` |
| `notebooks/01_phase1_reproduction.ipynb` | cell 13 (source) | `text_features_common = model.get_text_features(**text_inputs_common)` |
| `notebooks/01_phase1_reproduction.ipynb` | cell 13 (source) | `text_features_common = text_features_common / text_features_common.norm(dim=-1, keepdim=True)` |
| `notebooks/01_phase1_reproduction.ipynb` | cell 13 (source) | `similarity_common = image_features_array @ text_features_common.cpu().numpy().T` |
| `notebooks/01_phase1_reproduction.ipynb` | cell 16 (source) | `similarity_visdrone = visdrone_features_array @ text_features_common.cpu().numpy().T` |
| `results/phase1_lightning/Capstone_v1.ipynb` | cell 49 (source) | `text_inputs = processor(` |
| `results/phase1_lightning/Capstone_v1.ipynb` | cell 49 (source) | `text_features = model.get_text_features(**text_inputs)` |
| `results/phase1_lightning/Capstone_v1.ipynb` | cell 49 (source) | `text_features = text_features / text_features.norm(dim=-1, keepdim=True)` |
| `results/phase1_lightning/Capstone_v1.ipynb` | cell 51 (source) | `similarity = image_features_array @ text_features.cpu().numpy().T` |
| `results/phase1_lightning/Capstone_v1.ipynb` | cell 62 (source) | `text_inputs_common = processor(` |
| `results/phase1_lightning/Capstone_v1.ipynb` | cell 62 (source) | `text_features_common = model.get_text_features(**text_inputs_common)` |
| `results/phase1_lightning/Capstone_v1.ipynb` | cell 62 (source) | `text_features_common = text_features_common / text_features_common.norm(dim=-1, keepdim=True)` |
| `results/phase1_lightning/Capstone_v1.ipynb` | cell 63 (source) | `similarity_common = image_features_array @ text_features_common.cpu().numpy().T` |
| `results/phase1_lightning/Capstone_v1.ipynb` | cell 72 (source) | `similarity_visdrone = visdrone_features_array @ text_features_common.cpu().numpy().T` |

2. **What those hits are.** All `get_text_features` calls are in the two Phase-1 notebooks and are used only to build a zero-shot classifier: class-name strings → text embeddings → `argmax` of image·text similarity → a *label* for each whole image. That label then defines the class buckets whose centroids are compared across domains. No similarity value between a visual centroid and a text embedding is ever stored, reported or used as a metric. Verbatim (`notebooks/01_phase1_reproduction.ipynb` cells 9, 11, 13):

```python
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

dota_class_names = ["plane", "ship", "vehicle", "building", "harbor"]

text_inputs = processor(text=dota_class_names, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_features = model.get_text_features(**text_inputs)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# ---- cell 11 ----
similarity = image_features_array @ text_features.cpu().numpy().T
predictions = similarity.argmax(axis=1)

for i in range(5):
    print(f"{image_names[i]} -> {dota_class_names[predictions[i]]}")

# ---- cell 13 ----
mapped_predictions = [class_mapping.get(dota_class_names[p], dota_class_names[p]) for p in predictions]

text_inputs_common = processor(text=common_classes, return_tensors="pt", padding=True).to(device)
with torch.no_grad():
    text_features_common = model.get_text_features(**text_inputs_common)
    text_features_common = text_features_common / text_features_common.norm(dim=-1, keepdim=True)

similarity_common = image_features_array @ text_features_common.cpu().numpy().T
new_predictions = similarity_common.argmax(axis=1)
final_predictions = [common_classes[i] for i in new_predictions]

for i in range(5):
    print(f"{image_names[i]}: before={mapped_predictions[i]!r}  after_recompute={final_predictions[i]!r}")
```

3. **Fine-tuning uses the text tower only inside the contrastive loss** (`outputs.logits_per_image` in `finetune_clip_on_crops`); the text tower is frozen (verified bit-identical to base CLIP, §3.4.5); no text embedding is extracted or saved.

4. **No text embeddings exist in `embeddings/`.** All 44 `.npy` arrays are image-feature or label arrays (full inventory in §5.1). `results/` contains no similarity/margin JSON.

5. **`src/metrics.py` is 0 bytes; `notebooks/04_drift_analysis.ipynb` and `02_full_dataset.ipynb` have 0 cells.** `git branch -a` shows only `main` and `remotes/origin/main`; `git stash list` is empty; the two untracked files are the slide deck and its PDF export.

6. **What the project documents define as the metrics.** The proposal PDF (`Capstone Proposal.pdf`, methodology Phase 3, **[EXTERNAL]**) defines only `Drift(w) = ‖µ(E_w^s) − µ(E_w^a)‖` (the L2 centroid distance; the norm bars extract as `|||` from the PDF) and, for Phase 5, a *planned* altitude-conditioned text embedding `T'_w = f(T_w, altitude)` via a small MLP/adapter (not implemented). Phase 6 lists mAP, cross-altitude gap, semantic-drift reduction, “Semantic Confusion Rate, Drift vs Performance Correlation, In-Domain vs Cross-Domain mAP Difference & Per-Class mAP”. **Neither the proposal, the diagrams (`diagrams/fig4_3_drift_computation.drawio`: `D_L2(c) = ||μc^S − μc^U||_2`, `D_cos(c) = 1 − (μc^S · μc^U)/(||μc^S||·||μc^U||)`), the slide deck, nor any repo document defines “anchor alignment” or “semantic margin”.** If the prior paper version used those names, their definitions do not exist anywhere in this repository.

7. **[ANALYSIS] What would be needed** (not done): because the text tower of the fine-tuned model is identical to base CLIP and every cached image embedding is already L2-normalised in the model's joint space (512-d for CLIP and the fine-tuned arm, 768-d for RemoteCLIP), centroid-to-text similarities and margins over the four/two classes could be computed from `embeddings/**` plus freshly encoded prompt embeddings, with no image re-encoding. This is a possibility, not something present in the repo.

---

## 5. All experiment results

### 5.1 Inventory: every file under `results/` (and where results-like artifacts are not)

There are **no** `outputs/`, `logs/`, `runs/` or `checkpoints/` directories in the repository and **no** `.csv`, `.tsv`, `.pkl`, `.pickle`, `.log`, `.out` or `.h5` files anywhere (searched the whole tree except `data/raw`; the only `.txt` is the empty `requirements.txt`). The Lightning-era `outputs/` folder (~5.4 GB, `PROJECT_CONTEXT.md` §6) is not in the repo; only a 525 MB reduced export was kept (`capstone_phase1_export.tar.gz`, untracked/ignored). Run logs survive **only** as stdout stored inside notebooks. Results therefore exist as: 7 JSON files, PNG figures, printed notebook output, and cached `.npy` embeddings.

| File | Bytes | mtime | Produced by |
|---|---|---|---|
| `results/category_yardstick/drift_ratio_comparison.png` | 103,924 | 2026-08-22 11:48:40 | `make_drift_ratio_plot.py` |
| `results/category_yardstick/normalized_drift_reduction.png` | 119,504 | 2026-08-22 11:56:15 | `make_normalized_drift_plot.py` |
| `results/category_yardstick/yardstick_dota_vs_visdrone.png` | 82,648 | 2026-08-22 11:47:15 | `make_category_yardstick_plots.py` |
| `results/category_yardstick/yardstick_table.json` | 2,787 | 2026-08-22 11:47:16 | `make_category_yardstick_plots.py` |
| `results/category_yardstick/yardstick_xview_visdrone_uavid.png` | 84,503 | 2026-08-22 11:47:16 | `make_category_yardstick_plots.py` |
| `results/category_yardstick/yardstick_xview_vs_visdrone.png` | 75,530 | 2026-08-22 11:47:16 | `make_category_yardstick_plots.py` |
| `results/pca_tsne/pca_dota_vs_visdrone.png` | 346,289 | 2026-08-22 11:26:35 | `make_pca_tsne_plots.py` |
| `results/pca_tsne/pca_xview_visdrone_uavid.png` | 433,221 | 2026-08-22 11:26:57 | `make_pca_tsne_plots.py` |
| `results/pca_tsne/pca_xview_vs_visdrone.png` | 441,285 | 2026-08-22 11:26:46 | `make_pca_tsne_plots.py` |
| `results/pca_tsne/tsne_dota_vs_visdrone.png` | 358,616 | 2026-08-22 11:26:46 | `make_pca_tsne_plots.py` |
| `results/pca_tsne/tsne_xview_visdrone_uavid.png` | 550,669 | 2026-08-22 11:27:08 | `make_pca_tsne_plots.py` |
| `results/pca_tsne/tsne_xview_vs_visdrone.png` | 530,481 | 2026-08-22 11:26:57 | `make_pca_tsne_plots.py` |
| `results/phase1_lightning/Capstone_v1.ipynb` | 2,380,608 | 2026-08-13 17:29:11 | Lightning AI Studio (historical); the sound small-scale CLIP pipeline |
| `results/phase1_lightning/drift_scores.png` | 41,607 | 2026-08-13 17:29:11 | `model.ipynb` (missing) — invalid comparison |
| `results/phase1_lightning/tsne_drift.png` | 182,608 | 2026-08-13 17:29:11 | `model.ipynb` (missing) — invalid comparison |
| `results/phase1_local/distance_heatmap_local.png` | 73,823 | 2026-08-13 19:03:54 | `notebooks/01_phase1_reproduction.ipynb` cell 22 |
| `results/phase1_local/drift_scores_local.json` | 229 | 2026-08-13 19:03:54 | `notebooks/01_phase1_reproduction.ipynb` cell 23 |
| `results/phase1_local/drift_scores_local.png` | 50,754 | 2026-08-13 19:03:54 | `notebooks/01_phase1_reproduction.ipynb` cell 21 |
| `results/phase1_local/pca_combined.png` | 206,107 | 2026-08-13 19:03:39 | `notebooks/01_phase1_reproduction.ipynb` cell 19 |
| `results/phase1_local/pca_dota.png` | 126,405 | 2026-08-13 19:03:20 | `notebooks/01_phase1_reproduction.ipynb` cell 14 |
| `results/phase1_local/tsne_drift_local.png` | 198,572 | 2026-08-13 19:03:51 | `notebooks/01_phase1_reproduction.ipynb` cell 20 |
| `results/remoteclip_vs_clip/clip_vs_remoteclip_drift.png` | 54,763 | 2026-08-20 22:07:24 | `notebooks/03_remoteclip.ipynb` cell 8 |
| `results/remoteclip_vs_clip/drift_comparison.json` | 1,110 | 2026-08-20 22:06:54 | `run_remoteclip_vs_clip.py` (post-fix re-run, 2026-08-20 22:06) |
| `results/remoteclip_vs_clip/drift_comparison_3way.json` | 1,604 | 2026-08-19 16:47:07 | first (leaky) fine-tune notebook 04 v1 — **notebook not preserved** |
| `results/remoteclip_vs_clip/drift_comparison_3way_heldout.json` | 1,621 | 2026-08-20 21:36:59 | `notebooks/04_finetune_visdrone_clip.ipynb` cell 20 |
| `results/remoteclip_vs_clip/three_way_drift_comparison.png` | 66,097 | 2026-08-19 16:47:06 | notebook 04 v1 — **notebook not preserved** (leaky) |
| `results/remoteclip_vs_clip/three_way_drift_comparison_heldout.png` | 64,988 | 2026-08-20 21:36:59 | `notebooks/04_finetune_visdrone_clip.ipynb` cell 18 |
| `results/xview_vs_visdrone/drift_comparison.json` | 1,411 | 2026-08-20 22:23:47 | `run_xview_vs_visdrone.py` |
| `results/xview_vs_visdrone/drift_comparison_4class.json` | 2,088 | 2026-08-20 23:04:29 | `run_uavid_building.py` (merges the file above + building) |
| `results/xview_vs_visdrone/four_class_drift_comparison.png` | 51,182 | 2026-08-20 23:05:04 | **generating code not in repo** |
| `results/xview_vs_visdrone/xview_vs_visdrone_drift.png` | 59,642 | 2026-08-20 22:27:25 | **generating code not in repo** |

### 5.2 The experiment runs and their exact configurations

Run order is reconstructed from file-system timestamps (all local IST); git commits `5791ee2…c401c83` were made later in batches (see §9).

| Run | Date (mtime) | Script / notebook | Encoders | Data pools & sample size | Seeds | Outputs |
|---|---|---|---|---|---|---|
| **A** Phase-1 whole-image (Lightning) | undated (same notebook's dataset-download cells are dated 2026-04-05) | `results/phase1_lightning/Capstone_v1.ipynb` cells 46–84 | CLIP ViT-B/32 (HF) | first 300 DOTA + first 300 VisDrone **whole images** in `os.listdir` (unsorted) order; labels = zero-shot predicted over 7 mapped classes | none (t-SNE `random_state=42`) | printed drift scores only (cell 84); no JSON |
| **A′** Phase-1 crop sanity (Lightning) | undated | `results/phase1_lightning/Capstone_v1.ipynb` cells 39–44 | OpenAI `clip` ViT-B/32 (fp16 out) | crops from first 5 files of each dataset (`os.listdir()[:5]`) — **all** native classes, ground-truth labels | — | `embeddings/phase1_lightning/{dota,vis}_{feats,labels}.npy` |
| **B** Phase-1 local reproduction | 2026-08-13 19:02–19:03 | `notebooks/01_phase1_reproduction.ipynb` | CLIP ViT-B/32 (HF, fp32) | Part A: first 5 sorted files/dataset (crops); Part B: first 300 sorted whole images/dataset | t-SNE `random_state=42` | `results/phase1_local/*`, `embeddings/phase1_local/*` |
| **C** Lightning large-scale comparison | before Aug 13 | `model.ipynb` — **missing** | RemoteCLIP (DOTA) vs fine-tuned CLIP (VisDrone), two independent PCAs to 256-d | 28,767 DOTA crops; 217,153 VisDrone crops | unknown (notebook missing) | `embeddings/phase1_lightning/*_features.npy`, 2 PNGs. **Declared methodologically invalid** (`PROJECT_CONTEXT.md` §7a/§14a) |
| **D** first-pass 2-way (pre-fix) | 2026-08-14 (per `PROJECT_CONTEXT.md` §7b) | `run_remoteclip_vs_clip.py` (earlier version) | CLIP, RemoteCLIP | 300/class/domain, all DOTA + all VisDrone images, bicycle in small-vehicle | 42 | outputs **overwritten** by run G; numbers survive in `drift_comparison_3way.json` and `PROJECT_CONTEXT.md` §7b |
| **E** first 3-way (leaky) | 2026-08-19 15:11–16:47 | notebook 04 **v1** — overwritten | CLIP, RemoteCLIP (numbers loaded from D), fine-tuned (trained on all VisDrone) | eval 300/class from the same VisDrone pool used for training | eval 42 | `drift_comparison_3way.json`, `three_way_drift_comparison.png`, `*_finetuned_clip_*.npy` |
| **F** held-out 3-way | 2026-08-20 20:12–21:37 | `notebooks/04_finetune_visdrone_clip.ipynb` (v2) | CLIP, RemoteCLIP (re-encoded), fine-tuned (train pool only) | DOTA 300/class; VisDrone held-out 1,294 imgs → 300/class | eval 42; split 1234 | `drift_comparison_3way_heldout.json`, `three_way_drift_comparison_heldout.png`, `*_finetuned_clip_heldout_*.npy` |
| **G** 2-way re-run (post-fix) | 2026-08-20 22:03–22:07 | `run_remoteclip_vs_clip.py` | CLIP, RemoteCLIP | identical pools/seeds to F (script comment: “Must match notebooks/04…”) | eval 42; split 1234 | `drift_comparison.json`, `*_clip_*`, `*_remoteclip_*` arrays |
| **H** xView↔VisDrone | 2026-08-20 22:12–22:27 | `run_xview_vs_visdrone.py` | CLIP, RemoteCLIP | xView first 5,000 `train` images (streamed); VisDrone all 6,471 images; 300/class | 42 | `results/xview_vs_visdrone/drift_comparison.json`, `embeddings/xview_vs_visdrone/{xview,visdrone}_*` |
| **I** UAVid/xView building | 2026-08-20 23:01–23:05 | `run_uavid_building.py` | CLIP, RemoteCLIP | UAVid train+val (270 imgs, 1,199 building crops → 300); xView first 500 `train` images → 300 | 42 | `drift_comparison_4class.json`, `*_building_*` arrays |
| **J** analysis plots | 2026-08-22 11:26–11:56 | `make_pca_tsne_plots.py`, `make_category_yardstick_plots.py`, `make_drift_ratio_plot.py`, `make_normalized_drift_plot.py` | — | read cached `.npy` / JSON only; no encoding | PCA/t-SNE `random_state=42` | `results/pca_tsne/*`, `results/category_yardstick/*` |
| **K** seed sweep (intended) | script mtime 2026-08-19 18:19 | `verify_drift_seeds.py` | CLIP, RemoteCLIP, fine-tuned (**leaky** epoch-3) | all DOTA + **all** VisDrone crops; 300/class | [42, 123, 7, 2024, 99] | **none — `seed_sensitivity.json` does not exist** |

Exact constants of the driver scripts (verbatim):

`run_remoteclip_vs_clip.py:23-31`

```python
MAX_CROPS_PER_CLASS = 300  # reduced from 800 after the 30-min timeout; re-tune once real throughput is known
RANDOM_SEED = 42
BATCH_SIZE = 16

# Must match notebooks/04_finetune_visdrone_clip.ipynb exactly -- both this script and that
# notebook need to evaluate on the *same* VisDrone crops for the 2-way and 3-way results to be
# directly comparable (rather than two different measurements of "the same" thing).
HOLDOUT_FRACTION = 0.2
SPLIT_SEED = 1234
```

`run_xview_vs_visdrone.py:29-33`

```python
MAX_XVIEW_IMAGES = 5000  # bounded stream scan; enough for car/bus/truck at 300/class, van excluded
MAX_CROPS_PER_CLASS = 300
MIN_SAMPLES_FOR_DRIFT = 30  # classes below this on either side are reported but excluded from drift
RANDOM_SEED = 42
BATCH_SIZE = 16
```

`run_uavid_building.py:28-32`

```python
MAX_XVIEW_IMAGES_FOR_BUILDING = 500  # buildings are common in xView; should be plenty
MAX_CROPS_PER_CLASS = 300
MIN_SAMPLES_FOR_DRIFT = 30
RANDOM_SEED = 42
BATCH_SIZE = 16
```

`verify_drift_seeds.py:23-24`

```python
SEEDS = [42, 123, 7, 2024, 99]
MAX_CROPS_PER_CLASS = 300
```

`verify_drift_seeds.py:32-33`

```python
FINETUNED_CKPT = "models/finetuned_clip_visdrone/finetuned_clip_visdrone_epoch3.pt"
OUT_PATH = "results/remoteclip_vs_clip/seed_sensitivity.json"
```

### 5.3 Status of the items previously flagged “not yet done”

**(a) Results using more than 300 crops per category — none.**

Every encoded evaluation sample is capped by `MAX_CROPS_PER_CLASS = 300` (and `subsample_balanced`). Sample counts actually embedded, from the cached arrays (**[DERIVED]**, §5.7): DOTA↔VisDrone 300/class/domain; xView car/bus/truck 300 each (van 25); VisDrone (xView comparison) 300 each of car/bus/truck/van; building 300 (xView) + 300 (UAVid). Larger *pools* were scanned but not embedded: 146,631 VisDrone vehicle crops, 9,570 DOTA vehicle crops, 1,199 UAVid building crops. The fine-tuning used 167,699 crops (that is training, not drift evaluation). The Lightning-era arrays `remoteclip_dota_features.npy` (28,767 crops) and `finetuned_visdrone_features.npy` (217,153 crops) are the only large-N embedding sets, but their comparison is declared invalid and no drift number from them is reported.

**(b) Results using more than one random seed — none exist on disk.**

Repo-wide list of every seed / RNG setting (`git grep` + notebook source scan): `random.Random(seed)` in `split_visdrone_images` (1234) and `subsample_balanced` (42); `PCA/TSNE random_state=42` and `np.random.default_rng(42)` in `make_pca_tsne_plots.py`; `TSNE(random_state=42)` in the Phase-1 notebooks. There is **no `torch.manual_seed`, `np.random.seed`, `random.seed`, `cudnn.deterministic` or `PYTHONHASHSEED`** anywhere, so fine-tuning is unseeded. `verify_drift_seeds.py` was written to rerun the comparison for seeds 42/123/7/2024/99, but: (1) its output file `results/remoteclip_vs_clip/seed_sensitivity.json` does not exist on disk and never appears in git history; (2) it loads `models/finetuned_clip_visdrone/finetuned_clip_visdrone_epoch3.pt` — the **leaky** first-run checkpoint; (3) it scans `get_visdrone_crops(...)` with no `allowed_files`, i.e. evaluates on all VisDrone images including the ones that checkpoint was trained on; (4) it computes only L2. Even if run, it would not validate the held-out headline. `README.md`, `PROJECT_CONTEXT.md` (§21) and the deck (“Next Steps: Seed/sample-size robustness check”) all list the robustness check as still to do.

**(c) Results on the full DOTA dataset — none.**

Only `part1.zip` (469 images) and `labelTxt.zip` (1,411 label files; 469 matched) were ever extracted. `part2.zip` (4,757,024,280 B) and `part3.zip` (3,319,108,346 B) sit unextracted in `data/raw/DOTA/` (`PROJECT_CONTEXT.md` §4: pending disk space). DOTA's own annotation coarseness (only `small-vehicle`/`large-vehicle` are shareable with VisDrone) is a separate ceiling that more images would not lift. `notebooks/02_full_dataset.ipynb` — the planned full-dataset stage — is an empty notebook.

**(d) xView-based results — done, at validation scale.**

Runs H and I: car/bus/truck vs VisDrone (xView first 5,000 of 20,881 `train` images) and building vs UAVid (xView first 500 images). Single seed (42), 300 crops/class, both encoders frozen (CLIP ViT-B/32, RemoteCLIP ViT-L/14). `van` measured (25 xView vs 300 VisDrone samples in the cache) but excluded from the drift JSON by `MIN_SAMPLES_FOR_DRIFT = 30`. The fine-tuned arm has **not** been evaluated on xView or UAVid. No full-scale xView run exists.

**Other “not yet done” items confirmed absent:** full-scale VisDrone embedding runs; Phase-4 drift-vs-mAP correlation; Phase-5 alignment module; Phase-6 mAP / cross-altitude-gap evaluation; confidence intervals or bootstrap of any drift number; any classification/zero-shot accuracy on the crops.

### 5.4 Headline results in one place [values from the JSONs; percentage columns DERIVED]

Δ% = 100·(CLIP − arm)/CLIP (positive = lower drift than CLIP). `drift_ratio` is shown for completeness; a positive Δ there means a smaller ratio.

**DOTA ↔ VisDrone-held-out — 3 arms** (`drift_comparison_3way_heldout.json`; 300 crops/class/domain; eval seed 42; split seed 1234):

| Class | Metric | CLIP ViT-B/32 | RemoteCLIP ViT-L/14 | Δ% | Fine-tuned CLIP | Δ% |
|---|---|---:|---:|---:|---:|---:|
| small-vehicle | L2 | 0.2460 | 0.1892 | +23.1% | 0.1585 | +35.6% |
| small-vehicle | cosine | 0.0346 | 0.0199 | +42.5% | 0.0129 | +62.7% |
| small-vehicle | drift_ratio | 0.6498 | 0.5627 | +13.4% | 0.8165 | -25.7% |
| large-vehicle | L2 | 0.2596 | 0.2276 | +12.3% | 0.0955 | +63.2% |
| large-vehicle | cosine | 0.0406 | 0.0300 | +26.1% | 0.0048 | +88.2% |
| large-vehicle | drift_ratio | 0.6235 | 0.6252 | -0.3% | 0.4132 | +33.7% |

**xView ↔ VisDrone / UAVid — 2 arms** (`drift_comparison_4class.json`; 300 crops/class; seed 42; building = xView vs UAVid):

| Class | Metric | CLIP ViT-B/32 | RemoteCLIP ViT-L/14 | Δ% |
|---|---|---:|---:|---:|
| car | L2 | 0.3564 | 0.3024 | +15.2% |
| car | cosine | 0.0718 | 0.0503 | +29.9% |
| car | drift_ratio | 0.9931 | 0.9719 | +2.1% |
| bus | L2 | 0.3256 | 0.2802 | +13.9% |
| bus | cosine | 0.0603 | 0.0434 | +28.0% |
| bus | drift_ratio | 0.8443 | 0.8491 | -0.6% |
| truck | L2 | 0.2837 | 0.2366 | +16.6% |
| truck | cosine | 0.0451 | 0.0303 | +32.8% |
| truck | drift_ratio | 0.7550 | 0.7500 | +0.7% |
| building | L2 | 0.5417 | 0.3783 | +30.2% |
| building | cosine | 0.1822 | 0.0847 | +53.5% |
| building | drift_ratio | 1.2219 | 0.9809 | +19.7% |

Range of L2/cosine reductions over all class×arm×metric combinations **[DERIVED]**: **12.3% – 88.2%** (the mid-semester deck states “12–88%”). RemoteCLIP-only L2 reductions range 12.3% (large-vehicle) – 30.2% (building) (the deck states “12–30%”). `drift_ratio` nuance recorded in `PROGRESS_SUMMARY.md` item 5d and visible above: the fine-tuned arm's small-vehicle ratio (0.8165) is *higher* than RemoteCLIP's (0.5627) and CLIP's (0.6498) although its L2 is lower, because fine-tuning also shrinks the intra-domain radii (denominator).

**First-pass numbers that were superseded** (`drift_comparison_3way.json`, run D/E; bicycle-in-small-vehicle mapping and, for the fine-tuned arm, train/eval leakage):

| Class | CLIP (first) | RemoteCLIP (first) | Fine-tuned (first, leaky) | CLIP (corrected) | RemoteCLIP (corrected) | Fine-tuned (held-out) |
|---|---:|---:|---:|---:|---:|---:|
| small-vehicle | 0.2366 | 0.1720 | 0.1412 | 0.2460 | 0.1892 | 0.1585 |
| large-vehicle | 0.2615 | 0.2260 | 0.0964 | 0.2596 | 0.2276 | 0.0955 |

### 5.5 Every results JSON — full contents, producer and configuration

#### `results/remoteclip_vs_clip/drift_comparison.json`

- mtime 2026-08-20 22:06:54, 1,110 bytes
- Produced by / configuration: Run **G** (`run_remoteclip_vs_clip.py`), overwriting run D's file of the same name. Config keys stored: `max_crops_per_class 300`, `random_seed 42`, `holdout_fraction 0.2`, `split_seed 1234`. CLIP ViT-B/32 (512-d) and RemoteCLIP ViT-L/14 (768-d), DOTA (all 469 images) vs VisDrone **held-out** pool; classes `small-vehicle`, `large-vehicle`. The only file that stores `sat_intra_var`/`uav_intra_var`.

```json
{
  "clip_vit_b32": {
    "small-vehicle": {
      "l2_drift": 0.246,
      "cosine_drift": 0.0346,
      "drift_ratio": 0.6498,
      "sat_samples": 300,
      "uav_samples": 300,
      "sat_intra_var": 0.3286,
      "uav_intra_var": 0.4286
    },
    "large-vehicle": {
      "l2_drift": 0.2596,
      "cosine_drift": 0.0406,
      "drift_ratio": 0.6235,
      "sat_samples": 300,
      "uav_samples": 300,
      "sat_intra_var": 0.3855,
      "uav_intra_var": 0.4471
    }
  },
  "remoteclip_vit_l14": {
    "small-vehicle": {
      "l2_drift": 0.1892,
      "cosine_drift": 0.0199,
      "drift_ratio": 0.5627,
      "sat_samples": 300,
      "uav_samples": 300,
      "sat_intra_var": 0.3019,
      "uav_intra_var": 0.3707
    },
    "large-vehicle": {
      "l2_drift": 0.2276,
      "cosine_drift": 0.03,
      "drift_ratio": 0.6252,
      "sat_samples": 300,
      "uav_samples": 300,
      "sat_intra_var": 0.3444,
      "uav_intra_var": 0.3837
    }
  },
  "max_crops_per_class": 300,
  "random_seed": 42,
  "holdout_fraction": 0.2,
  "split_seed": 1234
}
```

#### `results/remoteclip_vs_clip/drift_comparison_3way.json`

- mtime 2026-08-19 16:47:07, 1,604 bytes
- Produced by / configuration: Run **E** (first, leaky 3-way). `clip_vit_b32` and `remoteclip_vit_l14` here are the **pre-fix** (bicycle-included) numbers (they equal the table in `PROJECT_CONTEXT.md` §7b), `finetuned_clip_visdrone` was trained on all VisDrone crops and evaluated on a sample from the same pool. Config keys: `eval_max_crops_per_class 300`, `eval_seed 42`, `train_max_crops_per_class null`, `train_num_epochs 3`, `train_batch_size 8`, `train_lr 1e-05`, `prompt_template "a drone photo of a {}"`. **Superseded — keep only for audit.**

```json
{
  "clip_vit_b32": {
    "small-vehicle": {
      "l2_drift": 0.2366,
      "cosine_drift": 0.0318,
      "drift_ratio": 0.6175,
      "sat_samples": 300,
      "uav_samples": 300,
      "sat_intra_var": 0.3286,
      "uav_intra_var": 0.4376
    },
    "large-vehicle": {
      "l2_drift": 0.2615,
      "cosine_drift": 0.0413,
      "drift_ratio": 0.6226,
      "sat_samples": 300,
      "uav_samples": 300,
      "sat_intra_var": 0.3855,
      "uav_intra_var": 0.4544
    }
  },
  "remoteclip_vit_l14": {
    "small-vehicle": {
      "l2_drift": 0.172,
      "cosine_drift": 0.0164,
      "drift_ratio": 0.5144,
      "sat_samples": 300,
      "uav_samples": 300,
      "sat_intra_var": 0.3019,
      "uav_intra_var": 0.367
    },
    "large-vehicle": {
      "l2_drift": 0.226,
      "cosine_drift": 0.0295,
      "drift_ratio": 0.6222,
      "sat_samples": 300,
      "uav_samples": 300,
      "sat_intra_var": 0.3444,
      "uav_intra_var": 0.3821
    }
  },
  "finetuned_clip_visdrone": {
    "small-vehicle": {
      "l2_drift": 0.1412,
      "cosine_drift": 0.0102,
      "drift_ratio": 0.7486,
      "sat_samples": 300,
      "uav_samples": 300
    },
    "large-vehicle": {
      "l2_drift": 0.0964,
      "cosine_drift": 0.0049,
      "drift_ratio": 0.4467,
      "sat_samples": 300,
      "uav_samples": 300
    }
  },
  "eval_max_crops_per_class": 300,
  "eval_seed": 42,
  "train_max_crops_per_class": null,
  "train_num_epochs": 3,
  "train_batch_size": 8,
  "train_lr": 1e-05,
  "prompt_template": "a drone photo of a {}"
}
```

#### `results/remoteclip_vs_clip/drift_comparison_3way_heldout.json`

- mtime 2026-08-20 21:36:59, 1,621 bytes
- Produced by / configuration: Run **F** (`notebooks/04_finetune_visdrone_clip.ipynb` cell 20). The headline DOTA↔VisDrone result. Config keys: `eval_max_crops_per_class 300`, `eval_seed 42`, `holdout_fraction 0.2`, `split_seed 1234`, `train_num_epochs 3`, `train_batch_size 8`, `train_lr 1e-05`, `prompt_template`, `notes`.

```json
{
  "clip_vit_b32": {
    "small-vehicle": {
      "l2_drift": 0.246,
      "cosine_drift": 0.0346,
      "drift_ratio": 0.6498,
      "sat_samples": 300,
      "uav_samples": 300
    },
    "large-vehicle": {
      "l2_drift": 0.2596,
      "cosine_drift": 0.0406,
      "drift_ratio": 0.6235,
      "sat_samples": 300,
      "uav_samples": 300
    }
  },
  "remoteclip_vit_l14": {
    "small-vehicle": {
      "l2_drift": 0.1892,
      "cosine_drift": 0.0199,
      "drift_ratio": 0.5627,
      "sat_samples": 300,
      "uav_samples": 300
    },
    "large-vehicle": {
      "l2_drift": 0.2276,
      "cosine_drift": 0.03,
      "drift_ratio": 0.6252,
      "sat_samples": 300,
      "uav_samples": 300
    }
  },
  "finetuned_clip_visdrone_heldout": {
    "small-vehicle": {
      "l2_drift": 0.1585,
      "cosine_drift": 0.0129,
      "drift_ratio": 0.8165,
      "sat_samples": 300,
      "uav_samples": 300
    },
    "large-vehicle": {
      "l2_drift": 0.0955,
      "cosine_drift": 0.0048,
      "drift_ratio": 0.4132,
      "sat_samples": 300,
      "uav_samples": 300
    }
  },
  "eval_max_crops_per_class": 300,
  "eval_seed": 42,
  "holdout_fraction": 0.2,
  "split_seed": 1234,
  "train_num_epochs": 3,
  "train_batch_size": 8,
  "train_lr": 1e-05,
  "prompt_template": "a drone photo of a {}",
  "notes": "Fixes vs first run: (1) bicycle removed from small-vehicle mapping, (2) VisDrone split 80/20 train/held-out by image before any crop extraction, fine-tuning only sees the 80% pool, all three arms evaluated only on the 20% held-out pool."
}
```

#### `results/xview_vs_visdrone/drift_comparison.json`

- mtime 2026-08-20 22:23:47, 1,411 bytes
- Produced by / configuration: Run **H** (`run_xview_vs_visdrone.py`). Config keys: `max_xview_images 5000`, `max_crops_per_class 300`, `min_samples_for_drift 30`, `random_seed 42`. car/bus/truck only (`van` gated out).

```json
{
  "clip_vit_b32": {
    "car": {
      "l2_drift": 0.3564,
      "cosine_drift": 0.0718,
      "drift_ratio": 0.9931,
      "xview_samples": 300,
      "visdrone_samples": 300
    },
    "bus": {
      "l2_drift": 0.3256,
      "cosine_drift": 0.0603,
      "drift_ratio": 0.8443,
      "xview_samples": 300,
      "visdrone_samples": 300
    },
    "truck": {
      "l2_drift": 0.2837,
      "cosine_drift": 0.0451,
      "drift_ratio": 0.755,
      "xview_samples": 300,
      "visdrone_samples": 300
    }
  },
  "remoteclip_vit_l14": {
    "car": {
      "l2_drift": 0.3024,
      "cosine_drift": 0.0503,
      "drift_ratio": 0.9719,
      "xview_samples": 300,
      "visdrone_samples": 300
    },
    "bus": {
      "l2_drift": 0.2802,
      "cosine_drift": 0.0434,
      "drift_ratio": 0.8491,
      "xview_samples": 300,
      "visdrone_samples": 300
    },
    "truck": {
      "l2_drift": 0.2366,
      "cosine_drift": 0.0303,
      "drift_ratio": 0.75,
      "xview_samples": 300,
      "visdrone_samples": 300
    }
  },
  "max_xview_images": 5000,
  "max_crops_per_class": 300,
  "min_samples_for_drift": 30,
  "random_seed": 42,
  "notes": "xView has no dedicated van category (Pickup Truck is the closest match and is rare); van is measured but likely excluded here for insufficient samples. car/bus/truck are the real result of this pass."
}
```

#### `results/xview_vs_visdrone/drift_comparison_4class.json`

- mtime 2026-08-20 23:04:29, 2,088 bytes
- Produced by / configuration: Run **I** (`run_uavid_building.py`): run H's JSON with `building` merged in (xView first 500 images vs UAVid train+val), `notes` string extended. The `visdrone_samples` key of `building` holds the UAVid count.

```json
{
  "clip_vit_b32": {
    "car": {
      "l2_drift": 0.3564,
      "cosine_drift": 0.0718,
      "drift_ratio": 0.9931,
      "xview_samples": 300,
      "visdrone_samples": 300
    },
    "bus": {
      "l2_drift": 0.3256,
      "cosine_drift": 0.0603,
      "drift_ratio": 0.8443,
      "xview_samples": 300,
      "visdrone_samples": 300
    },
    "truck": {
      "l2_drift": 0.2837,
      "cosine_drift": 0.0451,
      "drift_ratio": 0.755,
      "xview_samples": 300,
      "visdrone_samples": 300
    },
    "building": {
      "l2_drift": 0.5417,
      "cosine_drift": 0.1822,
      "drift_ratio": 1.2219,
      "xview_samples": 300,
      "visdrone_samples": 300
    }
  },
  "remoteclip_vit_l14": {
    "car": {
      "l2_drift": 0.3024,
      "cosine_drift": 0.0503,
      "drift_ratio": 0.9719,
      "xview_samples": 300,
      "visdrone_samples": 300
    },
    "bus": {
      "l2_drift": 0.2802,
      "cosine_drift": 0.0434,
      "drift_ratio": 0.8491,
      "xview_samples": 300,
      "visdrone_samples": 300
    },
    "truck": {
      "l2_drift": 0.2366,
      "cosine_drift": 0.0303,
      "drift_ratio": 0.75,
      "xview_samples": 300,
      "visdrone_samples": 300
    },
    "building": {
      "l2_drift": 0.3783,
      "cosine_drift": 0.0847,
      "drift_ratio": 0.9809,
      "xview_samples": 300,
      "visdrone_samples": 300
    }
  },
  "max_xview_images": 5000,
  "max_crops_per_class": 300,
  "min_samples_for_drift": 30,
  "random_seed": 42,
  "notes": "xView has no dedicated van category (Pickup Truck is the closest match and is rare); van is measured but likely excluded here for insufficient samples. car/bus/truck are the real result of this pass. | building added 2026-08-20: xView (satellite, streamed) vs UAVid (UAV, local Kaggle download of the official train+val split, VisDrone has no building class so UAVid substitutes for the UAV side on this one class only). Crops derived from UAVid's pixel-level masks via connected-component analysis, not native bounding boxes."
}
```

#### `results/phase1_local/drift_scores_local.json`

- mtime 2026-08-13 19:03:54, 229 bytes
- Produced by / configuration: Run **B** (`notebooks/01_phase1_reproduction.ipynb` cell 23). Whole-image, zero-shot-predicted labels, L2 only, 300 images/domain. **Not ground-truth labels.**

```json
{
  "infrastructure": 0.6334172487258911,
  "aircraft": 0.5363435745239258,
  "sports-field": 0.6164792776107788,
  "industrial": 0.45891204476356506,
  "vehicle": 0.5640590786933899,
  "recreational": 0.8188402056694031
}
```

#### `results/category_yardstick/yardstick_table.json`

- mtime 2026-08-22 11:47:16, 2,787 bytes
- Produced by / configuration: Run **J** (`make_category_yardstick_plots.py`) from the cached embeddings of runs G, H, I (so the DOTA table uses the *held-out* VisDrone embeddings). Three comparisons × two encoders.

```json
{
  "dota_vs_visdrone": {
    "CLIP ViT-B/32": {
      "small-vehicle": {
        "cross_altitude_cosine": 0.0346,
        "cross_category_cosine": 0.0199,
        "ratio": 1.7361
      },
      "large-vehicle": {
        "cross_altitude_cosine": 0.0406,
        "cross_category_cosine": 0.0199,
        "ratio": 2.0404
      }
    },
    "RemoteCLIP ViT-L/14": {
      "small-vehicle": {
        "cross_altitude_cosine": 0.0199,
        "cross_category_cosine": 0.0177,
        "ratio": 1.1249
      },
      "large-vehicle": {
        "cross_altitude_cosine": 0.03,
        "cross_category_cosine": 0.0177,
        "ratio": 1.6916
      }
    }
  },
  "xview_vs_visdrone": {
    "CLIP ViT-B/32": {
      "car": {
        "cross_altitude_cosine": 0.0718,
        "cross_category_cosine": 0.0134,
        "ratio": 5.3446
      },
      "bus": {
        "cross_altitude_cosine": 0.0603,
        "cross_category_cosine": 0.0104,
        "ratio": 5.7743
      },
      "truck": {
        "cross_altitude_cosine": 0.0451,
        "cross_category_cosine": 0.0087,
        "ratio": 5.1582
      }
    },
    "RemoteCLIP ViT-L/14": {
      "car": {
        "cross_altitude_cosine": 0.0503,
        "cross_category_cosine": 0.0121,
        "ratio": 4.1473
      },
      "bus": {
        "cross_altitude_cosine": 0.0434,
        "cross_category_cosine": 0.0094,
        "ratio": 4.624
      },
      "truck": {
        "cross_altitude_cosine": 0.0303,
        "cross_category_cosine": 0.0082,
        "ratio": 3.6866
      }
    }
  },
  "xview_visdrone_uavid": {
    "CLIP ViT-B/32": {
      "car": {
        "cross_altitude_cosine": 0.0718,
        "cross_category_cosine": 0.0368,
        "ratio": 1.9505
      },
      "bus": {
        "cross_altitude_cosine": 0.0603,
        "cross_category_cosine": 0.0334,
        "ratio": 1.8065
      },
      "truck": {
        "cross_altitude_cosine": 0.0451,
        "cross_category_cosine": 0.0301,
        "ratio": 1.4966
      },
      "building": {
        "cross_altitude_cosine": 0.1822,
        "cross_category_cosine": 0.0786,
        "ratio": 2.3189
      }
    },
    "RemoteCLIP ViT-L/14": {
      "car": {
        "cross_altitude_cosine": 0.0503,
        "cross_category_cosine": 0.0241,
        "ratio": 2.0908
      },
      "bus": {
        "cross_altitude_cosine": 0.0434,
        "cross_category_cosine": 0.0188,
        "ratio": 2.3023
      },
      "truck": {
        "cross_altitude_cosine": 0.0303,
        "cross_category_cosine": 0.0168,
        "ratio": 1.8081
      },
      "building": {
        "cross_altitude_cosine": 0.0847,
        "cross_category_cosine": 0.0399,
        "ratio": 2.1233
      }
    }
  }
}
```

#### Printed (not saved) results

**Phase-1 Lightning whole-image drift** (`results/phase1_lightning/Capstone_v1.ipynb` cell 84 output, verbatim; 5 classes, unsorted first-300 images, predicted labels):

```text
sports-field: Drift Score = 0.5320
infrastructure: Drift Score = 0.5714
aircraft: Drift Score = 0.6093
vehicle: Drift Score = 0.5907
industrial: Drift Score = 0.7169
```

**Phase-1 local vs historical** (`notebooks/01_phase1_reproduction.ipynb` cell 25):

```python
historical_drift_scores = {
    "sports-field": 0.5320,
    "infrastructure": 0.5714,
    "aircraft": 0.6093,
    "vehicle": 0.5907,
    "industrial": 0.7169,
}

print(f"{'class':<15}{'historical':>12}{'local':>12}{'delta':>12}")
for cls in sorted(set(historical_drift_scores) | set(drift_scores)):
    h = historical_drift_scores.get(cls)
    l = drift_scores.get(cls)
    if h is None or l is None:
        print(f"{cls:<15}{'n/a' if h is None else h:>12}{'n/a' if l is None else round(l,4):>12}{'':>12}")
    else:
        print(f"{cls:<15}{h:>12.4f}{l:>12.4f}{l-h:>12.4f}")
```

```text
class            historical       local       delta
aircraft             0.6093      0.5363     -0.0730
industrial           0.7169      0.4589     -0.2580
infrastructure       0.5714      0.6334      0.0620
recreational            n/a      0.8188            
sports-field         0.5320      0.6165      0.0845
vehicle              0.5907      0.5641     -0.0266
```

Local-vs-historical differences reach −0.258 (`industrial`); the local run adds a 6th class (`recreational`, 1 DOTA whole image). Two known reasons the subsets differ (§10): sorted vs unsorted `listdir`, and fp32 HF vs OpenAI-`clip` code paths. `PROJECT_CONTEXT.md` §22 calls this “reasonably consistent”; the per-class values are noisy.

**Historical Lightning large-scale comparison** (read from `results/phase1_lightning/drift_scores.png` and quoted in `PROJECT_CONTEXT.md` §7a): large-vehicle L2 = 0.20 / cosine = 0.36; small-vehicle L2 = 0.125 / cosine = 0.27 — produced with two independently fit PCA-256 projections; **not valid**.

**Fine-tuning epoch losses** — see §3.4.6.

### 5.6 Cross-check of the numbers quoted in the repo's documents against the JSONs [DERIVED]

| Where | Statement | Check |
|---|---|---|
| `PROJECT_CONTEXT.md` §7c / `PROGRESS_SUMMARY.md` / `PRESENTATION_SCRIPT.md` tables | small-vehicle 0.246 / 0.189 / 0.159; large-vehicle 0.260 / 0.228 / 0.096 | match `drift_comparison_3way_heldout.json` after rounding ✔ |
| same | RemoteCLIP −23% / −12%; fine-tuned −36% / −63% | 23.1%, 12.3%; 35.6%, 63.2% ✔ |
| `PROJECT_CONTEXT.md` §7d | car 0.356/0.302 (15%), bus 0.326/0.280 (14%), truck 0.284/0.237 (17%), building 0.542/0.378 (30%) | match `drift_comparison_4class.json`; 15.2%, 13.9%, 16.6%, 30.2% ✔ |
| `PROJECT_CONTEXT.md` §7b table | CLIP 0.2366/0.2615; RemoteCLIP 0.1720/0.2260 (−27%/−14%) | match `drift_comparison_3way.json` (first-pass, pre-fix). Marked superseded by §7c but still in the file ✔ (see §10) |
| `PROJECT_CONTEXT.md` §7c | large-vehicle fine-tuned 0.0964 → 0.0955; small 0.1412 → 0.1585 | match the two JSONs ✔ |
| `notebooks/03_remoteclip.ipynb` cell 6 (markdown) | “small-vehicle: ~27% lower L2, ~48% lower cosine; large-vehicle ~14% / ~29%; drift_ratio 0.51 vs 0.62” | these are the **pre-fix** numbers; the notebook's own executed cell 5 output shows 23.1% / 12.3% (post-fix). The narrative cell is stale ✘ |
| `PROJECT_CONTEXT.md` §4 | UAVid `test_data/Images` = 30 images | disk has **150** ✘ |

### 5.7 Cached embedding arrays (all git-ignored) [DERIVED]

Complete inventory of `embeddings/**/*.npy` (44 arrays). `norm` = min/mean/max L2 norm over the first 2,000 rows. Label arrays list per-class counts.

| Array | Shape | dtype | Bytes | Norms / label counts |
|---|---|---|---|---|
| `embeddings/phase1_lightning/dota_feats.npy` | 893×512 | float16 | 914,560 | norm 0.99949/1.0/1.0005 |
| `embeddings/phase1_lightning/dota_labels.npy` | 893 | <U13 | 46,564 | ship: 773, small-vehicle: 36, plane: 35, harbor: 27, large-vehicle: 12, tennis-court: 7, helicopter: 2, roundabout: 1 |
| `embeddings/phase1_lightning/finetuned_visdrone_features.npy` | 217153×512 | float32 | 444,729,472 | norm 1.0/1.0/1.0 |
| `embeddings/phase1_lightning/finetuned_visdrone_labels.npy` | 217153 | <U13 | 11,292,084 | small-vehicle: 116,342, person: 39,848, large-vehicle: 37,064, motorcycle: 23,899 |
| `embeddings/phase1_lightning/remoteclip_dota_features.npy` | 28767×768 | float32 | 88,372,352 | norm 1.0/1.0/1.0 |
| `embeddings/phase1_lightning/remoteclip_dota_labels.npy` | 28767 | <U18 | 2,071,352 | ship: 12,878, small-vehicle: 6,537, large-vehicle: 3,033, harbor: 2,252, plane: 1,538, tennis-court: 1,133, swimming-pool: 1,016, basketball-court: 111, baseball-diamond: 73, storage-tank: 57, soccer-ball-field: 52, roundabout: 48, ground-track-field: 19, helicopter: 18, bridge: 2 |
| `embeddings/phase1_lightning/vis_feats.npy` | 312×512 | float16 | 319,616 | norm 0.9995/0.99999/1.0005 |
| `embeddings/phase1_lightning/vis_labels.npy` | 312 | <U15 | 18,848 | car: 129, pedestrian: 96, motor: 38, bicycle: 26, person: 14, van: 5, awning-tricycle: 2, bus: 2 |
| `embeddings/phase1_local/dota_feats.npy` | 591×512 | float32 | 1,210,496 | norm 1.0/1.0/1.0 |
| `embeddings/phase1_local/dota_labels.npy` | 591 | <U13 | 30,860 | large-vehicle: 419, small-vehicle: 117, plane: 21, ship: 17, harbor: 17 |
| `embeddings/phase1_local/dota_whole_image_feats.npy` | 300×512 | float32 | 614,528 | norm 1.0/1.0/1.0 |
| `embeddings/phase1_local/dota_whole_image_predicted_labels.npy` | 300 | <U14 | 16,928 | sports-field: 122, infrastructure: 61, aircraft: 50, watercraft: 37, vehicle: 17, industrial: 12, recreational: 1 |
| `embeddings/phase1_local/vis_feats.npy` | 329×512 | float32 | 673,920 | norm 1.0/1.0/1.0 |
| `embeddings/phase1_local/vis_labels.npy` | 329 | <U15 | 19,868 | truck: 99, car: 75, pedestrian: 45, motor: 43, van: 27, person: 21, unknown: 6, bicycle: 5, awning-tricycle: 5, tricycle: 3 |
| `embeddings/phase1_local/visdrone_whole_image_feats.npy` | 300×512 | float32 | 614,528 | norm 1.0/1.0/1.0 |
| `embeddings/phase1_local/visdrone_whole_image_predicted_labels.npy` | 300 | <U14 | 16,928 | infrastructure: 146, sports-field: 100, vehicle: 36, recreational: 9, industrial: 6, aircraft: 3 |
| `embeddings/remoteclip_vs_clip/dota_clip_feats.npy` | 600×512 | float32 | 1,228,928 | norm 1.0/1.0/1.0 |
| `embeddings/remoteclip_vs_clip/dota_clip_labels.npy` | 600 | <U13 | 31,328 | small-vehicle: 300, large-vehicle: 300 |
| `embeddings/remoteclip_vs_clip/dota_finetuned_clip_feats.npy` | 600×512 | float32 | 1,228,928 | norm 1.0/1.0/1.0 |
| `embeddings/remoteclip_vs_clip/dota_finetuned_clip_heldout_feats.npy` | 600×512 | float32 | 1,228,928 | norm 1.0/1.0/1.0 |
| `embeddings/remoteclip_vs_clip/dota_finetuned_clip_heldout_labels.npy` | 600 | <U13 | 31,328 | small-vehicle: 300, large-vehicle: 300 |
| `embeddings/remoteclip_vs_clip/dota_finetuned_clip_labels.npy` | 600 | <U13 | 31,328 | small-vehicle: 300, large-vehicle: 300 |
| `embeddings/remoteclip_vs_clip/dota_remoteclip_feats.npy` | 600×768 | float32 | 1,843,328 | norm 1.0/1.0/1.0 |
| `embeddings/remoteclip_vs_clip/dota_remoteclip_labels.npy` | 600 | <U13 | 31,328 | small-vehicle: 300, large-vehicle: 300 |
| `embeddings/remoteclip_vs_clip/visdrone_clip_feats.npy` | 600×512 | float32 | 1,228,928 | norm 1.0/1.0/1.0 |
| `embeddings/remoteclip_vs_clip/visdrone_clip_labels.npy` | 600 | <U13 | 31,328 | small-vehicle: 300, large-vehicle: 300 |
| `embeddings/remoteclip_vs_clip/visdrone_finetuned_clip_feats.npy` | 600×512 | float32 | 1,228,928 | norm 1.0/1.0/1.0 |
| `embeddings/remoteclip_vs_clip/visdrone_finetuned_clip_heldout_feats.npy` | 600×512 | float32 | 1,228,928 | norm 1.0/1.0/1.0 |
| `embeddings/remoteclip_vs_clip/visdrone_finetuned_clip_heldout_labels.npy` | 600 | <U13 | 31,328 | small-vehicle: 300, large-vehicle: 300 |
| `embeddings/remoteclip_vs_clip/visdrone_finetuned_clip_labels.npy` | 600 | <U13 | 31,328 | small-vehicle: 300, large-vehicle: 300 |
| `embeddings/remoteclip_vs_clip/visdrone_remoteclip_feats.npy` | 600×768 | float32 | 1,843,328 | norm 1.0/1.0/1.0 |
| `embeddings/remoteclip_vs_clip/visdrone_remoteclip_labels.npy` | 600 | <U13 | 31,328 | small-vehicle: 300, large-vehicle: 300 |
| `embeddings/xview_vs_visdrone/uavid_building_clip_feats.npy` | 300×512 | float32 | 614,528 | norm 1.0/1.0/1.0 |
| `embeddings/xview_vs_visdrone/uavid_building_remoteclip_feats.npy` | 300×768 | float32 | 921,728 | norm 1.0/1.0/1.0 |
| `embeddings/xview_vs_visdrone/visdrone_clip_feats.npy` | 1200×512 | float32 | 2,457,728 | norm 1.0/1.0/1.0 |
| `embeddings/xview_vs_visdrone/visdrone_clip_labels.npy` | 1200 | <U5 | 24,128 | truck: 300, car: 300, van: 300, bus: 300 |
| `embeddings/xview_vs_visdrone/visdrone_remoteclip_feats.npy` | 1200×768 | float32 | 3,686,528 | norm 1.0/1.0/1.0 |
| `embeddings/xview_vs_visdrone/visdrone_remoteclip_labels.npy` | 1200 | <U5 | 24,128 | truck: 300, car: 300, van: 300, bus: 300 |
| `embeddings/xview_vs_visdrone/xview_building_clip_feats.npy` | 300×512 | float32 | 614,528 | norm 1.0/1.0/1.0 |
| `embeddings/xview_vs_visdrone/xview_building_remoteclip_feats.npy` | 300×768 | float32 | 921,728 | norm 1.0/1.0/1.0 |
| `embeddings/xview_vs_visdrone/xview_clip_feats.npy` | 925×512 | float32 | 1,894,528 | norm 1.0/1.0/1.0 |
| `embeddings/xview_vs_visdrone/xview_clip_labels.npy` | 925 | <U5 | 18,628 | car: 300, truck: 300, bus: 300, van: 25 |
| `embeddings/xview_vs_visdrone/xview_remoteclip_feats.npy` | 925×768 | float32 | 2,841,728 | norm 1.0/1.0/1.0 |
| `embeddings/xview_vs_visdrone/xview_remoteclip_labels.npy` | 925 | <U5 | 18,628 | car: 300, truck: 300, bus: 300, van: 25 |

Notes: the two float16 arrays (`phase1_lightning/dota_feats.npy`, `vis_feats.npy`) are the output of the OpenAI `clip` package (run A′) — **[ANALYSIS]** float16 is what `clip.load(...)` produces on a CUDA device; all local arrays are float32. `embeddings/remoteclip_vs_clip/{dota,visdrone}_finetuned_clip_{feats,labels}.npy` (no `heldout`) are the **first-run (leaky)** arrays — verified: they reproduce `drift_comparison_3way.json`'s fine-tuned entries exactly. The pre-fix CLIP/RemoteCLIP arrays of run D were overwritten by run G, so the pre-fix CLIP/RemoteCLIP numbers can no longer be regenerated from the cache.

---

## 6. Detection pipeline (teammate's SAM-based work)

**State: absent from this repository — not even scaffolding.** Facts:

- **No code.** A search of every `.py` file, every notebook code cell and `configs/` for `segment anything | SAM | GroundingDINO | YOLO | Faster R-CNN | Detectron | pycocotools | mAP | average precision | IoU | NMS | COCOeval | OWL-ViT` finds **no code hits** (only prose in `.md` files, listed below).

```text
(no matches in any .py file or notebook code cell)
```

Prose mentions in the four markdown files (complete list of matching lines for `SAM|GroundingDINO|mAP|detection`):

```text
README.md:1: # Semantic Drift in Open-Vocabulary Object Detection Across Altitudes
README.md:12: Open-vocabulary object detectors (CLIP, GroundingDINO, etc.) assume the relationship between a text concept (e.g. "vehicle") and its visual representation stays consistent across domains. In aerial imagery it doesn't: satellite im
README.md:31: **Not yet started:** full-dataset DOTA (parts 2–3)/VisDrone/xView scale-up, the altitude-conditioned alignment adapter, final evaluation metrics (mAP, cross-altitude gap), SAM-based detection integration with a teammate's work.
README.md:129: 5. Implement the altitude-conditioned alignment adapter (Phase 5) and evaluate mAP / cross-altitude gap / drift reduction (Phase 6).
README.md:130: 6. Integrate with the teammate's SAM-based detection pipeline (real detection, not just classification of pre-cropped regions) to test whether drift predicts detection failures.
PROJECT_CONTEXT.md:63: Object detection annotations / bounding boxes
PROJECT_CONTEXT.md:922: - [ ] Final evaluation metrics (mAP, cross-altitude gap, semantic drift reduction)
PROJECT_CONTEXT.md:925: - [ ] SAM-based detection integration with teammate's work (see PROGRESS_SUMMARY.md item 9)
PROGRESS_SUMMARY.md:65: A teammate is combining all three datasets and applying SAM (an AI model that finds object regions in images) to enable real object *detection*, not just classification of pre-cropped images. We planned out how their detection wor
PROGRESS_SUMMARY.md:94: - Combine with the teammate's SAM-based detection work to get real detection performance numbers (not just embedding-distance numbers) and see if drift actually predicts detection failures
PRESENTATION_SCRIPT.md:5: detection-integration detail are the easiest cuts; section 6's building result is worth keeping,
PRESENTATION_SCRIPT.md:171: ## 7. The bigger picture: connecting to real object detection (~1 min)
PRESENTATION_SCRIPT.md:178: applying **SAM**, a model that finds object regions in an image automatically, without needing
PRESENTATION_SCRIPT.md:179: labels. The plan is to feed SAM's detected regions into our fingerprinting and drift-measurement
PRESENTATION_SCRIPT.md:180: pipeline — so SAM finds *where* the objects are, and our system identifies *what* they are and
PRESENTATION_SCRIPT.md:189: Stepping back: open-vocabulary object detection lets AI systems recognize objects using plain
PRESENTATION_SCRIPT.md:204: the fourth vehicle category we had to drop for now, and integrating with the SAM-based detection
PRESENTATION_SCRIPT.md:205: pipeline so we can measure not just drift, but whether drift actually predicts real detection
```

- **No detector packages** in the environment: `segment_anything`, `ultralytics`, `pycocotools`, `opencv-python` are not installed in `hf`; (the Lightning notebook installed `opencv-python-headless` only to draw boxes, `Capstone_v1.ipynb` cells 26–30). There is no SAM checkpoint in the repo folder (`models/` holds only RemoteCLIP and the fine-tuned CLIP files).

- **No detection accuracy / mAP numbers of any kind**, per category or overall. The only quantities produced are embedding-space distances (§4). There is not even a zero-shot *classification accuracy* on the ground-truth crops.

- **Has anything correlating drift with detection-accuracy degradation been run? No.** No code, notebook, JSON or figure computes or plots a drift-versus-accuracy relationship. It appears only as planned work: proposal Phase 4 “Study the correlation between drift and mAP drop” (**[EXTERNAL]**), `README.md` next-step 6 “Integrate with the teammate's SAM-based detection pipeline … to test whether drift predicts detection failures”, `PROGRESS_SUMMARY.md` “What's next”, `PROJECT_CONTEXT.md` §21 (unchecked: “Final evaluation metrics (mAP, cross-altitude gap, semantic drift reduction)”, “SAM-based detection integration with teammate's work”) and the deck's “Next Steps: Phase 4: drift-vs-mAP correlation study … Integrate with teammate's SAM-based detection pipeline”.

- **What the repo says about the teammate's work** (verbatim): `PROGRESS_SUMMARY.md` item 9 — “A teammate is combining all three datasets and applying SAM (an AI model that finds object regions in images) to enable real object *detection*, not just classification of pre-cropped images. We planned out how their detection work and our drift-measurement work fit together into a single, complete open-vocabulary object detection pipeline — SAM finds *where* objects are, our models identify *what* they are and measure how consistent that identification is across altitudes.” `PRESENTATION_SCRIPT.md` §7 — “This integration is in progress, not complete yet.” The teammate is **not named** in the repo (README roles: Anoop Singh — Experimentation & Evaluation; Arnav Gupta — Baseline & Model Implementation; Khushpreet Kaur — Data Analysis & Preprocessing; Siddhant Saxena — Ablation Studies & Vocabulary Formulation; Kabir Oberoi — Semantic Analysis & Visualization). All 14 commits are by one author (`Caebyr`), so no teammate code has ever been committed.

- The proposal's Phase 2 (“Train 3 baseline OVD models … Run evaluations (in-domain and cross-altitude mAP)”) has no counterpart in the repo either; the “baseline” implemented here is frozen CLIP-family *embedding extraction* on ground-truth crops.

---

## 7. Figures / visualizations already generated

Every image file in the repo: 23 result figures under `results/` + 8 files under `diagrams/` (4 draw.io sources + 4 PNG previews). Dimensions are from the PNG headers; “Generated by” = the script/notebook that wrote the file. Four figures were opened and visually checked while preparing this report (the two Lightning PNGs and the two xView PNGs whose generating code is missing); all other descriptions come from the titles/labels in the generating code.

| File | px | Bytes | Generated by | Shows | Caveat |
|---|---|---|---|---|---|
| `results/phase1_local/pca_dota.png` | 1025×789 | 126,405 | `notebooks/01_phase1_reproduction.ipynb` cell 14 | 2-D PCA of 300 DOTA **whole-image** CLIP embeddings coloured by zero-shot-predicted 7-class label (title “PCA of DOTA Image Embeddings (local reproduction)”) | predicted labels |
| `results/phase1_local/pca_combined.png` | 1484×1034 | 206,107 | `notebooks/01_phase1_reproduction.ipynb` cell 19 | joint 2-D PCA of 300 DOTA + 300 VisDrone whole-image embeddings; colour = predicted class, marker = domain (○ satellite, × UAV) | predicted labels; the PROGRESS_SUMMARY guide recommends citing it only as “domain dominates” |
| `results/phase1_local/tsne_drift_local.png` | 1484×1035 | 198,572 | `notebooks/01_phase1_reproduction.ipynb` cell 20 | t-SNE (perplexity 30, lr 200, n_iter 1000, seed 42) of the same 600 embeddings | predicted labels |
| `results/phase1_local/drift_scores_local.png` | 1183×733 | 50,754 | `notebooks/01_phase1_reproduction.ipynb` cell 21 | bar chart of the Phase-1 per-class L2 drift (6 predicted classes) | predicted labels — per-class ranking noisy |
| `results/phase1_local/distance_heatmap_local.png` | 1053×883 | 73,823 | `notebooks/01_phase1_reproduction.ipynb` cell 22 | 7×7 satellite-class × UAV-class centroid L2 heat-map | **guide says skip** — rests on zero-shot-guessed labels |
| `results/phase1_lightning/drift_scores.png` | 1350×750 | 41,607 | `model.ipynb` (missing) | **opened & verified:** grouped bars “Semantic Drift per Class (RemoteCLIP vs Fine-tuned VisDrone CLIP)” — large-vehicle L2≈0.20 / cosine≈0.36; small-vehicle L2≈0.125 / cosine≈0.27 | **invalid comparison (two different models, two independent PCAs) — do not use** |
| `results/phase1_lightning/tsne_drift.png` | 1650×1200 | 182,608 | `model.ipynb` (missing) | **opened & verified:** t-SNE “Satellite vs UAV Embeddings per Class” with classes large-vehicle / small-vehicle / motorcycle, circles = satellite, triangles = UAV | **invalid comparison — do not use** |
| `results/remoteclip_vs_clip/clip_vs_remoteclip_drift.png` | 1783×741 | 54,763 | `notebooks/03_remoteclip.ipynb` cell 8 | two panels (CLIP ViT-B/32, RemoteCLIP ViT-L/14): L2 and cosine drift for small/large-vehicle, 300 crops/class | post-fix numbers (held-out VisDrone) |
| `results/remoteclip_vs_clip/three_way_drift_comparison.png` | 2383×741 | 66,097 | notebook 04 v1 (not preserved) | 3 panels CLIP / RemoteCLIP / fine-tuned — first run | **leaky & pre-bicycle-fix — superseded** |
| `results/remoteclip_vs_clip/three_way_drift_comparison_heldout.png` | 2383×741 | 64,988 | `notebooks/04_finetune_visdrone_clip.ipynb` cell 18 | 3 panels: CLIP ViT-B/32, RemoteCLIP ViT-L/14, CLIP fine-tuned on VisDrone (“held-out eval, no leakage”); L2 (blue) and cosine (orange) on one shared axis; suptitle “Three-way drift comparison -- held-out VisDrone eval, bicycle-fixed vocabulary” | the guide notes cosine bars look tiny on the shared axis |
| `results/xview_vs_visdrone/xview_vs_visdrone_drift.png` | 1783×741 | 59,642 | **generating code not in repo** (mtime 22:27, after run H) | **opened:** 2 panels CLIP / RemoteCLIP; L2 and cosine bars for car, bus, truck; suptitle “xView (satellite) vs VisDrone (UAV) semantic drift -- expanded vocabulary (300 crops/class)” | numbers = `drift_comparison.json` |
| `results/xview_vs_visdrone/four_class_drift_comparison.png` | 1933×741 | 51,182 | **generating code not in repo** (mtime 23:05, after run I) | **opened:** 2 panels CLIP / RemoteCLIP; car, bus, truck, building; suptitle “xView vs VisDrone/UAVid semantic drift -- 4 shared classes (car/bus/truck: VisDrone; building: UAVid)” | numbers = `drift_comparison_4class.json` |
| `results/pca_tsne/pca_dota_vs_visdrone.png` | 1785×1065 | 346,289 | `make_pca_tsne_plots.py` | PCA, 2 panels (left CLIP, right RemoteCLIP), ground-truth labels, DOTA vs VisDrone (2 classes); ○ satellite, × UAV | joint fit per panel; 600 pts/side (no subsampling) |
| `results/pca_tsne/tsne_dota_vs_visdrone.png` | 1785×1065 | 358,616 | `make_pca_tsne_plots.py` | t-SNE version of the above | perplexity 30 |
| `results/pca_tsne/pca_xview_vs_visdrone.png` | 1785×1065 | 441,285 | `make_pca_tsne_plots.py` | PCA, xView vs VisDrone, car/bus/truck (900 pts/side) |  |
| `results/pca_tsne/tsne_xview_vs_visdrone.png` | 1785×1065 | 530,481 | `make_pca_tsne_plots.py` | t-SNE version |  |
| `results/pca_tsne/pca_xview_visdrone_uavid.png` | 1785×1065 | 433,221 | `make_pca_tsne_plots.py` | PCA, 4 classes incl. building (xView vs VisDrone+UAVid) | 1,200 pts/side subsampled **randomly** to 900 (not class-balanced) |
| `results/pca_tsne/tsne_xview_visdrone_uavid.png` | 1785×1065 | 550,669 | `make_pca_tsne_plots.py` | t-SNE, 4 classes; per the commit message the building class forms two separate (satellite vs UAV) clusters | same random 900/side subsample |
| `results/category_yardstick/yardstick_dota_vs_visdrone.png` | 1783×741 | 82,648 | `make_category_yardstick_plots.py` | per model: cross-altitude cosine (same class) vs cross-category cosine (same domain, averaged), ratio printed over bars — DOTA↔VisDrone | values in `yardstick_table.json` |
| `results/category_yardstick/yardstick_xview_vs_visdrone.png` | 1783×741 | 75,530 | `make_category_yardstick_plots.py` | same, xView↔VisDrone (3 classes) |  |
| `results/category_yardstick/yardstick_xview_visdrone_uavid.png` | 1783×741 | 84,503 | `make_category_yardstick_plots.py` | same, 4 classes incl. building |  |
| `results/category_yardstick/drift_ratio_comparison.png` | 2082×815 | 103,924 | `make_drift_ratio_plot.py` | `drift_ratio` per class per encoder (3 arms for DOTA, 2 for xView/UAVid) with a dashed line at 1.0 | reads stored JSON |
| `results/category_yardstick/normalized_drift_reduction.png` | 2082×888 | 119,504 | `make_normalized_drift_plot.py` | L2 and cosine drift as % of the CLIP baseline, RemoteCLIP (and fine-tuned for DOTA) | reads stored JSON; also prints a summary table to stdout |

**Diagrams** (`diagrams/`, draw.io sources + PNG previews, committed in `5dc9698`; text labels extracted from the `.drawio` XML):

| File | Content (labels) |
|---|---|
| `diagrams/fig4_1_architecture.drawio` (+ `_preview.png`, 58,726 B) | Fig 4.1 overall architecture: DOTA (Satellite) & VisDrone (UAV) → Annotation Processing + Common Class Mapping → Ground-Truth Crop Extraction + Controlled Class Sampling → {CLIP ViT-B/32 (Baseline), RemoteCLIP ViT-L/14 (Baseline), CLIP (UAV Fine-Tuned)} → Embedding Generation + Normalization → Class-wise Aggregation → {L2 Distance, Cosine Distance, Intra-domain Variation} → Semantic Drift Analysis → {PCA, t-SNE, Heatmap / Comparison Plot} → Experimental Results |
| `diagrams/fig4_2_activity_diagram.drawio` (+ preview, 71,709 B) | Fig 4.2 UML activity: START → Load DOTA and VisDrone → Parse Ground-Truth Annotations → Identify Common Categories → Extract Object Crops → Apply Class-wise Sampling → Select Vision-Language Model → Encode Satellite Samples → Encode UAV Samples → Normalize Embeddings → Group Embeddings by Class → Calculate Class Centroids → Calculate L2 + Cosine Drift → Generate PCA / t-SNE / Heatmaps → Store Experimental Results → END; note “Executed independently for each of the 3 model configurations (see Fig. 4.4)” |
| `diagrams/fig4_3_drift_computation.drawio` (+ preview, 40,846 B) | Fig 4.3 drift computation: Satellite / UAV Object Crops → Shared Encoder f_v → embeddings E1^S…En^S / E1^U…En^U → class centroids μc(S), μc(U) → “D_L2(c) = ‖μc^S − μc^U‖₂   D_cos(c) = 1 − (μc^S · μc^U)/(‖μc^S‖·‖μc^U‖)” |
| `diagrams/fig4_4_domain_adaptation_evaluation.drawio` (+ preview, 36,219 B) | Fig 4.4 evaluation: CLIP ViT-B/32 (Pretrained), RemoteCLIP ViT-L/14 (Pretrained), CLIP (Pretrained) → VisDrone Training → Fine-Tuned CLIP (UAV-Adapted) → Common Evaluation Protocol on Satellite + UAV Crops → Drift Calculation (L2, Cosine) → Model Comparison |

The diagrams depict only L2, cosine and intra-domain variation as metrics — no text-embedding metric (§4.6) — and describe the DOTA↔VisDrone pipeline only; the xView/UAVid extension is not drawn.

**The repo's own guide to which figures to use or skip** (`PROGRESS_SUMMARY.md`, section “Suggested figures and result files for the report”, verbatim — it contains important caveats):

`PROGRESS_SUMMARY.md:99-127` (verbatim)

```markdown
## Suggested figures and result files for the report

*(This section is a guide for assembling the report — not meant to be copied into it directly.)*

**Most important — include these:**

1. **`results/remoteclip_vs_clip/three_way_drift_comparison_heldout.png`** — the main result: side-by-side bar charts comparing all three models' drift scores, on the corrected, leakage-free data. This is the single most complete figure and should be the centerpiece. (Note the `_heldout` suffix — there's an earlier, superseded version of this file without that suffix; use the `_heldout` one, it's the corrected result.) **Note for non-technical audiences:** L2 and cosine drift are plotted on one shared raw-units axis here, which visually buries the cosine bars (they're numerically small because CLIP embeddings are unit-normalized — L2 ≈ √(2×cosine_distance) — not because the effect is small; cosine actually shows a *bigger* % reduction than L2 in every case). If presenting to a non-ML audience, use `results/category_yardstick/normalized_drift_reduction.png` instead/alongside — same numbers, both metrics shown as % of the CLIP baseline on one common scale, so "shorter bar = less drift" reads correctly without unit confusion.
2. **`results/remoteclip_vs_clip/drift_comparison_3way_heldout.json`** — the exact numbers behind that figure, useful if the report wants a results table (the numbers in the table above come from this file).
3. **`results/phase1_local/drift_scores_local.png`** — shows the original finding reproduced locally, establishing that the base phenomenon (drift exists) was independently confirmed before building further on top of it. **Caveat if shown:** the 6 "classes" here are the model's own zero-shot guesses on whole images, not ground-truth DOTA/VisDrone labels (see the notebook's own methodological note) — cite this as "drift exists at all," not as a trustworthy per-category ranking. The per-class numbers/order are noisy; don't quote them as if precise.

4. **`results/xview_vs_visdrone/four_class_drift_comparison.png`** — the richer-vocabulary result: 4 shared categories (car, bus, truck, building), CLIP vs. RemoteCLIP side by side. The building bars are the visually striking part — clearly taller than every vehicle category. Strong second centerpiece figure alongside #1. Numbers behind it: `results/xview_vs_visdrone/drift_comparison_4class.json`.

**Good supporting evidence (include if there's room):**

5. **`results/phase1_local/tsne_drift_local.png`** (preferred) or **`pca_combined.png`** — a scatter plot where the images split almost entirely into a "satellite half" and a "drone half," regardless of object category. This is the clearest, most intuitive visual in the whole project: it shows that *which altitude a photo came from* affects the AI's internal representation more than *what object is in the photo*. Strongly recommended as a primary figure, not just supporting material. **Caveat if shown:** this one uses whole-image embeddings and Phase-1's zero-shot-guessed labels (see the notebook's own note) — cite it as "domain identity dominates the embedding space at the image level," not as a claim about any specific category.

5b. **`results/pca_tsne/*.png`** — the ground-truth-label counterpart to #5, built on the same crops and correct labels used for #1/#2/#4 (no zero-shot label-guessing anywhere). Six files: `{pca,tsne}_{dota_vs_visdrone,xview_vs_visdrone,xview_visdrone_uavid}.png`, each a two-panel CLIP-vs-RemoteCLIP scatter, colored by true class, satellite=circle/UAV=cross. `tsne_xview_visdrone_uavid.png` is the standout — the building class forms two visibly separate clusters (satellite vs. UAV) in both panels, a direct visual match to it having the largest drift score of anything measured. Generated by `make_pca_tsne_plots.py` from cached embeddings (no GPU needed to reproduce). These also fill the report's Fig 4.8 (PCA)/Fig 4.9 (t-SNE) placeholders with a correctly-labeled version — use the DOTA/VisDrone pair there since that's what the methodology section (§4.2) describes.

5c. **`results/category_yardstick/yardstick_*.png`** (3 files) — answers "the cosine drift numbers look tiny in absolute terms, does that matter?" by comparing cross-altitude drift (same category, different domain) against cross-category distance (different category, same domain, averaged) for every class. Cross-altitude drift is *larger* than a real category swap in every single case measured — 1.1×–2.3× for the 2-class and 4-class comparisons, 3.7×–5.8× for the 3-class vehicle-only comparison (bigger there because car/bus/truck are all mutually similar, so the "different category" yardstick itself is small). Good rebuttal figure if asked whether small absolute cosine numbers make the effect meaningless. Numbers behind it: `results/category_yardstick/yardstick_table.json`. Generated by `make_category_yardstick_plots.py`.

5d. **`results/category_yardstick/drift_ratio_comparison.png`** — visualizes `drift_ratio` (L2 drift ÷ average intra-domain spread) already saved in the JSON results, per class per model. Most ratios sit in the 0.6–1.2 range — the cross-domain gap is comparable to or larger than a class's own natural spread within one domain, not negligible next to normal variation. **One honest nuance if asked:** the fine-tuned CLIP's ratio for small-vehicle (0.82) is *higher* than RemoteCLIP's (0.56) even though its raw L2 drift is lower — because fine-tuning also tightens the intra-domain spread (the denominator), sometimes by more than it reduces the cross-domain gap (the numerator). Don't claim fine-tuning uniformly improves this particular ratio; the raw L2/cosine drift numbers (#1/#2) are the metric that consistently favors it. Generated by `make_drift_ratio_plot.py`.

**Skip this one:**

6. **`results/phase1_local/distance_heatmap_local.png`** — not recommended. Its "cross-domain confusion" reading (e.g. "infrastructure" from satellite sitting closer to "industrial" from drone than to itself) rests on the same zero-shot-guessed category labels as #3, so the specific pattern could just be CLIP confusing those two label words consistently, not a real cross-domain semantic effect. Drop it rather than present it as a finding.

**Correction — do not use these two:** `results/phase1_lightning/drift_scores.png` and `results/phase1_lightning/tsne_drift.png` (the original cloud-experiment charts). On closer inspection these were generated by the *invalid* comparison described earlier in this document (the one mixing two different AI models via a broken PCA step), not by the small-scale experiment that was actually sound. Including them without heavy caveats would mean putting a flawed result in the report. There's no saved chart from the sound small-scale cloud experiment to use instead — it was only ever shown on-screen in the original notebook, never saved to a file.

**Recommendation:** for a mid-semester report, lead with #1 and #4 as the two centerpiece figures (DOTA/VisDrone 3-way, and the richer xView/UAVid 4-class result), use the table from #2 for numbers, #5 as supporting visual evidence, and #3 only as "drift exists" context (not for its specific per-category numbers) — with the caveat above ready if asked. Skip #6. All files listed above already exist and are ready to use as-is.
```

---

## 8. Known issues, TODOs, limitations

### 8.1 TODO / FIXME / XXX / HACK markers

Commands run (both over **all** tracked files, and the second over notebook *source* cells, since `git grep` would only see raw JSON):

```text
git grep -n -I -E "TODO|FIXME|XXX|HACK" -- ':!*.ipynb' ':!*.json' ':!*.drawio'      ->  (no output)
python: for every notebooks/*.ipynb and results/phase1_lightning/*.ipynb, scan every cell's source for /TODO|FIXME|XXX|HACK/  ->  no matches
```

**Result (re-verified while building this report): there are zero TODO/FIXME/XXX/HACK markers anywhere in the code base.** The project tracks open work in prose instead (sections 8.3–8.6). The nearest things to inline TODO comments are these informal flags:

```text
run_remoteclip_vs_clip.py:23: MAX_CROPS_PER_CLASS = 300  # reduced from 800 after the 30-min timeout; re-tune once real throughput is known
src\dataset.py:16: # NOTE: bicycle is deliberately excluded from small-vehicle (2026-08-19 fix) -- DOTA's
```

Docstring-level caveats in the driver scripts (verbatim):

`run_xview_vs_visdrone.py:1-14`

```python
"""
Driver for the xView-vs-VisDrone drift comparison (expanded vocabulary: car/bus/truck/van).

xView is streamed from the HichTala/xview HF mirror -- no local download, see
PROJECT_CONTEXT.md for why. This is a *preparation* pass: xView has no dedicated "van" category
("Pickup Truck" is the closest match and is rare), so van is measured but likely to be excluded
from the reported drift if it doesn't clear MIN_SAMPLES_FOR_DRIFT -- a bigger scan later can
revisit that specifically. car/bus/truck are well-populated and are the real payoff here: three
genuinely shared categories between xView and VisDrone, vs. DOTA's fixed ceiling of two
(small-vehicle/large-vehicle) that no amount of extra scanning can raise.

No fine-tuning is involved in this comparison (both encoders are off-the-shelf, frozen), so
there's no train/eval leakage concern and no need for a held-out split here.
"""
```

`verify_drift_seeds.py:1-8`

```python
"""
Robustness check for the three-way drift comparison (results/remoteclip_vs_clip/drift_comparison_3way.json).

Reuses the already-trained fine-tuned checkpoint (no retraining) and re-runs the fast
encoding/drift-computation step across several different random seeds, to check whether the
reported ranking (fine-tuned-CLIP < RemoteCLIP < CLIP) holds up across different random samples
of evaluation crops, not just the one seed (42) already reported.
"""
```

`run_remoteclip_vs_clip.py:27-31`

```python
# Must match notebooks/04_finetune_visdrone_clip.ipynb exactly -- both this script and that
# notebook need to evaluate on the *same* VisDrone crops for the 2-way and 3-way results to be
# directly comparable (rather than two different measurements of "the same" thing).
HOLDOUT_FRACTION = 0.2
SPLIT_SEED = 1234
```

### 8.2 `README.md` — “Known issues / open questions” and “Not yet started” (verbatim)

`README.md:25-31`

```markdown
**Known issues / open questions:**
- All drift numbers so far are validation-scale (300 crops/class, one seed) — need a seed/sample-size sensitivity check before being treated as final.
- `van` (xView↔VisDrone) measured but excluded — only 25 samples in a 5,000-image xView scan, below the reliability threshold; needs a bigger scan.
- Phase-1's own whole-image drift scores still group images by the model's predicted class, not ground truth (separate from the RemoteCLIP/CLIP work above, which already uses ground truth).
- PCA/t-SNE visualizations from the historical Lightning run haven't been fully audited for whether projections were fit jointly or per-domain.

**Not yet started:** full-dataset DOTA (parts 2–3)/VisDrone/xView scale-up, the altitude-conditioned alignment adapter, final evaluation metrics (mAP, cross-altitude gap), SAM-based detection integration with a teammate's work.
```

README “Next steps” (verbatim):

`README.md:121-131`

```markdown
## Next steps

Following the proposal's phase order (see `PROJECT_CONTEXT.md` §15 for the full progression):

1. ~~Resolve the RemoteCLIP-vs-fine-tuned-CLIP methodology gap~~ — done, and hardened further (bicycle-mapping + train/eval-leakage fixes, see §7c); check robustness across seeds/sample sizes before treating as final.
2. ~~Expand beyond DOTA's 2-class ceiling~~ — done via xView + UAVid, 4 shared classes now (§7d); recover `van` as a 5th with a bigger xView scan.
3. Reproduce Phase-1's own whole-image drift scores with ground-truth labels instead of model-predicted labels (the RemoteCLIP-vs-CLIP comparison already uses ground truth; Phase-1's original numbers don't yet).
4. Scale to the full DOTA (parts 2–3), VisDrone, and xView datasets once disk space allows.
5. Implement the altitude-conditioned alignment adapter (Phase 5) and evaluate mAP / cross-altitude gap / drift reduction (Phase 6).
6. Integrate with the teammate's SAM-based detection pipeline (real detection, not just classification of pre-cropped regions) to test whether drift predicts detection failures.
7. Final figures, tables, and report writeup.
```

### 8.3 `PROJECT_CONTEXT.md` — methodological issues (§14, verbatim)

`PROJECT_CONTEXT.md:666-666` (verbatim)

```markdown
# 14. Methodological Issues To Resolve
```

### 8.4 `PROJECT_CONTEXT.md` — open items (§21 checklist, verbatim)

`PROJECT_CONTEXT.md:883-883` (verbatim)

```markdown
# 21. Current Project Status Checklist
```

### 8.5 `PROJECT_CONTEXT.md` — standing limits on what may be claimed and how work proceeds (§2 “Important interpretation”, §18, §19; verbatim)

`PROJECT_CONTEXT.md:50-52`

```markdown
### Important interpretation

Describe this carefully as evidence of cross-domain semantic drift in the tested representation and data. Do not claim that semantic drift is universal across all models and datasets.
```

`PROJECT_CONTEXT.md:791-817` (verbatim)

```markdown
# 18. Claude Extension Instructions

When working on this repository, Claude should:

1. Read `PROJECT_CONTEXT.md` before making major changes.
2. Inspect existing code before creating replacement code.
3. Preserve the completed Phase-1 experiment as a reference.
4. Avoid deleting or overwriting historical artifacts.
5. Prefer small, testable changes.
6. Run cheap tests before expensive GPU jobs.
7. Never launch a full-dataset embedding job without explicit confirmation that the pipeline has passed the smaller validation stage.
8. Reuse saved embeddings whenever possible.
9. Keep dataset paths configurable rather than hard-coded.
10. Keep model paths configurable.
11. Use the `hf` Conda environment.
12. Do not create another environment unless required by a demonstrated compatibility issue.
13. Do not reinstall PyTorch/CUDA without a demonstrated need.
14. Do not download datasets/models that are already available locally.
15. Do not silently alter class mappings or preprocessing.
16. Flag methodological issues instead of silently correcting historical results.
17. Separate data processing, embedding extraction, analysis, and visualization.
18. Save expensive outputs to disk so they can be reused.
19. Before a long-running job, estimate expected runtime/storage and verify output paths.
20. Keep the project reproducible enough for a research report.
21. Never mix embeddings from two different models/encoders within a single drift comparison (Section 14a).

---
```

`PROJECT_CONTEXT.md:819-835` (verbatim)

```markdown
# 19. What Claude Should NOT Do

Do NOT:

- delete Lightning reference results,
- overwrite `phase1_lightning/`,
- rerun thousands of images unnecessarily,
- change dataset class mappings without discussion,
- switch models without documenting the change,
- silently change preprocessing,
- assume semantic drift is proven universally,
- treat t-SNE/PCA visualization alone as statistical proof,
- claim a model improves/worsens drift without quantitative comparison computed within a single shared embedding space,
- train a new model before the baseline experiment is properly reproduced,
- install large packages or environments unnecessarily.

---
```

### 8.6 `PROGRESS_SUMMARY.md` — “What's next” (verbatim)

`PROGRESS_SUMMARY.md:89-97` (verbatim)

```markdown
## What's next

- Double-check the drone-specialized result across a few different random samples of the held-out data, to see how much the exact numbers move around
- Scale up the comparisons using the full datasets rather than the current sample sizes
- Recover the 4th vehicle category (van) with a larger xView scan, now that the pipeline is proven
- Combine with the teammate's SAM-based detection work to get real detection performance numbers (not just embedding-distance numbers) and see if drift actually predicts detection failures
- Eventually build a model component that actively reduces this drift, rather than just measuring it

---
```

### 8.7 Caveats written inside notebooks (verbatim markdown cells)

`notebooks/01_phase1_reproduction.ipynb` cell 0 (scope) and cell 17 (methodological caveat on predicted labels):

```markdown
# Phase-1 Reproduction (Local)

Reproduces, on the local Windows/RTX 3050 machine, the pipeline actually documented in
`results/phase1_lightning/Capstone_v1.ipynb` (the Lightning AI Studio notebook).

**Scope note (important):** the Lightning export also contained two much larger arrays —
`remoteclip_dota_features.npy` (28,767 crops, RemoteCLIP ViT-L/14, 768-dim) and
`finetuned_visdrone_features.npy` (217,153 crops, fine-tuned CLIP, 512-dim) — but the code that
produced them is **not present** in `Capstone_v1.ipynb`. Only the CLIP ViT-B/32 pipeline below
(893/312-crop sanity check + 300-image whole-image zero-shot drift experiment) is reproducible
from the available notebook. See the project's memory / chat log for the full breakdown; the
RemoteCLIP and fine-tuned-VisDrone stages are tracked as separate future work (Notebook 03), not
reproduced here.

Two sub-pipelines are reproduced, matching the historical notebook exactly:

- **Part A** — a small 5-image-per-dataset object-crop sanity check (historically used the
  `openai/CLIP` package; here it uses HF `transformers` CLIP with the same `ViT-B/32` weights to
  avoid installing an extra package — embeddings are equivalent, not identical bit-for-bit).
- **Part B** — the actual drift-bearing experiment: 300 whole DOTA images + 300 whole VisDrone
  images, classified zero-shot against a shared vocabulary, centroid drift computed per class.
```

```markdown
**Methodological caveat (flagging per PROJECT_CONTEXT.md Section 14, not fixing silently):**
the "class" each image is grouped under below is the *model's own zero-shot prediction*
(`final_predictions` / `visdrone_predictions`), not ground-truth DOTA/VisDrone annotations.
That means the drift score computed next conflates two effects that are not separated:
(1) genuine cross-domain semantic drift within a true class, and (2) zero-shot
misclassification noise pulling images into the wrong class bucket in the first place.
This matches exactly how the historical Lightning notebook computed it — reproduced as-is here
for comparability — but it should be treated as a first-pass estimate, not final evidence, until
a ground-truth-labeled version of this analysis is run (using the actual DOTA/VisDrone box
annotations + the class mapping in PROJECT_CONTEXT.md Section 5, rather than model predictions).
```

`notebooks/03_remoteclip.ipynb` cell 6 (result + “validation-scale” caveat — note it quotes the pre-fix numbers, §5.6):

```markdown
**Result:** RemoteCLIP shows lower drift than CLIP ViT-B/32 for both classes, on both L2 and
cosine distance (small-vehicle: ~27% lower L2, ~48% lower cosine; large-vehicle: ~14% lower L2,
~29% lower cosine). On the intra-class-variance-normalized `drift_ratio`, the gap is clear for
small-vehicle (0.51 vs 0.62) but essentially disappears for large-vehicle (0.622 vs 0.623) - so the
raw-distance improvement for large-vehicle is partly explained by RemoteCLIP's embeddings simply
being more tightly clustered per-domain overall, not necessarily a cleaner cross-domain alignment.

This is **validation-scale evidence** (300 crops/class, one random seed) that remote-sensing
specialization reduces semantic drift for vehicle classes - consistent with the proposal's Phase 6
hypothesis - but it is a first pass, not a final claim. Before treating this as a reportable
result: rerun at full scale (Notebook 02), check sensitivity to the random seed / sample size, and
extend to more classes once DOTA parts 2-3 provide broader category coverage.
```

`notebooks/04_finetune_visdrone_clip.ipynb` cells 0 and 16 (fixes and how to read the leakage check):

```markdown
# Notebook 04 (v2) - Third arm: CLIP fine-tuned on VisDrone, with a proper held-out split

**This version fixes two issues found after the first run:**

1. **Bicycle mapping** - `VISDRONE_TO_COMMON` no longer maps `bicycle` into `small-vehicle`.
   DOTA's `small-vehicle` label is satellite-altitude and can only resolve cars, not bicycles;
   mixing the two visually distinct object types into one bucket (bicycles were ~5.8% of it)
   added noise to the small-vehicle drift number. Fixed in `src/dataset.py`.

2. **Train/eval leakage (the important one)** - the first version fine-tuned on *every* VisDrone
   image, then drew its evaluation sample from that same complete pool. That means the "how well
   does this model represent VisDrone" measurement was computed on images it had already been
   directly trained to represent well - the reported 63% drift reduction was likely inflated by
   memorization, not a clean measure of generalization.

   **Fix:** VisDrone images are split by filename into an 80% training pool and a 20% held-out
   pool *before* any crops are extracted (fixed seed, so it's reproducible). Fine-tuning only
   ever sees the training pool. Evaluation only ever draws from the held-out pool - genuinely
   unseen data. DOTA needs no such split (it was never part of training in the first place).

   **For a fair three-way comparison, CLIP and RemoteCLIP are also re-evaluated on this same
   held-out VisDrone sample** (not the old evaluation crops from the first run), so all three
   arms are compared on identical, leakage-free data.

**Outputs are saved under new filenames** (`*_heldout.*`) rather than overwriting the first
run's results - so the original (leaky) numbers stay around for reference/comparison rather than
silently disappearing.

**How to run this:** same as before - run interactively in Jupyter/VS Code (cell-by-cell or
"Run All" against a live kernel), not via `jupyter nbconvert --execute`, so you get live progress.
Training scope is slightly smaller than the first run (80% of VisDrone instead of 100%), so
expect roughly 80% of the previous training time.
```

```markdown
**How to read this compared to the first run:** if the fine-tuned arm's drift numbers here are
noticeably higher (closer to CLIP/RemoteCLIP) than the first run's 0.141/0.096, that confirms the
first run's low numbers were substantially driven by memorization. If they're still clearly lower
than CLIP/RemoteCLIP, that's a much more trustworthy version of the original finding - now backed
by genuinely unseen evaluation data.
```

### 8.8 Code comments that state a limitation (verbatim)

`src/dataset.py:14-19`

```python
# Only classes with real DOTA/VisDrone counterparts (PROJECT_CONTEXT.md Section 5 / 14a) --
# DOTA has no person/motorcycle equivalent, so those are intentionally excluded here.
# NOTE: bicycle is deliberately excluded from small-vehicle (2026-08-19 fix) -- DOTA's
# "small-vehicle" label is satellite-altitude, which can only resolve cars, not bicycles.
# Including bicycle mixed two visually distinct object types into one bucket (~5.8% of the
# bucket was bicycles), adding noise to the small-vehicle drift measurement.
```

`src/dataset.py:31-41`

```python
# --- Expanded vocabulary: xView <-> VisDrone (separate from COMMON_VOCAB above) ---
# xView's fine-grained vehicle taxonomy lets car/bus/truck/van map 1:1 onto VisDrone's own
# categories instead of being lumped into small-vehicle/large-vehicle. DOTA cannot use this
# scheme -- it only has the coarse small-vehicle/large-vehicle split, no finer subtypes -- so
# this is for xView<->VisDrone comparisons specifically, not a replacement for COMMON_VOCAB.
# xView has no motorcycle or person/pedestrian class (satellite altitude can't resolve them),
# so those VisDrone categories are excluded here too, same reasoning as COMMON_VOCAB above.
# "building" added 2026-08-20: xView has a real "Building" class, and UAVid (a UAV semantic
# segmentation dataset) has one too -- pairing them adds a genuinely new shared category beyond
# vehicles, and buildings are far more common per-image than any single vehicle type (unlike
# "van", which is bottlenecked by how rare xView's closest match, Pickup Truck, actually is).
```

---

## 9. Chronology

### 9.1 `git log --oneline --all` (verbatim)

```text
c401c83 Point the figure guide's centerpiece figure at the normalized version too
05edfdf Add normalized (% of CLIP baseline) drift reduction plot
0399327 Add cross-altitude-vs-cross-category yardstick and drift_ratio visualizations
43941cb Add ground-truth-label PCA/t-SNE plots for all three drift comparisons
b8c5293 Caveat Phase-1's per-class figures in the report figure guide
1c33628 Update README and PROJECT_CONTEXT for xView/UAVid results and corrected numbers
6fc8402 Add mid-semester presentation script and plain-language progress summary
5dc9698 Add draw.io source for the midsem report's UML/architecture diagrams
bd32a2b Add seed-robustness verification script for the 3-way drift comparison
26eaadd Add xView vs. VisDrone/UAVid 4-class drift comparison (car/bus/truck/building)
8369b28 Add DOTA vs. VisDrone 3-way drift comparison (CLIP / RemoteCLIP / fine-tuned)
f82e4f6 Extend dataset/embeddings pipeline for xView, UAVid, and held-out evaluation
5791ee2 Remoteclip and fined-tuned model results comparison against clip results
36209f7 Initial commit: Phase-1 semantic drift reproduction
```

### 9.2 `git log --format='%an: %s'` (verbatim; `--all` added so every ref is covered — output identical without it since only `main` exists)

```text
Caebyr: Point the figure guide's centerpiece figure at the normalized version too
Caebyr: Add normalized (% of CLIP baseline) drift reduction plot
Caebyr: Add cross-altitude-vs-cross-category yardstick and drift_ratio visualizations
Caebyr: Add ground-truth-label PCA/t-SNE plots for all three drift comparisons
Caebyr: Caveat Phase-1's per-class figures in the report figure guide
Caebyr: Update README and PROJECT_CONTEXT for xView/UAVid results and corrected numbers
Caebyr: Add mid-semester presentation script and plain-language progress summary
Caebyr: Add draw.io source for the midsem report's UML/architecture diagrams
Caebyr: Add seed-robustness verification script for the 3-way drift comparison
Caebyr: Add xView vs. VisDrone/UAVid 4-class drift comparison (car/bus/truck/building)
Caebyr: Add DOTA vs. VisDrone 3-way drift comparison (CLIP / RemoteCLIP / fine-tuned)
Caebyr: Extend dataset/embeddings pipeline for xView, UAVid, and held-out evaluation
Caebyr: Remoteclip and fined-tuned model results comparison against clip results
Caebyr: Initial commit: Phase-1 semantic drift reproduction
```

**Contributors:** `git shortlog -sne --all` →

```text
14	Caebyr <koberoi008@gmail.com>
```

There is exactly **one** git author (`Caebyr <koberoi008@gmail.com>`, the configured git user) for all 14 commits; branches: `main`, `remotes/origin/main` only; no stashes. The four other team members named in `README.md` have no commits. Trailers: the initial commit `36209f7` carries `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` (see §10 on AI-assistance disclosure).

### 9.3 Commit table with timestamps

| Commit | Author date (+0530) | Subject | Diffstat |
|---|---|---|---|
| `c401c83` | 2026-08-22 11:57:17 +0530 | Point the figure guide's centerpiece figure at the normalized version too | 1 file changed, 1 insertion(+), 1 deletion(-) |
| `05edfdf` | 2026-08-22 11:56:42 +0530 | Add normalized (% of CLIP baseline) drift reduction plot | 2 files changed, 94 insertions(+) |
| `0399327` | 2026-08-22 11:49:57 +0530 | Add cross-altitude-vs-cross-category yardstick and drift_ratio visualizations | 8 files changed, 335 insertions(+) |
| `43941cb` | 2026-08-22 11:28:19 +0530 | Add ground-truth-label PCA/t-SNE plots for all three drift comparisons | 8 files changed, 194 insertions(+), 1 deletion(-) |
| `b8c5293` | 2026-08-22 11:14:33 +0530 | Caveat Phase-1's per-class figures in the report figure guide | 1 file changed, 4 insertions(+), 4 deletions(-) |
| `1c33628` | 2026-08-22 10:45:11 +0530 | Update README and PROJECT_CONTEXT for xView/UAVid results and corrected numbers | 2 files changed, 205 insertions(+), 39 deletions(-) |
| `6fc8402` | 2026-08-22 10:44:59 +0530 | Add mid-semester presentation script and plain-language progress summary | 2 files changed, 365 insertions(+) |
| `5dc9698` | 2026-08-22 10:44:51 +0530 | Add draw.io source for the midsem report's UML/architecture diagrams | 8 files changed, 314 insertions(+) |
| `bd32a2b` | 2026-08-22 10:44:41 +0530 | Add seed-robustness verification script for the 3-way drift comparison | 1 file changed, 107 insertions(+) |
| `26eaadd` | 2026-08-22 10:44:32 +0530 | Add xView vs. VisDrone/UAVid 4-class drift comparison (car/bus/truck/building) | 6 files changed, 406 insertions(+) |
| `8369b28` | 2026-08-22 10:44:23 +0530 | Add DOTA vs. VisDrone 3-way drift comparison (CLIP / RemoteCLIP / fine-tuned) | 7 files changed, 324 insertions(+) |
| `f82e4f6` | 2026-08-22 10:44:13 +0530 | Extend dataset/embeddings pipeline for xView, UAVid, and held-out evaluation | 2 files changed, 483 insertions(+) |
| `5791ee2` | 2026-08-21 17:58:09 +0530 | Remoteclip and fined-tuned model results comparison against clip results | 2 files changed, 4005 insertions(+), 16 deletions(-) |
| `36209f7` | 2026-08-13 19:21:20 +0530 | Initial commit: Phase-1 semantic drift reproduction | 29 files changed, 8798 insertions(+) |

Note the clustering — commits per day: 2026-08-13: 1, 2026-08-21: 1, 2026-08-22: 12. Twelve of the 14 commits (`f82e4f6` … `c401c83`) were made on 2026-08-22, seven of them within one minute of each other (10:44:13 to 10:45:11). The commits record *when work was committed*, not when it was done — the file-system timeline below shows the work was done Aug 13–20 (experiments) and Aug 22 (plots/diagrams).

### 9.4 Actual order of work — file-system modification times (local time, IST) [DERIVED]

| Time | Event |
|---|---|
| 2026-04-28 22:31 | `models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt` (1,710,613,765 B) — checkpoint file mtime (predates the 2026-08-13 project setup; original download context not recorded) |
| Lightning (dated outputs: 2026-04-05) | `Capstone_v1.ipynb` executed (its DOTA/VisDrone download attempts are timestamped 2026-04-05 in the saved outputs; the drift cells are undated); `model.ipynb` run — undated, file missing |
| 2026-08-13 16:33–17:29 | project scaffolding (`notebooks/02,04_drift_analysis` empty stubs, empty `src/*.py`, `test.py`); Lightning export unpacked into `embeddings/phase1_lightning/`, `results/phase1_lightning/` |
| 2026-08-13 18:22–18:58 | `capstone_phase1_export.tar.gz` (498,453,910 B); `configs/experiment.yaml`; HF cache: CLIP `pytorch_model.bin` (18:49) and a safetensors snapshot (18:58) |
| 2026-08-13 19:02–19:03 | **Run B** Phase-1 local reproduction (`embeddings/phase1_local/*`, `results/phase1_local/*`, notebook 01) |
| 2026-08-13 19:21 | **commit `36209f7` Initial commit** |
| 2026-08-14 18:44 | `src/embeddings.py` last modified (contains the fine-tuning functions); `PROJECT_CONTEXT.md` §7b run D (2-way, pre-fix) dated 2026-08-14 |
| 2026-08-19 14:32 | HF cache: `datasets--HichTala--xview` `refs/main` written (earliest recorded xView access) |
| 2026-08-19 15:11 / 15:57 / 16:44 | **Run E** — first (leaky) fine-tuning, epoch 1/2/3 checkpoints |
| 2026-08-19 16:47 | Run E results: `drift_comparison_3way.json`, `three_way_drift_comparison.png`, first-run fine-tuned arrays |
| 2026-08-19 18:19 | `verify_drift_seeds.py` written (never produced output) |
| 2026-08-20 20:12 / 20:48 / 21:24 | **Run F** — held-out fine-tuning, epoch 1/2/3 checkpoints (after the bicycle + leakage fixes) |
| 2026-08-20 21:36–21:37 | Run F results + `notebooks/04_finetune_visdrone_clip.ipynb` (v2) saved |
| 2026-08-20 22:03–22:07 | `run_remoteclip_vs_clip.py` edited; **Run G** (2-way post-fix; overwrote `drift_comparison.json` and the CLIP/RemoteCLIP arrays); notebook 03 saved |
| 2026-08-20 22:12–22:27 | `run_xview_vs_visdrone.py`; **Run H** xView↔VisDrone; `xview_vs_visdrone_drift.png` |
| 2026-08-20 22:44–22:56 | HF cache `datasets--lironui--UAVid-RIS` (22:44), then Kaggle UAVid data placed (22:47–22:56) |
| 2026-08-20 22:58–23:05 | `src/dataset.py` last modified (UAVid extraction); `run_uavid_building.py`; **Run I** building; `four_class_drift_comparison.png` |
| 2026-08-20 23:36–23:40 | `PRESENTATION_SCRIPT.md`, `PROJECT_CONTEXT.md`, `README.md` updated |
| 2026-08-21 17:58 | **commit `5791ee2`** (notebooks 03 and 04) |
| 2026-08-22 02:03–02:08 | draw.io diagrams and previews |
| 2026-08-22 10:44–10:45 | **commits `f82e4f6` … `1c33628`** (src, drivers, results, docs) |
| 2026-08-22 11:14–11:57 | commits `b8c5293`, `43941cb`, `0399327`, `05edfdf`, `c401c83`; **Run J** plots generated 11:26–11:56 |
| 2026-08-22 13:22–13:29 | `Capstone Research Proposal Presentation-2 - UPDATED.pptx` and `Midsem-ppt.pdf` (still untracked) |

Events dated in `PROJECT_CONTEXT.md` itself: §7b “2026-08-14, validation-scale”; §7c “(2026-08-19–20)”; §7d “(2026-08-20)”; `src/dataset.py` comments “2026-08-19 fix” (bicycle) and “added 2026-08-20” (building); `run_uavid_building.py` note “building added 2026-08-20”. The mid-semester deck footer reads “As of Aug 22, 2026”.

<details><summary>Per-commit file lists (<code>git log --all --stat</code>) — verbatim</summary>

```text
commit c401c83  2026-08-22 11:57:17 +0530
    Point the figure guide's centerpiece figure at the normalized version too

 PROGRESS_SUMMARY.md | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)
commit 05edfdf  2026-08-22 11:56:42 +0530
    Add normalized (% of CLIP baseline) drift reduction plot

 make_normalized_drift_plot.py                             |  94 ++++++++++++++++++++++++++++++++++++++++++++++++++++++
 results/category_yardstick/normalized_drift_reduction.png | Bin 0 -> 119504 bytes
 2 files changed, 94 insertions(+)
commit 0399327  2026-08-22 11:49:57 +0530
    Add cross-altitude-vs-cross-category yardstick and drift_ratio visualizations

 PROGRESS_SUMMARY.md                                           |   4 ++
 make_category_yardstick_plots.py                              | 156 ++++++++++++++++++++++++++++++++++++++++++++++++++
 make_drift_ratio_plot.py                                      |  65 +++++++++++++++++++++
 results/category_yardstick/drift_ratio_comparison.png         | Bin 0 -> 103924 bytes
 results/category_yardstick/yardstick_dota_vs_visdrone.png     | Bin 0 -> 82648 bytes
 results/category_yardstick/yardstick_table.json               | 110 +++++++++++++++++++++++++++++++++++
 results/category_yardstick/yardstick_xview_visdrone_uavid.png | Bin 0 -> 84503 bytes
 results/category_yardstick/yardstick_xview_vs_visdrone.png    | Bin 0 -> 75530 bytes
 8 files changed, 335 insertions(+)
commit 43941cb  2026-08-22 11:28:19 +0530
    Add ground-truth-label PCA/t-SNE plots for all three drift comparisons

 PROGRESS_SUMMARY.md                            |   4 +-
 make_pca_tsne_plots.py                         | 191 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 results/pca_tsne/pca_dota_vs_visdrone.png      | Bin 0 -> 346289 bytes
 results/pca_tsne/pca_xview_visdrone_uavid.png  | Bin 0 -> 433221 bytes
 results/pca_tsne/pca_xview_vs_visdrone.png     | Bin 0 -> 441285 bytes
 results/pca_tsne/tsne_dota_vs_visdrone.png     | Bin 0 -> 358616 bytes
 results/pca_tsne/tsne_xview_visdrone_uavid.png | Bin 0 -> 550669 bytes
 results/pca_tsne/tsne_xview_vs_visdrone.png    | Bin 0 -> 530481 bytes
 8 files changed, 194 insertions(+), 1 deletion(-)
commit b8c5293  2026-08-22 11:14:33 +0530
    Caveat Phase-1's per-class figures in the report figure guide

 PROGRESS_SUMMARY.md | 8 ++++----
 1 file changed, 4 insertions(+), 4 deletions(-)
commit 1c33628  2026-08-22 10:45:11 +0530
    Update README and PROJECT_CONTEXT for xView/UAVid results and corrected numbers

 PROJECT_CONTEXT.md | 180 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++----------
 README.md          |  64 ++++++++++++++++++++++-----------
 2 files changed, 205 insertions(+), 39 deletions(-)
commit 6fc8402  2026-08-22 10:44:59 +0530
    Add mid-semester presentation script and plain-language progress summary

 PRESENTATION_SCRIPT.md | 244 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 PROGRESS_SUMMARY.md    | 121 ++++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 365 insertions(+)
commit 5dc9698  2026-08-22 10:44:51 +0530
    Add draw.io source for the midsem report's UML/architecture diagrams

 diagrams/fig4_1_architecture.drawio                      | 101 ++++++++++++++++++++++++++++++++++++++++++++++++++++
 diagrams/fig4_1_architecture_preview.png                 | Bin 0 -> 58726 bytes
 diagrams/fig4_2_activity_diagram.drawio                  | 106 +++++++++++++++++++++++++++++++++++++++++++++++++++++++
 diagrams/fig4_2_activity_diagram_preview.png             | Bin 0 -> 71709 bytes
 diagrams/fig4_3_drift_computation.drawio                 |  51 ++++++++++++++++++++++++++
 diagrams/fig4_3_drift_computation_preview.png            | Bin 0 -> 40846 bytes
 diagrams/fig4_4_domain_adaptation_evaluation.drawio      |  56 +++++++++++++++++++++++++++++
 diagrams/fig4_4_domain_adaptation_evaluation_preview.png | Bin 0 -> 36219 bytes
 8 files changed, 314 insertions(+)
commit bd32a2b  2026-08-22 10:44:41 +0530
    Add seed-robustness verification script for the 3-way drift comparison

 verify_drift_seeds.py | 107 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 107 insertions(+)
commit 26eaadd  2026-08-22 10:44:32 +0530
    Add xView vs. VisDrone/UAVid 4-class drift comparison (car/bus/truck/building)

 results/xview_vs_visdrone/drift_comparison.json           |  53 +++++++++++++++++++
 results/xview_vs_visdrone/drift_comparison_4class.json    |  67 ++++++++++++++++++++++++
 results/xview_vs_visdrone/four_class_drift_comparison.png | Bin 0 -> 51182 bytes
 results/xview_vs_visdrone/xview_vs_visdrone_drift.png     | Bin 0 -> 59642 bytes
 run_uavid_building.py                                     | 135 ++++++++++++++++++++++++++++++++++++++++++++++++
 run_xview_vs_visdrone.py                                  | 151 ++++++++++++++++++++++++++++++++++++++++++++++++++++++
 6 files changed, 406 insertions(+)
commit 8369b28  2026-08-22 10:44:23 +0530
    Add DOTA vs. VisDrone 3-way drift comparison (CLIP / RemoteCLIP / fine-tuned)

 results/remoteclip_vs_clip/clip_vs_remoteclip_drift.png           | Bin 0 -> 54763 bytes
 results/remoteclip_vs_clip/drift_comparison.json                  |  46 ++++++++++++++
 results/remoteclip_vs_clip/drift_comparison_3way.json             |  65 +++++++++++++++++++
 results/remoteclip_vs_clip/drift_comparison_3way_heldout.json     |  59 ++++++++++++++++++
 results/remoteclip_vs_clip/three_way_drift_comparison.png         | Bin 0 -> 66097 bytes
 results/remoteclip_vs_clip/three_way_drift_comparison_heldout.png | Bin 0 -> 64988 bytes
 run_remoteclip_vs_clip.py                                         | 154 ++++++++++++++++++++++++++++++++++++++++++++++
 7 files changed, 324 insertions(+)
commit f82e4f6  2026-08-22 10:44:13 +0530
    Extend dataset/embeddings pipeline for xView, UAVid, and held-out evaluation

 src/dataset.py    | 281 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 src/embeddings.py | 202 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 483 insertions(+)
commit 5791ee2  2026-08-21 17:58:09 +0530
    Remoteclip and fined-tuned model results comparison against clip results

 notebooks/03_remoteclip.ipynb             |  318 +++++-
 notebooks/04_finetune_visdrone_clip.ipynb | 3703 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 4005 insertions(+), 16 deletions(-)
commit 36209f7  2026-08-13 19:21:20 +0530
    Initial commit: Phase-1 semantic drift reproduction

 .gitignore                                      |   34 +
 .vscode/settings.json                           |    4 +
 PROJECT_CONTEXT.md                              |  825 ++++++++++++
 README.md                                       |  107 ++
 configs/experiment.yaml                         |   75 ++
 data/processed/.gitkeep                         |    0
 notebooks/01_phase1_reproduction.ipynb          | 3267 ++++++++++++++++++++++++++++++++++++++++++++++
 notebooks/02_full_dataset.ipynb                 |   16 +
 notebooks/03_remoteclip.ipynb                   |   16 +
 notebooks/04_drift_analysis.ipynb               |   16 +
 requirements.txt                                |    0
 results/figures/.gitkeep                        |    0
 results/full_dataset/.gitkeep                   |    0
 results/phase1/.gitkeep                         |    0
 results/phase1_lightning/Capstone_v1.ipynb      | 4422 +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 results/phase1_lightning/drift_scores.png       |  Bin 0 -> 41607 bytes
 results/phase1_lightning/tsne_drift.png         |  Bin 0 -> 182608 bytes
 results/phase1_local/distance_heatmap_local.png |  Bin 0 -> 73823 bytes
 results/phase1_local/drift_scores_local.json    |    8 +
 results/phase1_local/drift_scores_local.png     |  Bin 0 -> 50754 bytes
 results/phase1_local/pca_combined.png           |  Bin 0 -> 206107 bytes
 results/phase1_local/pca_dota.png               |  Bin 0 -> 126405 bytes
 results/phase1_local/tsne_drift_local.png       |  Bin 0 -> 198572 bytes
 src/centroids.py                                |    0
 src/dataset.py                                  |    0
 src/embeddings.py                               |    0
 src/metrics.py                                  |    0
 src/visualization.py                            |    0
 test.py                                         |    8 +
 29 files changed, 8798 insertions(+)
```

</details>

---

## 10. Anything else notable

Items are ordered roughly by how much they could change how a paper is written. Tags as defined in the Preface.

### 10.1 Evaluation-design facts that constrain what can be claimed

1. **Domain = dataset.** Every “satellite vs UAV” contrast is also a dataset contrast (DOTA: `imagesource: GoogleEarth` for all 469 images, GSD median 0.133 m/px [DERIVED]; xView; vs VisDrone/UAVid drone footage). Sensor, resolution, scene type, annotator conventions and label granularity change together with altitude. The repo never separates altitude from these factors. **[ANALYSIS]**

2. **Resolution / scale confound is large and measurable.** DOTA `small-vehicle` crops have median **34×33.5 px** (eval sample) and are up-sampled ≈6.6× to the 224 px model input; VisDrone `small-vehicle` crops have median **50.5×39.5 px**; DOTA `large-vehicle` 77×66 vs VisDrone 51.5×44.5 (§2.6 table). The encoder preprocessing resizes the *short* side to 224 and centre-crops, so elongated objects (buses, trucks, buildings) lose their ends. DOTA boxes are oriented quads converted to axis-aligned boxes, so rotated vehicles include extra background. **[ANALYSIS on DERIVED numbers]**

3. **Sampling is clustered within images.** `subsample_balanced` has no per-image cap. Replaying the seed-42 sampling **[DERIVED reconstruction]**:

| Evaluation sample | Class | Crops | Distinct source images | Max crops from one image |
|---|---|---:|---:|---:|
| DOTA eval sample (seed 42) | large-vehicle | 300 | 68 | 42 |
| DOTA eval sample (seed 42) | small-vehicle | 300 | 119 | 43 |
| VisDrone held-out eval sample (seed 42) | large-vehicle | 300 | 241 | 5 |
| VisDrone held-out eval sample (seed 42) | small-vehicle | 300 | 244 | 4 |
| VisDrone expanded eval sample, all images (seed 42) | bus | 300 | 257 | 3 |
| VisDrone expanded eval sample, all images (seed 42) | car | 300 | 289 | 3 |
| VisDrone expanded eval sample, all images (seed 42) | truck | 300 | 272 | 3 |
| VisDrone expanded eval sample, all images (seed 42) | van | 300 | 284 | 2 |

The DOTA-side samples come from only **68** (large-vehicle) and **119** (small-vehicle) of the 469 images, with up to 42–43 crops from a single image, whereas VisDrone-side samples span ~240–290 images with ≤5 crops each. The 300 crops are therefore far from independent on the DOTA side; no confidence interval or bootstrap accounts for that. (xView/UAVid image counts cannot be reconstructed offline.)

4. **No uncertainty quantification anywhere.** Every reported number is a single point estimate from one seed, n = 300/class/domain; no CI, bootstrap, permutation test or repeated-training variance exists (§5.3b).

5. **Fine-tuning is unseeded and run once per configuration.** No `torch.manual_seed`; two runs exist but they differ in data (leaky vs held-out), so they are not a variance estimate. The two runs' large-vehicle drift (0.0964 vs 0.0955) happen to agree closely; small-vehicle differs (0.1412 vs 0.1585) — with the caveat that the runs also differ in mapping and training data.

6. **Different comparisons use different pools and vocabularies**, so numbers are not interchangeable across tables: DOTA↔VisDrone uses VisDrone's *held-out 20%* and a merged vocabulary (`van` → `large-vehicle`, `car` → `small-vehicle`); xView↔VisDrone uses *all* VisDrone images and an unmerged vocabulary (`car`, `bus`, `truck`, `van`). Under the DOTA mapping 55% of VisDrone `large-vehicle` crops are vans (20,534 of 37,064; all-image counts). `building` pairs xView with **UAVid**, not VisDrone (VisDrone has no building class), so it changes the UAV dataset as well as the class. The fine-tuned arm was never evaluated on xView/UAVid, and it was trained on `small-vehicle/large-vehicle/motorcycle/person` only.

7. **Training-loss floor** (§3.4.6): with 4 distinct prompt strings in a batch of 8, duplicate prompts act as false negatives; the loss cannot go below ≈1.09 even for a perfect classifier, so the logged 1.48→1.35 says little about fit. **[ANALYSIS]**

8. **Cosine and L2 measure different things here** because centroids are not unit-norm (§4.5): `l2_drift` includes centroid-length change; the repo's “L2 ≈ √(2·cos)” note is approximate. **[ANALYSIS]**

9. **What was *not* measured**: any per-crop or per-class classification accuracy, retrieval accuracy, text-alignment, or detection quality — the study is purely about centroid geometry of image embeddings.

10. **Raw drift reductions conflate cluster compaction with domain alignment [ANALYSIS on DERIVED numbers from §4.5].** The intra-domain radius (mean distance to the centroid) is smaller for RemoteCLIP than for CLIP, and much smaller for the fine-tuned arm, on *both* domains — including DOTA, which the fine-tuned model never saw:

| Class | Encoder | DOTA intra radius | VisDrone intra radius | L2 drift | drift_ratio (L2 ÷ mean radius) |
|---|---|---:|---:|---:|---:|
| small-vehicle | CLIP | 0.3286 | 0.4286 | 0.2460 | 0.6498 |
| small-vehicle | RemoteCLIP | 0.3019 | 0.3707 | 0.1892 | 0.5627 |
| small-vehicle | Fine-tuned (held-out) | 0.2523 | 0.136 | 0.1585 | 0.8165 |
| large-vehicle | CLIP | 0.3855 | 0.4471 | 0.2596 | 0.6235 |
| large-vehicle | RemoteCLIP | 0.3444 | 0.3837 | 0.2276 | 0.6252 |
| large-vehicle | Fine-tuned (held-out) | 0.2221 | 0.2399 | 0.0955 | 0.4132 |

So part of the L2 (and cosine) reduction reported for the specialised arms is a global contraction of the embedding clusters; `drift_ratio` is the repo's only normalisation for this, and it moves the other way for the fine-tuned small-vehicle class (0.6498 → 0.8165) and is essentially unchanged for RemoteCLIP large-vehicle (0.6235 → 0.6252). A paper claiming that specialisation *aligns* domains, rather than merely tightens clusters, would need to argue from the ratio or from a scale-free measure.

### 10.2 Failed / abandoned / superseded approaches (with evidence)

| Approach | What happened | Evidence |
|---|---|---|
| Lightning large-scale “RemoteCLIP (DOTA) vs fine-tuned CLIP (VisDrone)” drift | Declared invalid (different models, two independent PCA-256 fits); redone as same-encoder-both-domains | `PROJECT_CONTEXT.md` §7a, §14a |
| ViT-L/14 fine-tune on Lightning (to match RemoteCLIP's dimensionality without PCA) | cut off at epoch 3, 70% through; no checkpoint or embeddings produced | `PROJECT_CONTEXT.md` §7a item 5, §7 (no `visdrone_vit14_epoch3.pt`) |
| Running RemoteCLIP extraction inside a `jupyter nbconvert --execute` kernel | hung the full 1,800 s timeout on ~3,200 crops (benchmark ≈14–15 crops/s ⇒ ≈4 min); a plain script ran at full speed. Still visible: RemoteCLIP ran at 1.7–2.4 crops/s inside the notebook kernel in notebook 04 | `PROJECT_CONTEXT.md` §7b runtime note; `README.md`; `notebooks/04_finetune_visdrone_clip.ipynb` cell 13 [LOG] |
| `MAX_CROPS_PER_CLASS = 800` | reduced to 300 after the timeout; comment says “re-tune once real throughput is known” — never re-tuned | `run_remoteclip_vs_clip.py:23` |
| First fine-tuned arm | train/eval leakage + bicycle mapping bug; reported “63%” reduction re-measured after fixes | notebook 04 cell 0; `PROJECT_CONTEXT.md` §7c |
| Direct dataset downloads on Lightning | DOTA `train.zip` 404; VisDrone GitHub zip 404; Kaggle `huanghanchina/dota-v15` 403 | `results/phase1_lightning/Capstone_v1.ipynb` cells 6, 7, 15 [LOG] |
| UAVid from the official DLR/UTwente server | unreachable (connection refused); a Hugging Face UAVid variant (`lironui/UAVid-RIS`, referring-segmentation derived, cannot be redistributed per its card) was fetched at 2026-08-20 22:44 — card only, no data — then Kaggle `giavuongnguyen/modified-uavid-dataset` was used. The repo never mentions `UAVid-RIS` | `PROJECT_CONTEXT.md` §7d; HF cache `datasets--lironui--UAVid-RIS` [EXTERNAL] |
| Newer `transformers` | CVE-2025-32434 guard refuses `torch.load` under torch<2.6 for `pytorch_model.bin` ⇒ pinned to 4.49.0 instead of upgrading torch. The HF cache holds a stray `model.safetensors`-only snapshot consistent with an earlier attempt | `README.md`, `PROJECT_CONTEXT.md` §8; HF cache [EXTERNAL] |
| Seed-robustness check (`verify_drift_seeds.py`) | written, never run to completion / no output; would target the leaky checkpoint | §5.3(b) |
| Empty scaffold | `src/{metrics,centroids,visualization}.py`, notebooks 02 and 04_drift_analysis — created, never filled | §4.1 |

### 10.3 Artifacts that are referenced but missing

- `model.ipynb` (Lightning) and its checkpoints `visdrone_clip_epoch{1,2,3}.pt`, `visdrone_vit14_epoch{1,2}.pt` (§3.5).

- The v1 (leaky) version of notebook 04 and its training log (§3.4.7).

- Generating code for `results/xview_vs_visdrone/four_class_drift_comparison.png` and `xview_vs_visdrone_drift.png` (§7).

- Raw pre-subsample xView class counts and UAVid crop count (never logged), the pre-fix CLIP/RemoteCLIP arrays of run D (overwritten), and any stdout log of runs G–I (only the notebooks kept output).

- Origin/download command for: the DOTA `part*.zip`/`labelTxt.zip`, the VisDrone zip, and the RemoteCLIP checkpoint (§2, §3.3).

- `Capstone Proposal.pdf` — `README.md` says to drop it into the repo; it is on the Desktop (`C:\Users\Kabir Oberoi\Desktop\Capstone Proposal.pdf`, 15 pages, 319,203 B) but not in the repo folder.

### 10.4 Documentation inconsistencies to be aware of

| Item | Detail |
|---|---|
| UAVid test split size | `PROJECT_CONTEXT.md` §4 says 30 images; disk has 150 (§2.4). |
| Stale first-pass table | `PROJECT_CONTEXT.md` §7b still shows the pre-fix table (labelled superseded by §7c); `notebooks/03_remoteclip.ipynb` cell 6 quotes the pre-fix percentages while its executed cell 5 shows post-fix (§5.6). |
| README repo-structure block | does not mention: `run_xview_vs_visdrone.py`, `run_uavid_building.py`, `verify_drift_seeds.py`, `make_pca_tsne_plots.py`, `make_category_yardstick_plots.py`, `make_drift_ratio_plot.py`, `make_normalized_drift_plot.py`, `diagrams`, `pca_tsne`, `category_yardstick`, `xview_vs_visdrone` (checked by substring search of `README.md`). |
| Project-structure section of `PROJECT_CONTEXT.md` (§11/§12) | still labels notebooks 02, 03, 04_drift_analysis “(planned)” although 03 is complete and `04_finetune_visdrone_clip.ipynb` (not listed) exists. |
| `configs/experiment.yaml` | `n_images: 300  # matches historical cell (image_files[:300])` — but the local run sorts the listing while the Lightning run did not, so the two 300-image subsets differ (§2.1). The file also lists 5 DOTA classes under `dota.classes` although the shared vocabulary has 15 keys. It covers Phase-1 only; no config file exists for runs D–J (their constants live in the scripts). |
| `README.md` open question on PCA/t-SNE audit | The sound Phase-1 notebooks fit PCA/t-SNE **jointly** on the combined features (`pca.fit_transform(combined_features)`), so that part is resolved; the two saved Lightning PNGs come from the missing `model.ipynb` and remain unaudited (§4.4, §7). |
| Notebook kernelspec names | the notebooks carry four different kernelspec display names (`Python (hf)` in notebook 01, `Python (hf-capstone)` in 03, `hf` in 04_finetune, `Python 3` in the two empty notebooks); `PROJECT_CONTEXT.md` §8 says a dedicated `hf-capstone` kernel was added to run notebooks headlessly with `nbconvert`. |

### 10.5 Reproducibility notes

- The repository *reproduces the historical crop counts* exactly (9,570 DOTA; 153,406 VisDrone pre-fix; 167,699 train pool) — the extraction logic is faithful to the original.

- All headline numbers can be regenerated from `embeddings/**` without a GPU (the drift formulas are 6 lines of numpy; §4.2), and `make_*.py` regenerate all analysis figures from cache; but re-encoding needs the git-ignored `models/` and `data/raw/` folders, which are not in git.

- `requirements.txt` is empty and no lock file exists; the environment is described only by prose and by the pip list in §1.4.

- Non-determinism: unseeded fine-tuning; the sample of crops depends on `os.listdir` sorting (`sorted` is used in all `get_*_crops`) and Python's `random.Random` — stable across platforms for a fixed seed; xView depends on the HF stream order.

### 10.6 Disclosure-relevant repository facts

- The repo was developed with an AI coding assistant: the initial commit has the trailer `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`; `PROJECT_CONTEXT.md` states it is “intended for use with the Claude extension in VS Code” and contains standing instructions addressed to it (§18–19, quoted in §8.5); `.claude/settings.local.json` (git-ignored) exists. If the target venue has an AI-assistance disclosure policy, this is the relevant provenance.

- Untracked/ignored on disk: `Capstone Research Proposal Presentation-2 - UPDATED.pptx` (13,217,931 B) and `Midsem-ppt.pdf` (3,939,987 B) are untracked; `capstone_phase1_export.tar.gz`, `models/**`, `embeddings/**`, `data/raw/**` are git-ignored (so a fresh clone lacks all of them).

- Mid-semester deck claims (`Midsem-ppt.pdf`/`.pptx`, **[EXTERNAL]**, text extracted): “DOTA↔VisDrone (3-way): RemoteCLIP and our own fine-tuned CLIP both show consistently lower drift than general CLIP, on held-out data with no train/eval leakage”; “xView↔VisDrone/UAVid (4-class): RemoteCLIP reduces drift on all 4 categories — building shows the largest drift and the largest reduction (30%)”; “specialization cuts drift 12–88% across every class and metric tested”; “6 object categories tested; 2 independent satellite datasets (DOTA, xView); 2 independent UAV datasets (VisDrone, UAVid); RemoteCLIP reduces drift on ALL 6 categories (12–30%)”. Each is consistent with the JSONs (§5.4, §5.6) but all rest on single-seed, n = 300 point estimates (§10.1). The deck's phase-completion bar chart is an image and could not be read.

