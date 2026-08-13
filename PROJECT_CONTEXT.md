# CAPSTONE PROJECT — PROJECT CONTEXT & EXECUTION PLAN

## Purpose

This is the authoritative working context for the Capstone project. It is intended for use with the Claude extension in VS Code so development can continue without repeatedly reconstructing the project history.

**Important:** Clearly distinguish completed results, recovered historical artifacts, and planned experiments. Do not present planned results as completed findings.

---

# 1. Project Overview

## Research theme

The project investigates **semantic drift in vision-language representations across aerial-image domains/viewpoints**, with a focus on satellite/UAV-style imagery and vision-language representations.

Core research question:

> **Does the semantic representation of the same object/class remain consistent when the visual domain changes, particularly across different aerial imaging conditions such as satellite and UAV imagery?**

The central research goal is to investigate whether semantically corresponding objects occupy systematically different regions of embedding space across aerial domains, and whether stronger/domain-specialized vision-language models change that behavior.

---

# 2. What Has Already Been Completed

## Phase-1 — Initial semantic-drift experiment: COMPLETED

The first stage was executed in the previous Lightning AI Studio environment.

The experiment was initially run on approximately **50 images per relevant dataset/class**, primarily to validate:
- dataset parsing,
- object/crop extraction,
- embedding extraction,
- label mapping,
- centroid/distribution analysis.

It was then expanded to approximately **300 images per dataset/class**.

### Main Phase-1 finding

The experiments showed evidence of **semantic drift**:

> Semantically corresponding objects/classes from different aerial domains do not occupy exactly the same region of the learned embedding space.

The same nominal semantic category can have a different embedding distribution depending on the imaging domain.

This is the key result established so far.

### Important interpretation

Describe this carefully as evidence of cross-domain semantic drift in the tested representation and data. Do not claim that semantic drift is universal across all models and datasets.

---

# 3. Original Experimental Pipeline

The original pipeline was approximately:

```text
Aerial dataset
    ↓
Object detection annotations / bounding boxes
    ↓
Object crops
    ↓
Vision-language image encoder
    ↓
Image embeddings
    ↓
Class/domain grouping
    ↓
Centroids + embedding distributions
    ↓
Cross-domain comparison
    ↓
Semantic-drift evidence
```

Two major datasets/domains used are:

- DOTA
- VisDrone

The existing experiment included RemoteCLIP and other CLIP/fine-tuned representations.

---

# 4. Dataset Details

## DOTA

DOTA is already available locally in ZIP parts plus a label archive.

Intended location:

```text
data/raw/DOTA/
```

Current local structure:

```text
data/
└── raw/
    └── DOTA/
        ├── images/
        ├── labelTxt/
        ├── labelTxt.zip
        ├── part1.zip
        ├── part2.zip
        └── part3.zip
```

Do not unnecessarily re-download DOTA. Only `part1.zip` + `labelTxt.zip` have been extracted so far (469 images) — matches the historical Lightning Phase-1 subset. `part2.zip`/`part3.zip` remain unextracted pending disk space for the full-dataset stage.

The previous pipeline used DOTA object annotations/bounding boxes and extracted object crops before embedding.

## VisDrone

VisDrone is also already available locally and fully extracted.

Location:

```text
data/raw/VisDrone/VisDrone2019-DET-train/
├── images/         (6,471 images)
└── annotations/    (6,471 annotation files)
```

---

# 5. Important Class Mapping

The previous experiment used semantic mapping between dataset-specific categories.

Relevant mapping included:

```text
DOTA:
small-vehicle → small-vehicle
large-vehicle → large-vehicle

VisDrone:
car/bicycle → small-vehicle
van/truck/bus → large-vehicle
motor/tricycle → motorcycle
person/pedestrian → person
```

This mapping must be verified against the original Phase-1 notebook/artifacts before final experiments.

**Do not silently change class mappings.**

---

# 6. Original Lightning Environment

The previous experiments were run in a Lightning AI Studio.

Important artifacts were stored under:

```text
/teamspace/studios/this_studio/workspace/outputs/
```

That directory was approximately **5.4 GB**.

A reduced Phase-1 export was created containing:

```text
*.npy
*.png
*.ipynb
```

The reduced export was approximately **525 MB** and was archived as:

```text
capstone_phase1_export.tar.gz
```

This archive was downloaded locally and has since been extracted and organized (see Section 7).

### Historical/reference principle

Do not overwrite historical Lightning artifacts with new local outputs.

The desired workflow is:

```text
Lightning Phase-1
       ↓
historical/reference results
       ↓
local reproduction
       ↓
validate reproducibility
       ↓
new full-scale experiments
```

---

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

## 7a. `model.ipynb` — the notebook that explains the large artifacts

`Capstone_v1.ipynb` only documents the small-scale CLIP ViT-B/32 pipeline. A second Lightning notebook, `model.ipynb`, was recovered separately and explains `remoteclip_dota_features.npy` and `finetuned_visdrone_features.npy`:

1. **RemoteCLIP ViT-L/14** (frozen, off-the-shelf, remote-sensing-pretrained) embeds every object crop from all 469 local DOTA images → `remoteclip_dota_features.npy`, labeled with the raw 15-class DOTA taxonomy.
2. A **separate, different model** — vanilla `openai/clip-vit-base-patch32` — is fine-tuned on VisDrone crops (contrastive loss, last-2 vision-encoder layers + projection unfrozen, prompts like `"an aerial photo of a {class}"`) over all 217,153 VisDrone crops, saving `visdrone_clip_epoch{1,2,3}.pt`.
3. The fine-tuned checkpoint is used to re-embed all 217,153 VisDrone crops → `finetuned_visdrone_features.npy`.
4. **The drift measurement step fits two independent PCA models — one on DOTA, one on VisDrone — each reducing separately to 256-dim**, then takes centroid distances across those two independently-fit spaces (large-vehicle: L2=0.20/cosine=0.36; small-vehicle: L2=0.125/cosine=0.27).
5. A later cell attempts a real fix — fine-tuning CLIP ViT-L/14 (matching RemoteCLIP's dimensionality, avoiding the need for PCA) — but is cut off at epoch 3, 70% through. No checkpoint or embeddings were produced from this attempt.

**This comparison is methodologically invalid as-is** (see Section 14) and its numbers should not be cited in the report until redone properly.

---

# 8. Local Environment — Current Baseline

The project is being migrated away from Lightning AI to a local Windows/VS Code setup.

## Local project root

```text
C:\Users\Kabir Oberoi\Desktop\Capstone
```

## Correct Conda environment

Use:

```text
hf
```

Interpreter:

```text
C:\anaconda\envs\hf\python.exe
```

Do NOT use the `tf` environment for this project.

### `tf` environment

Tested:

```text
Python 3.9.23
PyTorch 1.10.2
CUDA available: False
```

Therefore it is not the correct environment.

### `hf` environment

Tested successfully:

```text
Python 3.9.25
PyTorch 2.5.1
CUDA: 11.8
CUDA available: True
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
VRAM: approximately 4 GB
Transformers: pinned to 4.49.0 (see note below)
OpenCLIP: installed successfully
```

**`transformers` version note:** versions ≥4.5x enforce a hard guard (CVE-2025-32434) that refuses `torch.load`-based checkpoint loading unless torch ≥2.6. Since `openai/clip-vit-base-patch32` ships only a `pytorch_model.bin` (no `safetensors`), this blocks loading under torch 2.5.1. Rather than upgrade the verified torch/CUDA setup, `transformers` is pinned to `4.49.0`, which predates the guard. `nbconvert`/`nbclient`/`nbformat`/`ipykernel` (a dedicated `hf-capstone` kernel) were also added to run notebooks headlessly.

### Important

Do NOT reinstall PyTorch/CUDA unnecessarily.

Do NOT create another Conda environment unless a real compatibility problem is demonstrated.

---

# 9. Local GPU Constraints

GPU:

```text
NVIDIA GeForce RTX 3050 Laptop GPU
~4 GB VRAM
```

This is sufficient for the project's embedding-extraction experiments, but batch sizes must remain small.

Start with:

```text
batch size = 1–4
```

Reduce if CUDA out-of-memory occurs.

Use inference/no-gradient execution wherever possible:

```python
with torch.no_grad():
    ...
```

Potentially use FP16 only after confirming that the particular model implementation supports it safely.

Do not assume the 4 GB GPU is insufficient; test the model first.

---

# 10. RemoteCLIP Status

RemoteCLIP has already been installed/downloaded locally.

Existing local model:

```text
models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt
```

Therefore:

**Do NOT download or reinstall RemoteCLIP unless the existing local implementation is demonstrably incompatible.**

---

# 11. Local Project Structure (implemented)

```text
Capstone/
│
├── data/
│   ├── raw/
│   │   ├── DOTA/            (part1 extracted; part2/part3 pending disk space)
│   │   └── VisDrone/        (fully extracted)
│   └── processed/
│
├── models/
│   └── RemoteCLIP/
│       └── RemoteCLIP-ViT-L-14.pt
│
├── embeddings/
│   ├── phase1_lightning/    (historical — do not overwrite)
│   ├── phase1_local/        (local Phase-1 reproduction)
│   ├── clip/
│   ├── remoteclip/
│   └── finetuned/
│
├── results/
│   ├── phase1_lightning/    (historical reference notebook + figures)
│   ├── phase1_local/        (local reproduction figures + drift scores)
│   ├── full_dataset/
│   └── figures/
│
├── notebooks/
│   ├── 01_phase1_reproduction.ipynb   (done)
│   ├── 02_full_dataset.ipynb          (planned)
│   ├── 03_remoteclip.ipynb            (planned)
│   └── 04_drift_analysis.ipynb        (planned)
│
├── src/
│   ├── dataset.py
│   ├── embeddings.py
│   ├── centroids.py
│   ├── metrics.py
│   └── visualization.py
│
├── configs/
│   └── experiment.yaml
│
├── requirements.txt
├── README.md
└── PROJECT_CONTEXT.md
```

---

# 12. Meaning of the Notebook Phases

The notebooks are stages of one reproducible pipeline, not independent projects.

## Notebook 01 — Phase-1 Reproduction

```text
01_phase1_reproduction.ipynb
```

Purpose:

> Reproduce the completed Lightning Phase-1 experiment locally.

Status: **done.** Uses the existing 300-image setup as the benchmark (not the full dataset). Reproduces local dataset parsing, crop extraction, model loading, embedding generation, and the Phase-1 analysis, and compares local results with Lightning results (see `results/phase1_local/`).

---

## Notebook 02 — Full Dataset

```text
02_full_dataset.ipynb
```

Purpose:

> Scale the validated pipeline to the complete available datasets.

It should:
- load DOTA/VisDrone,
- extract relevant object crops,
- run the selected embedding model,
- save embeddings and labels to disk.

It should NOT redo expensive model inference every time analysis changes.

---

## Notebook 03 — RemoteCLIP

```text
03_remoteclip.ipynb
```

Purpose:

> Compare standard CLIP-style representations with remote-sensing-specialized representations such as RemoteCLIP — done properly this time (same frozen encoder applied to both domains per comparison; see Section 7a/14 for what went wrong in the first attempt).

Core scientific question:

> Does remote-sensing specialization reduce the observed semantic drift?

Do not assume the outcome in advance.

---

## Notebook 04 — Drift Analysis

```text
04_drift_analysis.ipynb
```

Purpose:

> Analyze already-generated embeddings without rerunning expensive model inference.

Potential analyses:
- centroid cosine similarity/distance,
- inter-domain distances,
- intra-class variance,
- embedding distribution comparisons,
- PCA,
- t-SNE/UMAP where appropriate,
- class-wise comparisons,
- statistical analysis,
- final figures/tables.

This notebook should consume saved embeddings from previous stages.

---

# 13. Critical Experimental Principle

**Never rerun expensive embedding extraction if existing embeddings can answer the question.**

Example:

```text
Need a new PCA plot?
        ↓
Use existing embeddings
        ↓
DO NOT rerun RemoteCLIP
```

But:

```text
Changed model
        ↓
New embeddings required
```

or:

```text
Changed preprocessing
        ↓
New embeddings required
```

This is particularly important because the local GPU has only ~4 GB VRAM.

---

# 14. Methodological Issues To Resolve

## 14a. Cross-model PCA mismatch (confirmed, Section 7a)

`model.ipynb`'s large-scale drift comparison embeds DOTA with RemoteCLIP and VisDrone with a separately fine-tuned CLIP, then fits **two independent PCA models** (one per domain) to force both into 256-dim before differencing centroids. There is no shared embedding space underlying this comparison — the reported numbers conflate real drift with (a) the fact that the two sides come from different model weights entirely, and (b) arbitrary PCA basis/rotation mismatch between the two independent fits. The VisDrone-side model was also actively fine-tuned to pull its classes toward text prompts while the DOTA side was untouched, which alone would manufacture apparent "drift" regardless of any real domain effect.

**Fix:** never mix models across domains in a single drift score. Run the *same* frozen encoder over both DOTA and VisDrone crops to get one drift number per encoder, then compare encoders against each other (e.g. "CLIP drift" vs. "RemoteCLIP drift"), matching the proposal's own Phase 6 design.

## 14b. Predicted vs. ground-truth labels

The Phase-1 (`Capstone_v1.ipynb` / `01_phase1_reproduction.ipynb`) drift scores group images by the model's own zero-shot predicted class, not ground-truth annotations. This conflates real cross-domain drift with zero-shot misclassification noise. Treat as a first-pass estimate; redo with ground-truth labels + the Section 5 class mapping before treating as final evidence.

## 14c. Joint vs. separate PCA/t-SNE fits generally

Before using any PCA/t-SNE result as final scientific evidence, inspect the exact implementation: whether PCA was fitted jointly or separately, which embeddings were used, preprocessing, normalization, number of components, and exact distance metrics. Do NOT blindly reproduce an old PCA implementation and call it final.

This is methodological cleanup, not a reason to discard the Phase-1 finding.

---

# 15. Experimental Progression

Recommended order:

```text
STEP 0
Recover Lightning Phase-1 artifacts                          [DONE]
        ↓
STEP 1
Inspect old notebook(s) + saved embeddings                   [DONE — see Section 7a]
        ↓
STEP 2
Test local RemoteCLIP on one image                           [NOT YET DONE]
        ↓
STEP 3
Reproduce 300-image Phase-1 locally                           [DONE]
        ↓
STEP 4
Compare local vs Lightning results                            [DONE]
        ↓
STEP 5
Fix the RemoteCLIP-vs-fine-tuned-CLIP methodology (14a)        [NOT YET DONE]
        ↓
STEP 6
Reproduce drift with ground-truth labels (14b)                 [NOT YET DONE]
        ↓
STEP 7
Run full DOTA/VisDrone dataset                                 [NOT YET DONE]
        ↓
STEP 8
Run specialized-model comparison (proper, single-encoder)       [NOT YET DONE]
        ↓
STEP 9
Perform final drift analysis + alignment module (Phase 5/6)     [NOT YET DONE]
        ↓
STEP 10
Create final figures/tables and write research report          [NOT YET DONE]
```

Do not skip directly to Step 7.

---

# 16. Artifact Naming Convention

Use descriptive names.

```text
embeddings/
├── phase1_lightning/
│   ├── remoteclip_dota_features.npy
│   ├── remoteclip_dota_labels.npy
│   ├── finetuned_visdrone_features.npy
│   ├── finetuned_visdrone_labels.npy
│   ├── dota_feats.npy
│   ├── dota_labels.npy
│   ├── vis_feats.npy
│   └── vis_labels.npy
│
├── phase1_local/
│   ├── dota_feats.npy / dota_labels.npy
│   ├── vis_feats.npy / vis_labels.npy
│   ├── dota_whole_image_feats.npy / dota_whole_image_predicted_labels.npy
│   └── visdrone_whole_image_feats.npy / visdrone_whole_image_predicted_labels.npy
│
└── full_dataset/           (not yet populated)
    ├── remoteclip_dota_features.npy
    ├── remoteclip_dota_labels.npy
    ├── remoteclip_visdrone_features.npy
    └── remoteclip_visdrone_labels.npy
```

Do not overwrite historical artifacts.

---

# 17. Reproducibility Requirements

Every major experiment should record:

- dataset version/source,
- number of images,
- number of object crops,
- class mapping,
- model name,
- checkpoint,
- preprocessing,
- image resolution,
- batch size,
- device,
- embedding dimension,
- normalization,
- random seed where applicable,
- output file names.

Where practical, store configuration in:

```text
configs/experiment.yaml
```

The Phase-1 reproduction run's actual configuration is recorded there already — do not fabricate values, record the actual values used.

---

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

# 20. Research Story Being Built

```text
Problem
  ↓
Vision-language representations may not remain semantically aligned
across different aerial domains.
  ↓
Phase-1
  ↓
Observe semantic drift on controlled subsets.                      [DONE]
  ↓
Validation
  ↓
Reproduce the result locally and ensure it is not an environment
artifact.                                                            [DONE]
  ↓
Methodology fix
  ↓
Resolve the cross-model PCA mismatch and predicted-label caveats.   [NEXT]
  ↓
Scaling
  ↓
Test the phenomenon on the full datasets.
  ↓
Model comparison
  ↓
Test whether domain-specialized representations such as RemoteCLIP
mitigate the drift — with a single shared encoder per comparison.
  ↓
Analysis
  ↓
Quantify and visualize the cross-domain differences.
  ↓
Conclusion
  ↓
Determine whether the observed drift is:
- reduced by domain specialization,
- persistent despite specialization,
- or sensitive to representation/model choice.
```

The final conclusion must follow the actual results.

---

# 21. Current Project Status Checklist

## Completed

- [x] Phase-1 50-image experiment (Lightning)
- [x] Phase-1 300-image experiment (Lightning)
- [x] Evidence of semantic drift observed
- [x] Local DOTA available (part1; part2/3 pending)
- [x] Local VisDrone available (full)
- [x] Local NVIDIA GPU available
- [x] PyTorch CUDA verified
- [x] Transformers installed (pinned to 4.49.0)
- [x] OpenCLIP installed
- [x] RemoteCLIP code + checkpoint available locally
- [x] Lightning Phase-1 archive downloaded, extracted, and organized
- [x] Historical embeddings/notebook inspected
- [x] `model.ipynb` (RemoteCLIP/fine-tuned-VisDrone pipeline) recovered and inspected
- [x] Phase-1 reproduced locally end-to-end

## In progress / next up

- [ ] Test local RemoteCLIP on one image
- [ ] Fix the RemoteCLIP-vs-fine-tuned-CLIP methodology (Section 14a)
- [ ] Reproduce drift with ground-truth labels (Section 14b)

## Not yet completed

- [ ] Full DOTA experiment (parts 2–3)
- [ ] Full VisDrone experiment at full scale with a fixed methodology
- [ ] Rigorous final drift metrics
- [ ] Standard CLIP vs RemoteCLIP comparison at full scale (single shared encoder)
- [ ] Altitude-conditioned alignment module (Phase 5)
- [ ] Final evaluation metrics (mAP, cross-altitude gap, semantic drift reduction)
- [ ] Final visualizations
- [ ] Final report/paper

---

# 22. Definition of Success for the Next Stage

The next stage is NOT successful merely because RemoteCLIP loads.

The first milestone is:

> **A local run reproduces the Phase-1 pipeline and produces results that are reasonably consistent with the existing Lightning experiment under the same configuration.**

This milestone has been reached (Section 21). The next milestone is resolving the cross-model methodology gap (Section 14a) before scaling further.

---

# 23. Final Guidance

This is a research project, not just a coding project.

Prioritize:

```text
Reproducibility
    >
Correct methodology
    >
Controlled experiments
    >
Quantitative evidence
    >
Visualization
    >
Scale
```

Do not optimize for getting a result quickly at the expense of being able to explain exactly how the result was obtained.

The intended research story is:

> **We observed semantic drift across aerial domains, validated it locally, are resolving a methodology gap in the cross-model comparison, and will next test whether domain-specialized vision-language models mitigate the drift once that comparison is done correctly.**

The final claim must be determined by the experiments, not assumed beforehand.
