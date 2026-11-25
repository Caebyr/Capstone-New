# Uncertainty, crop classification and text alignment (October 2026)

Added for the RAIT 2026 submission. Everything here reuses the cached embeddings behind Tables II–III; no image was re-embedded for any reported number.

## Commands (hf env, from the repo root, in this order)

```
python build_crop_manifests.py        # ~25 min: replays crop extraction, writes manifests, verifies rows (GPU, streams xView)
python drift_stats.py                 # ~13 min: permutation tests + image/sequence bootstrap for visual drift (CPU)
python classify_crops.py              # <1 min: zero-shot crop classification + A_c, delta_m_c point estimates (GPU for text only)
python classify_crops.py --bootstrap  # ~7 min: same + image/sequence bootstrap CIs
python join_results.py                # joined drift / classification / text table
```

All scripts exit non-zero if any built-in check fails. The final runs passed: manifests 14/14 list–model identity checks, `drift_stats.py` 56/56, `classify_crops.py` 56/56 plus synthetic unit checks.

## Assets

| Asset | Identity |
|---|---|
| CLIP ViT-B/32 | Hugging Face `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| RemoteCLIP ViT-L/14 | `models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt`, sha256 `fcc2a7e21e171f4ffcb7a9c0206b8b74ac0c9eb83c67b576958b7a4ed6c8cecb` |
| Fine-tuned CLIP | `models/finetuned_clip_visdrone_heldout/finetuned_clip_visdrone_heldout_epoch3.pt`, sha256 `3c150faac15620b4edce39abfe50d4191f029df85e45ad4a03f17fb10be0ba2a` |
| xView | Hugging Face dataset `HichTala/xview`, revision `00a1b977dc16f6eda77cce0236ca2c2a13ac840b`, first 5,000 streamed train chips (500 for buildings) |
| Software for these analyses | torch 2.5.1+cu118, transformers 4.49.0, open_clip_torch 3.3.0, scikit-learn 1.3.2, numpy 1.26.4, Pillow 11.3.0 |

The cached embeddings themselves were produced in August 2026 in the original `C:naconda\envs\hf` environment: Python 3.9.25, torch 2.5.1 (CUDA 11.8), torchvision 0.20.1, transformers 4.49.0, open_clip 3.3.0, numpy 1.26.4, Pillow 11.1.0. That environment was lost in the laptop reset, but its versions were printed in the pre-reset session transcripts; see `docs/paper/PROJECT_FACTS_FOR_PAPER.md` and `results/paper_facts/original_environment_records.txt`. (Corrected 14 Oct: an earlier version of this note said those versions were not recorded.)

## Crops and crop IDs

`results/crop_manifests/<list>.csv` gives, for row *i* of each cached `*_feats.npy`: crop ID, label, dataset, source image, sequence (where one exists), box, and size.

| List | Crops | Images | Sequences | Embedding files (prefix) |
|---|---|---|---|---|
| p1_dota | 600 (300 small-, 300 large-vehicle) | 149 | — | `embeddings/remoteclip_vs_clip/dota_{clip,remoteclip,finetuned_clip_heldout}` |
| p1_visdrone_heldout | 600 | 435 | 98 | `embeddings/remoteclip_vs_clip/visdrone_{clip,remoteclip,finetuned_clip_heldout}` |
| p2_xview | 925 (300 car/bus/truck, 25 van: van unused) | 594 | — | `embeddings/xview_vs_visdrone/xview_{clip,remoteclip}` |
| p2_visdrone | 1,200 (300 each car/bus/truck/van: van unused) | 1,002 | 120 | `embeddings/xview_vs_visdrone/visdrone_{clip,remoteclip}` |
| p3_xview_building | 300 | 127 | — | `embeddings/xview_vs_visdrone/xview_building_{clip,remoteclip}` |
| p3_uavid_building | 300 | 164 | 27 | `embeddings/xview_vs_visdrone/uavid_building_{clip,remoteclip}` |

**How the manifests were rebuilt.** The original runs did not save crop provenance. Crop extraction is deterministic, so `build_crop_manifests.py` replays it with the original settings (seed 42, 300 per class, held-out split seed 1234, and the label mappings in use at the time).

- **Pre-"Building" xView mapping.** The car/bus/truck xView run predates "Building" being added to `XVIEW_TO_EXPANDED`, so it is replayed without it.
- **Label check:** for every list and model, the replayed label sequence equals the cached labels exactly.
- **Row-identity check:** every crop was re-encoded with every model and compared with all cached rows:
  - PNG sources (DOTA, UAVid) reproduce exactly (cosine 1.0000).
  - JPEG sources (VisDrone, xView) reproduce at median cosine 0.994–0.999 under a libjpeg-turbo build of Pillow. *(Clarified 14 Oct: the cause is the JPEG decoding library, not the Pillow version. Decoding with a conda Pillow 11.1.0 built against IJG libjpeg 9, as in the original environment, reproduces every cached embedding to float precision. See `docs/paper/CLARIFICATIONS_FOR_PAPER.md`, items 9–10.)*
  - Under libjpeg-turbo, 98.7–100% of crops are nearest to their own cached row; with the original decoder every row matches, apart from three pairs of exact duplicate crops from overlapping xView chips (`results/paper_facts/exact_duplicate_crops.json`).
- **Sensitivity.** If everything were re-embedded today, reported L2 displacements would move by at most 0.009 and drift ratios by at most 0.025, all inside the image-bootstrap intervals. Full list: `manifest_report.json` → `reencode_drift_sensitivity`.

**Sequence IDs.**
- UAVid: the `seqN` file prefix, which is documented.
- VisDrone: the first field of the file name (208 distinct values over 6,471 images). *(Clarified 14 Oct: 175 of these groups behave like video clips, but 33 are sets of unrelated still images, so call them filename-prefix groups, not video sequences. See `docs/paper/CLARIFICATIONS_FOR_PAPER.md`, item 7.)*
- xView chips: no scene ID is exposed by the mirror, so overhead resampling is chip-level.

## Statistics (`drift_stats.py`)

- **Point estimates** reproduce Tables II and III to the 4th decimal; one Y_c differs by 1 in the 4th decimal because of float32 vs float64 arithmetic.
- **Permutation test:** whole source images are reassigned between domains, keeping each domain's image count. 2,000 permutations; statistic: centroid L2.
- **Bootstrap:** 2,000 replicates of source images drawn with replacement, independently per domain, with the same draw applied to every encoder (paired encoder differences). 95% percentile intervals.
- **Sequence-level sensitivity:** UAV crops resampled by group instead of image (UAVid video sequences; VisDrone filename-prefix groups); overhead resampling unchanged.
- **Centroids** are arithmetic means of unit-norm embeddings and are never re-normalized, for L2 and radii.

## Classifier (`classify_crops.py`)

- **Prompt:** `"a photo of a {label}"`, with label words `small vehicle`, `large vehicle`, `car`, `bus`, `truck`, `building`. The same prompt is used for every model and domain; nothing is tuned.
- **Text embeddings:** from each model's own text tower, unit-normalized. The fine-tuned model's text tower was frozen during fine-tuning, and its text embeddings equal CLIP's (checked).
- **Prediction:** the candidate with the highest cosine to the cached unit-norm image embedding.
- **Tasks:**
  - P1: {small vehicle, large vehicle}.
  - P2+P3 pooled: {car, bus, truck, building}.
  - The two vocabularies are never merged, and a one-candidate task raises an error.
- **Text metrics:**
  - a_c = cos(μ_c, t_c).
  - m_c = a_c − max_{k≠c} cos(μ_c, t_k), at the arithmetic centroid.
  - A_c = |a_c^overhead − a_c^UAV|.
  - Δm_c = m_c^overhead − m_c^UAV.
- **Outputs:** `results/classification/predictions.csv` holds every crop's prediction and all candidate scores (score columns follow the task's category order).

## Main results

| | CLIP | RemoteCLIP |
|---|---|---|
| P2+P3 accuracy, overhead → UAV | 32.9% → 60.2% (gap −27.3 pp, CI −31.6 to −23.3) | 27.3% → 51.8% (gap −24.6 pp, CI −30.1 to −19.3) |
| Overhead vehicles labelled… | car: 813 of 900 | building: 839 of 900 |
| Δm_c < 0 (smaller overhead margin), CI excludes 0 | 4 of 4 categories | 3 of 4 (truck includes 0) |

**P1.**
- No encoder's accuracy gap excludes zero.
- CLIP labels 98% of overhead crops "small vehicle".
- The fine-tuned model reaches 84.8% overhead and 84.2% UAV accuracy, but it was trained on this UAV task, so it is not zero-shot.

**Visual drift.**
- All 14 category–encoder displacements beat all 2,000 permutations (p < 0.001; fine-tuned large-vehicle p = 0.0003 with 20,000).
- Y_c intervals lie above 1 in all eight pooled cells.
- RemoteCLIP's ratio is lower than CLIP's only for small-vehicle and building.
- The fine-tuned small-vehicle ratio increase is not distinguishable from zero (−1.5% to +52.2%).

**Exploratory (4 categories).** Displacement ranks with A_c and |Δm_c| (Spearman 0.8 for CLIP, 0.4 for RemoteCLIP) but not with recall loss (0.02 pooled).

Full tables:
- `results/drift_stats/drift_stats_table.md`
- `results/classification/classification_bootstrap_table.md`
- `results/joined/joined_table.md`

## Limitations specific to these additions

- **One prompt.** RemoteCLIP's own evaluations use remote-sensing prompts, so its overhead "building" bias may be partly prompt-dependent. No second prompt was run.
- **Crop classification is not detection.** It uses ground-truth crops: no localization, and no background or unlisted classes.
- **The pooled UAV side mixes two datasets.** VisDrone supplies the vehicles and UAVid the buildings, so the four-way confusions on the UAV side cross datasets.
- **Unit of uncertainty.** Intervals cover resampling of the sampled crops' images (one seed, 300 per category and domain), not new samples from the full datasets.
- **Dependence the resampling can't capture.** xView chips from the same scene cannot be grouped, so overhead dependence may be under-represented.
