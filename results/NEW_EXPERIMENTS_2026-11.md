# Full-scale checks, controls and detection (November 2026)

This is the work after the RAIT submission (paper under review). There are two strands:

- **A:** re-test every paper claim on the full datasets and over many seeds.
- **B:** move from ground-truth crops to real detectors.

All runs were on 25–26 Nov 2026, on an RTX 3050 Laptop (4 GB) in the rebuilt `hf` env. Nothing in the paper's August cache was overwritten: the full-scale embeddings live in `embeddings/full_scale/`.

## Commands (hf env, from the repo root, in this order)

Set `HF_HOME="D:/C drive data/Kabir Oberoi/.cache/huggingface"` and `HF_HUB_OFFLINE=1` first. Every model is loaded from the cache by its pinned revision.

```
# A: full-scale drift
python build_full_pools.py                     # A1  ~2 min, CPU: every eligible GT box, 6/6 count checks vs the paper
python encode_full_scale.py sample             # A2  sampling manifest (<=5,000 crops per category and domain, seed 20261125)
python encode_full_scale.py encode --model clip        # A2  GPU
python encode_full_scale.py encode --model remoteclip
python encode_full_scale.py encode --model finetuned
python drift_full_scale.py                     # A3  ~47 min, CPU: FULL + 20 seeds x 300 crops, claim checks
python same_domain_baseline.py                 # A4  CPU: noise floor / same platform / cross platform
python controls_a5.py resample --model clip    # A5  GPU: UAV crops shrunk to overhead size (also remoteclip, finetuned)
python controls_a5.py prompts                  # A5  GPU: five prompt templates + ensemble
python controls_a5.py analyze                  # A5  CPU

# B: detection
python build_detection_sets.py                 # B1  COCO ground truth for 5 benchmark sets + scorer self-test
python run_detectors.py --detector owlv2       # B2  GPU
python run_detectors.py --detector gdino       # B2  GPU, ~2 h+
python eval_detection.py                       # B2/B3  ~35 min, CPU: COCO AP, found-vs-named, 500 bootstrap reps
python sam_pipeline.py propose                 # B4  GPU, ~1.5 h (SAM automatic masks on the image subset)
python sam_pipeline.py name --model clip       # B4  GPU, ~10 min (then --model remoteclip, ~1.5 h)
python sam_pipeline.py evaluate                # B4  CPU
python drift_detection_link.py owl             # B5  GPU, ~15 min: OWLv2's own embedding of every GT box
python drift_detection_link.py link            # B5  CPU
```

## New assets

| Asset | Identity |
|---|---|
| OWLv2 B/16 (ensemble) | `google/owlv2-base-patch16-ensemble` @ `cfd3195ba4ea9592eec887ded089f4c08eff231d` |
| Grounding DINO tiny | `IDEA-Research/grounding-dino-tiny` @ `a2bb814dd30d776dcf7e30523b00659f4f141c71` |
| SAM ViT-B | `facebook/sam-vit-base` @ `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f` |
| CLIP ViT-L/14 (downloaded, not yet used) | `openai/clip-vit-large-patch14` @ `32bd64288804d66eefd0ccbe215aa642df71cc41` |
| xView | `HichTala/xview` @ `00a1b977…`, now stored locally as parquet: train 20,881 / val 2,762 chips |
| DOTA parts 2–3 | extracted to `data/raw/DOTA/images_part2_part3/`; part1 stays in `images/`, so the paper's replay is unchanged |
| COCO scoring | `pycocotools==2.0.8`, the only package added (`requirements.txt`) |

All revisions are also in `results/detection/model_revisions.json`.

## A1. Full pools (`results/full_scale/pools/`)

`src/full_scale.py` lists boxes with exactly the same filters as `src/dataset.py`. Six checks reproduce the paper's August counts:

- DOTA part1;
- VisDrone held-out;
- VisDrone, all images;
- xView, first 5,000 chips;
- xView, first 500 chips for buildings;
- UAVid buildings.

| Dataset | Pool |
|---|---|
| DOTA (561 images with vehicles, parts 1–3) | small-vehicle 22,909, large-vehicle 15,212 |
| VisDrone (6,157 images) | car 109,567, van 20,534, truck 11,297, bus 5,233 (held-out: 22,368 / 4,037 / 2,260 / 1,044) |
| xView train (17,745 chips with boxes) | building 391,248, car 13,837, truck 18,782, bus 3,455, van 146 |
| UAVid train+val (269 images) | building 1,199, car 391 (mask components) |

## A2. Encoding

- A cap of 5,000 crops per category per domain, sampled at random with seed 20261125: 65,052 crops in total.
  - CLIP and RemoteCLIP encode all of them.
  - The fine-tuned CLIP encodes the 20,000 P1 crops.
- Crops shared with the paper's cache reproduce exactly for PNG sources.
  - JPEG sources are re-decoded with today's Pillow (libjpeg-turbo, not IJG libjpeg 9), giving a median cosine of about 0.994–0.997 against the August rows.
  - This shift is smaller than any effect reported here.

## A3. Paper claims at full scale and over 20 seeds (`results/full_scale/drift_full_scale.md`)

There are two designs:

- **FULL:** all cached crops, a 1,000-rep image bootstrap and a 1,000-perm image-level permutation test.
- **SEEDS:** the paper's 300-crop design, repeated with 20 seeds.

**Every qualitative claim in the paper holds at FULL scale and in at least 19 of 20 seeds.** The one at 19/20 is "fine-tuned small-vehicle ratio > CLIP".

| | Paper (300 crops) | FULL |
|---|---|---|
| P1 CLIP L2 small / large | 0.246 / 0.260 | 0.218 / 0.221 |
| P2+P3 balanced accuracy, CLIP O / U | 32.9 / 60.2 | 32.6 / 61.8 |
| P2+P3 balanced accuracy, RemoteCLIP O / U | 27.3 / 51.8 | 28.9 / 53.3 |
| Y_c (8 cells) | 1.50–2.32 | 1.75–2.20, every interval above 1 |
| Overhead vehicle crops named "car" by CLIP / "building" by RemoteCLIP | 813 / 839 of 900 | 11,956 / 12,244 of 13,455 |

**Questions the paper left open, now answered:**

- **Fine-tuned small-vehicle ratio vs CLIP:** +13.7% [+3.5, +22.6]. The paper's interval included zero; fine-tuning *raises* normalised drift for small vehicles.
- **RemoteCLIP ratio vs CLIP:**
  - car −3.9% and truck −4.7%, intervals now excluding zero;
  - building −18.7% and small-vehicle −9.1%, the large effects, unchanged.
- **P1 accuracy gap:** RemoteCLIP −4.9 pp [−8.0, −1.6], now resolved. CLIP's −4.0 pp [−9.0, +0.8] is still not resolved.
- **Van** (146 xView crops, unused in the paper): ratio 1.08 for CLIP and 1.05 for RemoteCLIP, the largest of any vehicle.

P1 drift at full scale is 10–15% below the paper's value. The paper's large-vehicle sample (0.2596) lies just above the 20-seed range (0.207–0.246), so that single draw ran high. No conclusion changes.

**Note:** in P2+P3 the FULL categories are unbalanced (bus 3,455 overhead, building 1,199 UAV), so balanced accuracy is the figure to compare with the paper.

## A4. Same-domain baseline (`results/full_scale/same_domain_baseline.md`)

| Level | Comparisons | CLIP cosine drift |
|---|---|---|
| L0 noise floor (two image sets of one dataset) | DOTA part1 vs parts 2–3; VisDrone train vs held-out; xView chip halves | ≤ 0.004 |
| L1 same platform | DOTA vs xView; VisDrone vs UAVid (car) | 0.025–0.031 |
| L2 cross platform | xView vs VisDrone / UAVid | 0.072–0.147 (2.5–5× L1) |
| L2 cross platform | DOTA vs VisDrone (P1) | 0.028–0.029 (≈ L1) |

**Reading:**

- xView-vs-drone drift (P2, P3) is far larger than switching between two datasets from the same platform.
- **P1 drift (DOTA vs VisDrone) is no larger than DOTA vs xView, so P1 cannot be attributed to the platform.**
- RemoteCLIP shows the same pattern.
- The category matches across datasets are approximate (DOTA small/large-vehicle vs xView car / bus+truck+van).

## A5. Controls (`results/full_scale/controls_a5.md`)

### (a) Resolution

- 2,000 UAV crops per category, each shrunk to the size of a random overhead crop of the same category.
  - Upsampling is never applied.
  - The short side is kept at 16 px or more, the minimum every real crop meets.
- Median shrink factors:
  - P2+P3: car 0.48, bus 0.49, truck 0.54, building 0.31;
  - P1: small-vehicle 0.74, large-vehicle 1.00.

| P2+P3, CLIP | cos O vs U | cos O vs U-shrunk | Accuracy O / U / U-shrunk |
|---|---|---|---|
| car | 0.072 | 0.023 | 35.2 / 59.1 / 46.3 (4-way, all categories) |
| building | 0.147 | 0.051 | |

**Reading:**

- **Matching the apparent object size removes about two-thirds of the P2+P3 drift** and about half of the accuracy gap.
  - RemoteCLIP falls from 48.6% to 39.7%, against 31.1% overhead.
  - On shrunk UAV crops, RemoteCLIP labels 62% "building", its overhead failure mode.
- The remaining car drift (0.023) is about the size of the same-platform drift in A4.
- In P1, shrinking changes accuracy by 1–2 pp at most.
- **Conclusion:** in P2+P3, much of "platform drift" behaves like an object-scale effect. The rest, including half the accuracy gap, is not explained by scale.

### (b) Prompts

The templates tested were:

- `a photo of a {}`
- `a satellite photo of a {}`
- `an aerial photo of a {}`
- `a drone photo of a {}`
- `{}`
- an ensemble of the five

**Reading:**

- Overhead accuracy stays below UAV accuracy for every template in P2+P3.
- RemoteCLIP depends on the prompt: "a satellite photo of a {}" raises its P2+P3 overhead accuracy from 31.1% to 44.1%, above CLIP's best template (38.0%).
  - **The paper's "RemoteCLIP is no better than CLIP overhead" holds for the paper's prompt only.**
- The fine-tuned model changes by less than 1.5 pp.
- These are plain accuracies over the full cache, not balanced ones.

## B1. Detection benchmark (`results/detection/benchmark_report.json`)

| Set | Images | GT boxes (evaluated) | Task |
|---|---|---|---|
| dota_tiles (80 part-1 scenes, 1024 px tiles) | 284 | small-vehicle 3,116, large-vehicle 986 | D1, overhead |
| visdrone_ho_D1 (held-out 20%) | 1,294 | small-vehicle 22,368, large-vehicle 7,341 | D1, UAV |
| xview_val | 2,762 | car 1,475, bus 763, truck 2,101, building 38,658 | D2, overhead |
| visdrone_ho_D2 | 1,294 | car 22,368, bus 1,044, truck 2,260 | D2, UAV |
| uavid_tv (train+val, 2×2 tiles) | 1,080 | car 364, building 915 | D2, UAV |

Ignore regions (iscrowd) are boxes that score neither a hit nor a false alarm:

- any box under 16 px;
- a box less than 70% inside its tile;
- DOTA "difficult";
- VisDrone class 0;
- vans in D2.

The ground-truth JSONs are in `data/processed/detection/gt/` (git-ignored, about 44 MB). `src/detection_eval.py` scores with pycocotools (maxDets 10/100/500) and passes a synthetic self-test.

## B2/B3. OWLv2 and Grounding DINO, zero-shot (`results/detection/detection_results.md`)

- Both detectors use the same prompt words in both domains, with no tuning.
- OWLv2 uses "a photo of a {}"; Grounding DINO uses the bare class words.
- Settings: batch 1, peak VRAM about 2 GB.

| AP50, overhead vs UAV | OWLv2 | Grounding DINO |
|---|---|---|
| small-vehicle | 36.9 vs 58.6 | 18.4 vs 55.8 |
| large-vehicle | 20.6 vs 27.1 | 1.4 vs 23.0 |
| car | 10.3 vs 78.5 | 5.7 vs 75.5 |
| bus | 6.5 vs 52.7 | 0.9 vs 24.5 |
| truck | 8.1 vs 38.8 | 1.0 vs 27.8 |
| building | 18.8 vs 33.3 | 10.8 vs 35.0 |

**Found vs named.**

- *Found* means a true box is overlapped at IoU 0.5 or more by a detection with score 0.1 or more.
- *Named* means that, of the found boxes, the best detection has the right class.

**Reading:**

- **Overhead loss is mostly a finding loss.**
  - OWLv2 finds 61% of xView cars vs 89% of VisDrone cars, and 29% of xView buildings vs 86% of UAVid buildings.
  - Once found, cars and buildings are named correctly 90% or more of the time.
- Bus and truck also lose on naming overhead:
  - OWLv2 46% / 57%;
  - Grounding DINO 16% / 23%.
- Large-vehicle naming is poor in both domains.
- Every overhead-vs-UAV difference in finding rate has a 95% image-bootstrap interval excluding zero.

## B4. SAM + CLIP / RemoteCLIP (`results/detection/sam_pipeline_results.md`)

**Setup:**

- SAM ViT-B automatic masks (32×32 grid, pred_iou 0.88, stability 0.92) produce boxes, at most 300 per image.
- Each box crop is named with "a photo of a {}" against the task's classes plus five background prompts. A proposal is dropped if a background prompt wins.
- Scored on a seeded subset (DOTA 120, VisDrone 150, xView 300, UAVid 150 images), with OWLv2 and Grounding DINO scored on the same images.
- Peak VRAM was 4.4 GiB on VisDrone with points_per_batch 64. Lower it to 16 for a re-run.

| Mean AP50 | SAM + CLIP | SAM + RemoteCLIP | OWLv2 | G-DINO |
|---|---|---|---|---|
| dota_tiles | 7.4 | 5.1 | 25.0 | 9.0 |
| visdrone_ho_D1 | 16.3 | 16.1 | 43.0 | 38.4 |
| xview_val | 0.5 | 1.2 | 10.9 | 4.6 |
| visdrone_ho_D2 | 18.4 | 21.0 | 54.4 | 41.5 |
| uavid_tv | 19.0 | 10.7 | 29.5 | 34.3 |

**Reading:**

- **SAM's proposals are not the bottleneck.** They cover about 55% of GT boxes at IoU 0.5 in every set, overhead and UAV alike (xView: 41.6% of cars, whose median short side is 17 px).
- **The naming step is the bottleneck.**
  - On xView, CLIP and RemoteCLIP send about 97% of the proposals to a background prompt.
  - Only 4–5% of xView objects survive, against 39–44% on VisDrone from the same ~55% coverage.
- This is the paper's text misalignment (Δm_c < 0) appearing as missed detections.

## B5. Drift in OWLv2's space; drift vs detection loss (`results/detection/drift_vs_detection.md`)

**OWLv2 space:**

- For 400 seeded images per set, each true box is matched to the patch whose predicted box overlaps it most, and that patch's class embedding is kept (L2-normalised).
- At most 3,000 boxes are kept per category per domain.
- **Drift in OWLv2's space:**
  - cosine 0.076–0.160, normalised ratio 0.59–0.80, all intervals well above zero;
  - Δm < 0 for 5 of 6 categories (large-vehicle +0.007);
  - Δm is largest for building (−0.063) and car (−0.044).
- **Best-patch IoU ≥ 0.5 for 88–100% of true boxes in both domains.**
  - So OWLv2 usually has a well-placed box for an overhead object but scores it too low.
  - That is the same story as B4.
- Ignored boxes (<16 px) are excluded, which leaves only 168 xView cars in the sample.

**Link:** the six categories, with CLIP / RemoteCLIP / OWLv2-space drift measures against each detector's AP50, found and named drops.

- **No Spearman correlation is significant.** With n = 6, p < 0.05 needs |ρ| ≥ 0.89; the largest observed is 0.77.
- Raw drift magnitude has no consistent sign.
- The most consistent signal is the text margin: a more negative Δm goes with a larger *naming* drop for all three encoders and both detectors (ρ −0.49 to −0.77).
- This matches the paper: drift size does not predict loss, but text misalignment tracks it. It is descriptive only.

## Figures (`results/figures_2026-11/`, made by `python scripts/plots/make_november_figures.py`)

Each PNG has a CSV beside it holding the plotted numbers.

| Figure | Shows |
|---|---|
| `fig1_detection_ap50.png` | AP50 overhead vs UAV per category, OWLv2 and Grounding DINO (B2) |
| `fig2_same_domain_baseline.png` | CLIP cosine drift at three levels of difference, with 95% CIs (A4) |
| `fig3_resolution_control.png` | Drift before and after shrinking UAV crops to overhead size (A5) |
| `fig4_sam_found_vs_proposed.png` | SAM proposal coverage vs objects still found after naming (B4) |

## What changes for the paper's claims

| Paper statement | After November |
|---|---|
| Drift is real, larger than between-category spacing, RemoteCLIP raw drift lower | Holds at full scale, 20/20 seeds |
| RemoteCLIP normalised drift lower only for small-vehicle and building | Also slightly lower for car and truck; the large effects are unchanged |
| Fine-tuning: small-vehicle unresolved | Resolved: normalised drift *rises* by 13.7% |
| Overhead recognition is much worse (four-way) | Holds at full scale and with every prompt; also holds for real detectors (2–10× lower AP50) |
| RemoteCLIP is no better than CLIP overhead | Only with "a photo of a {}"; with a satellite prompt it is better |
| Drift size does not predict accuracy loss | Also holds for detection loss (n = 6) |
| Drift is a platform effect | **Qualify:** P1 drift equals same-platform drift (A4); much of P2+P3 drift is object scale (A5) |

## Not done / next

- **Same-backbone encoder:** CLIP ViT-L/14 is downloaded but not yet encoded.
- **Altitude within one dataset (UAVDT):** needs a download.
- **Fine-tuned model for P2/P3.**
- **Distribution-level (shape) measures.**
- **A SAM re-run with smaller point batches,** and SAM on the full sets rather than the subset.
