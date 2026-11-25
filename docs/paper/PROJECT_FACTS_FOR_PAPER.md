# Project facts for the paper (evidence only)

Collected 14 October 2026 (sections on crop identity, JPEG decoding, CLIP weight files and VisDrone grouping updated later the same day after direct tests) from the project at `D:\C drive data\Kabir Oberoi\Desktop\Capstone` and the pre-reset records that survived on this machine. Nothing in the manuscript was edited. No cache, checkpoint or existing result was overwritten, and nothing was retrained.

**How to read the statuses**

| Status | Meaning |
|---|---|
| Verified from run/output records | A file written by the run, a logged command with its output, or a value recomputed from saved outputs |
| Present in code/configuration; execution not established | The code allows or implies it, but no record shows it happened |
| Conflicting records | Two records disagree; both are given |
| Not located in the files checked | Searched for and not found; the places searched are listed |

**Evidence files referred to below**

| Short name | Path |
|---|---|
| Records | `results/paper_facts/original_environment_records.txt` — verbatim excerpts, with source line numbers, from the two pre-reset Claude Code session transcripts |
| Transcript A | `C:\Users\user\.claude\projects\d--C-drive-data-Kabir-Oberoi-Desktop-Capstone\4aa508a9-a91c-4fc0-9978-d51009196897.jsonl` (13 Aug – 19 Sep 2026) |
| Transcript B | `D:\C drive data\Kabir Oberoi\.claude\projects\c--Users-Kabir-Oberoi-Desktop-Capstone\e6103a86-523d-4a30-8d40-ff97f0ca724e.jsonl` (19 – 21 Sep 2026) |
| Facts | `results/paper_facts/facts.json` — written by `collect_paper_facts.py` (read-only inspection) |
| Manifests | `results/crop_manifests/*.csv` and `manifest_report.json` — written by `build_crop_manifests.py` |
| Drift stats | `results/drift_stats/drift_stats.json` — written by `drift_stats.py` |
| Classification | `results/classification/classification_bootstrap.json`, `predictions.csv` — written by `classify_crops.py --bootstrap` |
| Joined | `results/joined/joined_table.csv`, `association.json` — written by `join_results.py` |
| Identity | `results/paper_facts/crop_identity_conda_pillow_11.1.0_jpeg9.json` and `crop_identity_pip_pillow_11.1.0.json` (with `.log` files) — written by `verify_crop_identity.py` |
| Weight files | `results/paper_facts/clip_weight_files.json` |
| Prefix groups | `results/paper_facts/visdrone_prefix_groups.json` |
| Duplicates | `results/paper_facts/exact_duplicate_crops.json` |

Transcript timestamps are UTC. Log lines inside them are local time (IST = UTC + 5:30).

---

## A. Fact table

### A1. Software versions

| Requested fact | Finding | Evidence | Status |
|---|---|---|---|
| Interpreter that ran the embedding jobs | `C:\anaconda\envs\hf\python.exe` ran `run_remoteclip_vs_clip.py` (P1 CLIP and RemoteCLIP), `run_xview_vs_visdrone.py` (P2) and `run_uavid_building.py` (P3) on 20 Aug 2026. The log end times (22:06, 22:23, 23:04 IST) equal the modification times of the cached `.npy` files. | Records §2 (Transcript A lines 1699, 1737, 2296 and monitor events); file times in `embeddings/` | Verified from run/output records |
| Interpreter for the fine-tuned CLIP embeddings | Notebook `04_finetune_visdrone_clip.ipynb` was run interactively in VS Code. Its saved outputs show `c:\anaconda\envs\hf\lib\site-packages\...` and kernel Python 3.9.25. A process listing at 16:18 UTC on 20 Aug shows the kernel as `C:\anaconda\envs\hf\python.exe`. Cache files are timed 21:36 IST on 20 Aug. | Notebook metadata and cell 7 output; Transcript A lines 1560–1568 | Verified from run/output records |
| Python | 3.9.25 | Notebook `language_info.version` in notebooks 01, 03, 04; Records §1 (Transcript B line 164) | Verified from run/output records |
| PyTorch | 2.5.1 (the version string has no `+cu118` suffix) | Records §1: Transcript A line 195 (13 Aug), line 366 (13 Aug), Transcript B line 164 (19 Sep) | Verified from run/output records |
| CUDA | 11.8, `cuda available True`, NVIDIA GeForce RTX 3050 Laptop GPU, 4.0 GiB | Records §1 (Transcript B line 164). Also `PROJECT_CONTEXT.md` lines 429–439. | Verified from run/output records (printed 19 Sep; see note 1) |
| torchvision | 0.20.1 | Records §1 (Transcript B line 164) | Verified from run/output records (printed 19 Sep; see note 1) |
| Transformers | 4.49.0. It was 4.57.3 until 13:20 UTC on 13 Aug, when `pip install "transformers==4.49.0"` was run; a load test at 13:29 UTC printed `transformers 4.49.0`. | Records §1: Transcript A lines 195, 328, 366; Transcript B line 164 | Verified from run/output records |
| OpenCLIP (`open_clip`) | 3.3.0 | Records §1: Transcript A line 235 (13 Aug); Transcript B line 164 (19 Sep) | Verified from run/output records |
| Pillow | 11.1.0 | Records §1: Transcript A line 195 (13 Aug); Transcript B line 164 (19 Sep) | Verified from run/output records |
| JPEG decoding library behind Pillow | IJG libjpeg 9 (a conda build of Pillow), not libjpeg-turbo. The build was not printed in August; it is established by test: a conda-built Pillow 11.1.0 (linked to `jpeg 9f`, no libjpeg-turbo) reproduces every cached embedding of every JPEG-sourced crop to float precision, while the pip wheels of Pillow 11.1.0 and 11.3.0 (libjpeg-turbo 3.1.0 and 3.1.1) both do not and decode identically to each other. Conda's `pillow 11.1.0` builds declare the dependency `jpeg >=9e,<10a`. Whether the original library was 9e or 9f is not recorded. | Identity; `conda search pillow=11.1.0 --info` | Verified from run/output records (by reproduction test) |
| NumPy | 1.26.4 | Records §1: Transcript A line 195; Transcript B line 164 | Verified from run/output records |
| scikit-learn | 1.3.2 | Records §1: Transcript A lines 195, 377; Transcript B line 164 | Verified from run/output records |
| Matplotlib | 3.9.1 | Records §1: Transcript A line 195; Transcript B line 164 | Verified from run/output records |
| SciPy, datasets | SciPy 1.13.1 (printed 20 Aug, 17:27 UTC, five minutes before the UAVid run that uses it). `datasets` 4.4.1 (printed 19 Aug). | Records §1: Transcript A lines 2240, 1196; Transcript B line 164 | Verified from run/output records |
| Link between these versions and the 20 Aug runs | The same interpreter printed identical versions before the runs (13 Aug) and after them (19 Sep) for every package printed on both dates: torch, Pillow, scikit-learn, Matplotlib, NumPy, tqdm, open_clip. The only installs logged between those dates are `nbconvert nbclient nbformat` and `transformers==4.49.0` (13 Aug), `kaggle` (20 Aug), and `pymupdf`, `python-pptx` (21 Aug, after the runs). | Records §1; Transcript A lines 241, 328, 2188, 2608, 2753 | Verified from run/output records (see note 1) |
| Environment for the October analyses (manifests, permutation tests, bootstrap, classifier, joins) | A rebuilt environment at `C:\Users\user\anaconda3\envs\hf`: Python 3.9.25, torch 2.5.1+cu118, torchvision 0.20.1+cu118, transformers 4.49.0, open_clip_torch 3.3.0, numpy 1.26.4, scikit-learn 1.3.2, scipy 1.13.1, **Pillow 11.3.0**, **matplotlib 3.9.4**, datasets 4.5.0. | `requirements.txt` (written 4 Oct from that environment); output files dated 12–14 Oct | Verified from run/output records |
| Which record the first manuscript draft's version list came from | Draft 1 listed "Pillow 11.3.0, Matplotlib 3.9.4". Those are the October rebuild. The original environment had Pillow 11.1.0 and Matplotlib 3.9.1. | `requirements.txt` vs Records §1 | Conflicting records (original-run records are the ones above) |
| Manuscript statement "the complete original preprocessing environment is not recovered" | The original versions are recorded in the two session transcripts. That statement was based on an earlier search that had not examined the transcripts. | Records §1 | Conflicting records (superseded by the records located) |
| Full package list (`pip freeze`) of the original environment | Only a count was logged: `pip list --format=freeze` gave 137 lines on 19 Sep. The list itself was not saved. | Transcript B line 385 | Not located in the files checked (see "Where I searched") |

Note 1. torchvision and the CUDA build were printed only on 19 Sep, after the runs. No install, upgrade or reinstall of torch or torchvision appears in either transcript between 13 Aug and 19 Sep, and `PROJECT_CONTEXT.md` (committed 13 Aug) already states "PyTorch 2.5.1, CUDA: 11.8".

### A2. Checkpoints and preprocessing

| Requested fact | Finding | Evidence | Status |
|---|---|---|---|
| CLIP ViT-B/32 identifier and revision | Hugging Face `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. The pre-reset cache's `refs/main` holds this revision, with files downloaded 13 Aug 2026. `pytorch_model.bin` sha256 `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f`; identical in today's cache. | Facts → `clip_cache_original`, `clip_cache_current` | Verified from run/output records |
| Which CLIP weight file was loaded | Under Transformers 4.49.0 (the original version), `CLIPModel.from_pretrained("openai/clip-vit-base-patch32")` logs "loading weights file pytorch_model.bin from cache at …\snapshots\3d74acf9…\pytorch_model.bin", followed by "Attempting to create safetensors variant / Safetensors PR exists". That background check is what downloads the second snapshot `c237dc49a33fc61debc9276459120b7eac67e7ef` holding `model.safetensors`; the same thing happened in the original cache on 13 Aug and again in today's cache. The August run itself did not log the file name. | Weight files → `from_pretrained_log` | Verified from run/output records (behaviour of the same library version on the same cache layout) |
| Do the two CLIP weight files hold the same weights? | Yes. `pytorch_model.bin` (revision `3d74acf…`) and `model.safetensors` (snapshot `c237dc49…`, sha256 `99d28a652e6ec46629ab7047a0ac82c69b1fe11e0ce672c43af65d3a9a3fc05d`) each contain 400 tensors with identical names; all 400 are bit-identical (151,277,440 values compared, maximum absolute difference 0.0). The model loaded by `from_pretrained` equals the `.bin` tensors for every parameter. | Weight files | Verified from run/output records |
| Weights and preprocessing reproduce the cache | Yes, exactly. With the conda-built Pillow 11.1.0 decoder, all 9,050 crop–encoder pairs (every crop of every list under every encoder) reproduce their cached row at cosine > 0.999999; the worst is 0.99999975 and the largest element-wise difference is 1.8e-06 (float32 rounding). With pip-wheel Pillow (libjpeg-turbo), PNG sources (DOTA, UAVid) are still exact and JPEG sources (VisDrone, xView) give median cosine 0.9938–0.9992. | Identity; Manifests → `manifest_report.json` | Verified from run/output records |
| RemoteCLIP ViT-L/14 checkpoint | Local file `models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt`, 1,710,613,765 bytes, dated 28 Apr 2026, sha256 `fcc2a7e21e171f4ffcb7a9c0206b8b74ac0c9eb83c67b576958b7a4ed6c8cecb`. Size and sha256 are identical to `RemoteCLIP-ViT-L-14.pt` in the official Hugging Face repository `chendelong/RemoteCLIP` (repository revision `bf1d8a3ccf2ddbf7c875705e46373bfe542bce38`, queried 14 Oct 2026). | Facts → `local_checkpoints`, `remoteclip_official_release` | Verified from run/output records (file identity). The download source used in April is not recorded. |
| RemoteCLIP model construction | `open_clip.create_model_and_transforms("ViT-L-14", pretrained=None)`, then `load_state_dict(torch.load(path), strict=False)`; the code raises if any key is missing or unexpected. Embeddings are `model.encode_image(...)`, 768-d. | `src/embeddings.py` lines 8–17, 38–39; run log line "WARNING:root:No pretrained weights loaded for model 'ViT-L-14'" in Records §2 | Verified from run/output records |
| Fine-tuned CLIP checkpoint | `models/finetuned_clip_visdrone_heldout/finetuned_clip_visdrone_heldout_epoch3.pt`, sha256 `3c150faac15620b4edce39abfe50d4191f029df85e45ad4a03f17fb10be0ba2a`, written 20 Aug 2026 21:24 IST. Notebook 04 output: 5,177 train-pool and 1,294 held-out images; 167,699 training crops; trainable parameters 14,570,496 of 151,277,313; three epochs, average loss 1.4800, 1.4040, 1.3525. | Facts → `local_checkpoints`; notebook 04 cells 3, 5, 7 | Verified from run/output records |
| Fine-tuned model's text tower | Frozen during fine-tuning. Its text embeddings for the classifier prompts equal CLIP's to within 1e-5. | `src/embeddings.py` lines 128–131, 142–143; Classification → `checks` | Verified from run/output records |
| Input resolution | 224 × 224 for all three encoders | Facts → `clip_image_processor_effective`, `remoteclip_open_clip` | Verified from run/output records |
| Resize, crop, interpolation, normalisation | See section C. Both pipelines resize the shortest side to 224 with bicubic interpolation and then centre-crop 224 × 224. | Facts (same keys) | Verified from run/output records |
| Colour conversion | Every source image is opened with `Image.open(...).convert("RGB")` before cropping; xView chips arrive decoded from the dataset. Both processors also convert to RGB. | `src/dataset.py` lines 113, 146, 197–198 | Present in code/configuration (it is the code that produced the logged runs) |
| Padding or crop margins | None. Crops are cut exactly at the annotation rectangle. Because of the centre crop, the parts of an elongated crop outside the central square are not seen by the encoder (aspect-ratio figures in D1). | `src/dataset.py` lines 123, 168, 213, 259; section C | Verified from run/output records (boxes in Manifests) |
| Actual processor configuration | Printed in section C from the loaded objects under transformers 4.49.0, open_clip 3.3.0 and torchvision 0.20.1, the same three versions as the original environment. | Facts | Verified from run/output records |

### A3. Crop and cache correspondence

| Requested fact | Finding | Evidence | Status |
|---|---|---|---|
| Embeddings ↔ crop IDs, labels, source images, boxes | One CSV per crop list gives, for row *i* of each cached `*_feats.npy`: crop ID, label, dataset, source image, sequence (where one exists), box and crop size. | Manifests | Verified from run/output records |
| How the correspondence was established | The original runs did not save provenance. In October the extraction was replayed with the logged settings and checked four ways: (1) the per-class crop counts before sampling equal the counts printed by the August runs, for all six lists; (2) the replayed label sequence equals the cached `*_labels.npy` for every list and encoder; (3) every crop was re-encoded and compared with all cached rows; (4) the optimal one-to-one matching between re-encoded crops and cached rows was computed. | `build_crop_manifests.py`; `verify_crop_identity.py`; Identity | Verified from run/output records |
| Row-identity result | Every row is identified. With the original JPEG decoder (conda-built Pillow 11.1.0), each of the 3,925 crops reproduces its own cached row to float precision under every encoder (9,050 crop–encoder pairs; cosine > 0.999999 for all). A control that deliberately misaligns rows within each label gives median cosine 0.70–0.97. The earlier ambiguous cases (for example 592 of 600 nearest to their own row for fine-tuned CLIP on P1 VisDrone) occurred only under the pip-wheel decoder; under the original decoder that case is 600 of 600. Table in D2. | Identity | Verified from run/output records |
| Rows not nearest to their own row under the original decoder | Six crops, forming three pairs of exact duplicates: the same object cut from two overlapping xView chips, with identical crop size and identical embeddings (cosine 1.000000 between the two cached rows). P2 xView bus: rows 180 and 357 (chips 109 and 105), rows 804 and 860 (chips 4821 and 4831). P3 xView building: rows 94 and 210 (chips 110 and 106). For these, "nearest row" is a tie between two identical rows. No other list contains an exact duplicate. | Identity → `rows_not_nearest_to_own`; Duplicates | Verified from run/output records |
| Replayed crop counts equal the original logs | Equal for all six lists (`raw_counts_equal_original_log` is true for each). Original logs: DOTA raw crops 9,570 (large-vehicle 3,033; small-vehicle 6,537); VisDrone held-out raw crops 29,709 (small 22,368; large 7,341); xView raw crops 5,891 (car 1,269; truck 3,694; bus 903; van 25); VisDrone all-image raw crops 146,631 (car 109,567; van 20,534; truck 11,297; bus 5,233); UAVid raw building components 1,199 (872 from 200 train images, 327 from 70 validation images); xView raw building crops 3,711 from 500 chips. | Records §2; Identity → `raw_counts_replayed` | Verified from run/output records |
| Sampling seeds | Crop sampling seed 42 with 300 crops per class (`subsample_balanced`). VisDrone image split seed 1234, hold-out fraction 0.2. | `results/remoteclip_vs_clip/drift_comparison_3way_heldout.json`; `results/xview_vs_visdrone/drift_comparison_4class.json`; notebook 04 | Verified from run/output records |
| Same evaluation crops across encoders | Within each task and domain, crop IDs and labels are identical and in the same order for every encoder. | Drift stats → `checks`; Classification → `checks` | Verified from run/output records |
| Fine-tuned CLIP hold-out | P1 UAV crops come only from the 1,294 held-out VisDrone images (435 of them supply the 600 sampled crops). The fine-tuned embeddings were produced by notebook 04 and the CLIP and RemoteCLIP embeddings by `run_remoteclip_vs_clip.py`; both list "VisDrone eval crops (held-out pool only): 600" and the manifests match all three. | Notebook 04 cells 3, 9; Records §2; Manifests | Verified from run/output records |
| xView mapping used for P2 | The cached xView labels contain car, truck, bus and van (25) and no building, so P2 was run before "Building" was added to the mapping. P2 VisDrone labels contain 300 van crops that are not evaluated. | `embeddings/xview_vs_visdrone/*_labels.npy`; Records §2 | Verified from run/output records |

### A4. Crop extraction

| Requested fact | Finding | Evidence | Status |
|---|---|---|---|
| VisDrone filter and clipping order | The 16-pixel test is applied to the annotated width and height first; the box is clipped to the image afterwards; there is no size test after clipping. Class 0 (ignored regions) and unmapped classes are dropped. The score, truncation and occlusion fields are not read. | `src/dataset.py` lines 155–169 | Present in code/configuration (the code that produced the logged runs) |
| Do evaluated VisDrone crops fall below 16 pixels? | No. Among the evaluated crops, none has a side below 16: P1 VisDrone 0 of 600 (minimum width 16, minimum height 16); P2 VisDrone car/bus/truck 0 of 900 (minimum 16 and 16). | Facts → `evaluated_crop_sizes` | Verified from run/output records |
| Other datasets, same question | DOTA and xView apply the test after clipping. Observed minimum side 16 in P1 DOTA, P2 xView and P3 xView; UAVid minimum width 32, minimum height 18. No evaluated crop anywhere has a side below 16. | Facts → `evaluated_crop_sizes`; `src/dataset.py` lines 119–121, 255–257 | Verified from run/output records |
| DOTA oriented boxes | Converted to the axis-aligned rectangle of the minimum and maximum corner coordinates, clipped to the image, then size-tested. | `src/dataset.py` lines 85–100, 119–123 | Present in code/configuration |
| UAVid building extraction | For each label image, pixels equal to RGB (128, 0, 0) form a mask. `scipy.ndimage.label` with its default structuring element (4-connectivity) finds connected components. A component is kept if it has at least 800 pixels and its bounding rectangle is at least 16 pixels in each direction. The crop is that bounding rectangle, so it can include non-building pixels. The rectangle's right and bottom edges are the maximum pixel index, which PIL treats as exclusive, so each crop omits its last column and row. | `src/dataset.py` lines 174–218; Facts | Present in code/configuration (the code that produced the logged runs) |
| UAVid instance verification | None exists in the code. No step checks that a component is a single building. No record of manual inspection was found. | `src/dataset.py`; Transcripts A and B | Not located in the files checked |
| UAVid components actually sampled | 1,199 raw components (logged); 300 sampled from 164 images and 27 sequences; at most 9 crops from one image. | Records §2; Facts → `evaluated_crop_sizes.p3_uavid_building` | Verified from run/output records |
| DOTA platform/sensor metadata | Each DOTA label file begins with `imagesource:` and `gsd:` lines. All 149 sampled images, and all 469 local images, have `imagesource:GoogleEarth`. GSD is present for all 149: 148 values lie between 0.107 and 0.508 m/pixel (median 0.131); one file (`P0256`) has `gsd:1.34110450745e-06`. | Facts → `dota_metadata`; `data/raw/DOTA/labelTxt/*.txt` lines 1–2 | Verified from run/output records |
| DOTA release | 1,411 label files and 469 images are present locally; no label file contains the class `container-crane`, and the classes seen are the 15 DOTA-v1.0 classes. No file states the release name. | `data/raw/DOTA/labelTxt` | Present in code/configuration; execution not established (release name is an inference from class list and counts) |

### A5. Grouping and statistical runs

| Requested fact | Finding | Evidence | Status |
|---|---|---|---|
| UAV "sequence" IDs: VisDrone | The grouping key is the first underscore-separated field of the file name (e.g. `0000002` in `0000002_00005_d_0000014.jpg`); 208 distinct values over the 6,471 training images. It is **not** a video ID for every group. 175 groups (2,105 images) have a varying second field (frame numbers such as 1, 460, 571, 1001…) and their images resemble one another (mean 16×16 grey-thumbnail correlation 0.35 within a group, 0.38 for adjacent frames, against 0.10 between groups): these behave like video clips. 33 groups (4,366 images; prefixes 9999937–9999997; 13 to 434 images each) have the second field equal to `00000` for every image and show no within-group resemblance (0.10, the same as between groups): these are sets of still images, not videos. The 33 still-image groups supply 374 of the 600 evaluated P1 VisDrone crops and 646 of the 900 evaluated P2 VisDrone crops. No dataset document describing the naming was found. | `src/dataset.py` lines 166–169; Prefix groups | Verified from run/output records (what the groups contain). Not located: any dataset documentation of the naming. |
| What the "sequence-level" sensitivity analysis therefore is | For VisDrone it resamples filename-prefix groups, a grouping coarser than the image (98 groups in P1, of which 65 are clip-like and 33 are still-image sets). For UAVid it resamples documented video sequences. It is a coarser-cluster sensitivity analysis; it is video-sequence resampling only for UAVid and for the clip-like VisDrone groups. | Prefix groups; A5 rows below | Verified from run/output records |
| UAV sequence IDs: UAVid | The `seqN` prefix of the file name. The dataset readme states that the data consist of sequences and that images are named by frame index. | `src/dataset.py` line 214; `data/raw/UAVid/modified_uavid_dataset/readme.txt` | Verified from run/output records |
| xView source-scene grouping | Not available and not used. The mirror `HichTala/xview` (revision `00a1b977dc16f6eda77cce0236ca2c2a13ac840b`, the same revision as in the pre-reset cache) exposes `image_id`, `image`, `width`, `height` and `objects{bbox_id, category, bbox, area}` only. Overhead resampling is by chip (`image_id`). | Facts → `xview_mirror` | Verified from run/output records |
| Permutation unit | Whole source images. For one category and encoder, the images contributing crops in either domain are pooled; images are randomly reassigned so that the number of "overhead" images is unchanged. | `drift_stats.py` lines 72–94 | Verified from run/output records |
| Permutation null and statistic | Null: the category's images are exchangeable between the two domains. Statistic: L2 distance between the two centroids (arithmetic means of unit-norm embeddings). | `drift_stats.py` lines 72–94 | Verified from run/output records |
| Permutation replicate count and p-value | 2,000 permutations per cell. p = (1 + number of permuted statistics ≥ observed) / (2,000 + 1). All 14 cells returned 0.0004998 (= 1/2001), the smallest attainable value. | `drift_stats.py` line 91; Drift stats → `tasks.*.permutation` | Verified from run/output records |
| Extended permutation run | Fine-tuned CLIP, large-vehicle: 20,000 permutations, seed 7, p = 0.0003; observed L2 0.09545; null 99.9th percentile 0.08554; null maximum 0.09904. | Drift stats → `extra_permutation` | Verified from run/output records |
| Bootstrap unit | Source images. Each replicate draws as many images as the domain has, with replacement; each crop is weighted by the number of times its image was drawn. | `src/stats.py` lines 44–49 | Verified from run/output records |
| Bootstrap replicate count, seed, interval | 2,000 replicates; 95% percentile interval (2.5th and 97.5th percentiles). Seeds: 20261012 for the drift statistics (one generator used in sequence for the permutation tests and both bootstraps); 20261013 for the classification bootstrap (re-seeded for each task). | `drift_stats.py` lines 30–32, 99; `classify_crops.py` lines 45–46, 225; `src/stats.py` lines 52–55; Drift stats → `settings` | Verified from run/output records |
| Pairing across encoders | In each replicate the same image draw is applied to every encoder, so encoder differences are computed within a replicate. | `drift_stats.py` lines 139–153; `classify_crops.py` lines 230–232, 270 | Verified from run/output records |
| Independence between domains | The overhead and UAV draws are separate calls to the generator in each replicate. | `drift_stats.py` line 140; `classify_crops.py` line 231 | Verified from run/output records |
| Replicates missing a category | Drift bootstrap: a replicate in which a category receives zero weight in either domain is excluded for that category. Observed: all 2,000 replicates were valid for every category, encoder and resampling level. Classification bootstrap: missing categories are not counted separately. | `drift_stats.py` lines 57–60, 146–154; Facts → `drift_valid_replicates_min` = 2000 | Verified from run/output records (drift); Present in code only (classification) |
| Sequence-level sensitivity analysis | Completed for both the drift statistics and the classification results. UAV crops are resampled by sequence (P1: 98 sequences; P2+P3: 134); overhead crops stay image-level. Sequence-level intervals are wider: the ratio of sequence-level to image-level interval width has median 1.09 (range 0.96 to 1.94) over the 100 drift intervals and median 1.16 (range 0.98 to 1.75) over the 41 classification intervals. No conclusion changes: every interval that excludes zero at image level also excludes zero at sequence level and vice versa, and every Y_c lower bound stays above 1. Full comparison in D4 and D5. | Drift stats → `tasks.*.bootstrap.sequence`; Classification → `tasks.*.bootstrap.sequence` | Verified from run/output records |

### A6. Additional completed evaluations

| Requested fact | Finding | Evidence | Status |
|---|---|---|---|
| Alternative prompts | Not run. The only prompt in the code is `"a photo of a {}"`; no result file for another prompt exists. | `classify_crops.py` line 38; `results/` | Not located in the files checked |
| Additional sampling seeds | `verify_drift_seeds.py` exists (seeds 42, 123, 7, 2024, 99). It was started once, on 19 Aug 2026, against the earlier non-held-out checkpoint. Its log shows "--- seed 42 ---" at 18:23 and "--- seed 123 ---" at 18:43; the monitor's 30-minute limit then ended it. It writes its output only at the end, so no file and no per-seed value exists. It was not run again. | Records §2 (Transcript A lines 1393–1410); Transcript A lines 1424–1426, 3771; no `seed_sensitivity.json` on disk | Present in code/configuration; execution not established (started, not completed) |
| Independent samples | Not run. All reported values use the single seed-42 sample. | `results/` | Not located in the files checked |
| Full-dataset runs | Not run. `results/full_dataset/` contains only `.gitkeep`; notebook `02_full_dataset.ipynb` has no cells. | Repository | Not located in the files checked |
| Detector evaluations | Not run. No detector code or result is in the repository. On 19 Sep the original environment reported `segment_anything`, `ultralytics` and `groundingdino` as not installed. | Repository; Records §1 (Transcript B line 164) | Not located in the files checked |
| Paired differences between encoder accuracy gaps | Completed. Pooled task, RemoteCLIP gap minus CLIP gap: point +2.75 percentage points; image-level interval −3.45 to +8.66; sequence-level interval −4.27 to +10.13. P1 values in D6. | Classification → `tasks.*.bootstrap.*.remoteclip.accuracy_gap_diff_vs_clip_pp`; Facts → `accuracy_gaps` | Verified from run/output records |
| Paired per-domain accuracy difference between encoders | Completed on 14 Oct from the saved predictions (4,000 image-bootstrap replicates, seed 11). Pooled task, RemoteCLIP minus CLIP: overhead −5.67 points, interval −10.89 to −0.58; UAV −8.42 points, interval −11.62 to −5.31. | Facts → `paired_encoder_accuracy_difference` | Verified from run/output records |

### A7. Correlations

| Requested fact | Finding | Evidence | Status |
|---|---|---|---|
| The pooled Spearman value "0.02" | Exact value 0.0238095. It is the Spearman rank correlation, over 8 rows (4 pooled categories × 2 encoders: CLIP and RemoteCLIP), between the cosine centroid displacement and the **absolute** per-category recall gap, where recall gap = overhead recall − UAV recall in percentage points. | `join_results.py` line 102; Joined → `association.json` → `pooled_cos.recall_gap_pp`; rows in D7 | Verified from run/output records |
| Sign convention | Absolute value of (overhead − UAV). With the signed gap instead, the pooled value is 0.000 (and 0.000 for UAV − overhead). | Facts → `correlations.pooled_8_points` | Verified from run/output records |
| Tie handling | `scipy.stats.spearmanr`, which assigns average ranks to ties. There are no ties in either variable among the 8 rows. | Facts → `correlations.ties_present`, ranks | Verified from run/output records |
| What the pooled value mixes | It ranks values from two encoders together. Cosine displacement is on a different scale for each encoder (RemoteCLIP's values are lower in every category). | D7 | Verified from run/output records |
| Within-encoder correlations with A_c and \|Δm_c\| | Cosine displacement vs A_c: CLIP 0.8, RemoteCLIP 0.4. Cosine displacement vs \|Δm_c\|: CLIP 0.8, RemoteCLIP 0.4. Each is over 4 categories. | Joined → `association.json` → `per_model`; Facts → `correlations.within_encoder` | Verified from run/output records |
| Within-encoder correlations with the recall gap | Cosine displacement vs \|recall gap\|: CLIP 0.4, RemoteCLIP −0.2. With the signed gap: CLIP −0.4, RemoteCLIP 0.2. | Facts → `correlations.within_encoder` | Verified from run/output records |

---

## B. Software versions

**Environment that generated the cached embeddings (August 2026, `C:\anaconda\envs\hf`)**

| Package | Version | Printed on |
|---|---|---|
| Python | 3.9.25 | notebook metadata (13 and 20 Aug); 19 Sep |
| PyTorch | 2.5.1 | 13 Aug; 19 Sep |
| CUDA (as reported by torch) | 11.8 | 19 Sep |
| torchvision | 0.20.1 | 19 Sep |
| Transformers | 4.49.0 | 13 Aug (after the pin); 19 Sep |
| OpenCLIP (`open_clip`) | 3.3.0 | 13 Aug; 19 Sep |
| Pillow | 11.1.0 | 13 Aug; 19 Sep |
| NumPy | 1.26.4 | 13 Aug; 19 Sep |
| SciPy | 1.13.1 | 20 Aug; 19 Sep |
| datasets | 4.4.1 | 19 Aug; 19 Sep |
| huggingface_hub | 0.36.0 | 19 Sep |
| tokenizers | 0.21.4 | 19 Sep |
| safetensors | 0.7.0 | 19 Sep |
| tqdm | 4.67.1 | 13 Aug; 19 Sep |
| Operating system | Windows-10-10.0.26200-SP0 | 19 Sep |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4.0 GiB | 19 Sep |

**Analysis and plotting in the same original environment**

| Use | Package versions | Record |
|---|---|---|
| Drift values in Tables II and III (computed 20–22 Aug) | NumPy 1.26.4 | Records §2; Transcript A line 4365 (`make_category_yardstick_plots.py`) |
| `final_premst_results/` figures, including the t-SNE (generated 20 Sep, 20:21–20:25 UTC) | scikit-learn 1.3.2, Matplotlib 3.9.1, NumPy 1.26.4 | Records §3 (Transcript B lines 965, 993) |
| Notebook execution | nbconvert 7.17.1, nbformat 5.10.4, ipykernel 6.30.1 | Records §1 (Transcript B line 164) |

**Environment for the October analyses (rebuilt, `C:\Users\user\anaconda3\envs\hf`)**

Python 3.9.25; torch 2.5.1+cu118; torchvision 0.20.1+cu118; transformers 4.49.0; open_clip_torch 3.3.0; numpy 1.26.4; scikit-learn 1.3.2; scipy 1.13.1; Pillow 11.3.0; matplotlib 3.9.4; datasets 4.5.0. Source: `requirements.txt`.

This environment produced the crop manifests, the permutation tests, the bootstrap intervals, the classifier's text embeddings and predictions, and the joined tables. It did not produce any image embedding used in a reported number.

**Differences between the two environments:** Pillow 11.1.0 (conda build, IJG libjpeg 9) → 11.3.0 (pip wheel, libjpeg-turbo 3.1.1); Matplotlib 3.9.1 → 3.9.4; datasets 4.4.1 → 4.5.0; huggingface_hub 0.36.0 → 0.36.2; ipykernel 6.30.1 → 6.31.0; tqdm 4.67.1 → 4.70.1. Torch, torchvision, Transformers, OpenCLIP, NumPy, scikit-learn, SciPy, tokenizers and safetensors are the same versions.

---

## C. Preprocessing by encoder

### CLIP ViT-B/32 and fine-tuned CLIP (same processor)

- **Processor:** `CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")`; image side is `CLIPImageProcessor`.
- **`preprocessor_config.json` as stored in the checkpoint revision** (sha256 `910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd`, identical in the original and current caches):

```json
{"crop_size": 224, "do_center_crop": true, "do_normalize": true, "do_resize": true,
 "feature_extractor_type": "CLIPFeatureExtractor",
 "image_mean": [0.48145466, 0.4578275, 0.40821073],
 "image_std": [0.26862954, 0.26130258, 0.27577711],
 "resample": 3, "size": 224}
```

- **Effective configuration of the loaded processor (transformers 4.49.0):**

```json
{"do_resize": true, "size": {"shortest_edge": 224}, "resample": 3,
 "do_center_crop": true, "crop_size": {"height": 224, "width": 224},
 "do_rescale": true, "rescale_factor": 0.00392156862745098,
 "do_normalize": true,
 "image_mean": [0.48145466, 0.4578275, 0.40821073],
 "image_std": [0.26862954, 0.26130258, 0.27577711],
 "do_convert_rgb": true, "image_processor_type": "CLIPImageProcessor"}
```

- **In words:**
  1. Convert to RGB.
  2. Resize so the shortest side is 224 pixels, bicubic interpolation (`resample` 3), aspect ratio kept.
  3. Centre-crop 224 × 224.
  4. Rescale by 1/255.
  5. Normalise with the mean and standard deviation above.
- **Call:** `processor(images=batch, return_tensors="pt")`, batch size 16, `torch.no_grad()`.
- **Embedding:** `model.get_image_features(...)` (512-d), divided by its L2 norm, stored as float32.
- **Fine-tuning only:** text is tokenised with `padding="max_length", truncation=True, max_length=77`; the image side is unchanged.
- **Source:** `src/embeddings.py` lines 59–87, 101–112.

### RemoteCLIP ViT-L/14

- **Transform:** the `preprocess` returned by `open_clip.create_model_and_transforms("ViT-L-14", pretrained=None)` under open_clip 3.3.0 and torchvision 0.20.1.
- **Preprocess configuration reported by the model:**

```json
{"size": [224, 224], "mode": "RGB",
 "mean": [0.48145466, 0.4578275, 0.40821073],
 "std": [0.26862954, 0.26130258, 0.27577711],
 "interpolation": "bicubic", "resize_mode": "shortest", "fill_color": 0}
```

- **Transform list as printed:**

```
Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
CenterCrop(size=(224, 224))
MaybeConvertMode()
MaybeToTensor()
Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
```

- **In words:**
  1. Resize the shortest side to 224, bicubic, antialiased.
  2. Centre-crop 224 × 224.
  3. Convert to RGB.
  4. Convert to a tensor in [0, 1].
  5. Normalise with the same mean and standard deviation as CLIP.
- **Call:** `preprocess(crop)` per crop, stacked in batches of 16, `torch.no_grad()`.
- **Embedding:** `model.encode_image(...)` (768-d), divided by its L2 norm, stored as float32.
- **Source:** `src/embeddings.py` lines 8–17, 28–56.

### Common to all three encoders

- **Padding or margin:** none is added around a crop.
- **Classifier prompt (October):** `"a photo of a {label}"` with label words `small vehicle`, `large vehicle`, `car`, `bus`, `truck`, `building`. Text embeddings come from each model's own text tower and are divided by their L2 norm.
- **Pixel decoding:** the original embeddings were decoded by a conda-built Pillow 11.1.0 that uses IJG libjpeg 9. This was tested in isolation:
  - Pip-wheel Pillow 11.1.0 and 11.3.0 (libjpeg-turbo 3.1.0 and 3.1.1) decode the same JPEG files to identical pixels, so the Pillow version is not the cause.
  - Conda-built Pillow 11.1.0 (libjpeg 9f) decodes the same files differently: across 10 VisDrone images, 25.9% of pixels differ from the libjpeg-turbo decode, by at most 57 grey levels and 0.25 on average per channel.
  - Re-encoding with the conda-built decoder reproduces every cached embedding to float precision; re-encoding with either pip wheel gives median cosine 0.9938–0.9992 on JPEG-sourced crops. PNG-sourced crops are exact under all three.
  - xView chips arrive as encoded images inside the dataset files and are decoded by the same Pillow, so they follow the same pattern.

---

## D. Result tables and excerpts

Values are copied by script from the named files. Original result files that store four decimals are shown with four; values from the October analyses are shown to four or more decimals, and the full-precision numbers are in the JSON keys named under each table.

### D1. Size and shape of the evaluated crops

Source: `results/crop_manifests/*.csv` (width and height columns), evaluated categories only. "Long/short" is the longer side divided by the shorter side; with a centre crop, anything beyond the central square of an elongated crop is not seen by the encoder.

| Crop list | Evaluated crops | Min width | Min height | Crops with a side < 16 | Median shorter side | Median long/short | Share with long/short > 1.5 | Share > 2 |
|---|---|---|---|---|---|---|---|---|
| p1_dota | 600 | 16 | 16 | 0 | 34 | 1.39 | 42.7% | 21.8% |
| p1_visdrone_heldout | 600 | 16 | 16 | 0 | 36 | 1.50 | 49.8% | 23.0% |
| p2_xview | 900 | 16 | 16 | 0 | 18 | 1.25 | 28.2% | 9.3% |
| p3_xview_building | 300 | 16 | 16 | 0 | 37 | 1.38 | 38.0% | 16.3% |
| p2_visdrone | 900 | 16 | 16 | 0 | 37 | 1.55 | 53.2% | 24.9% |
| p3_uavid_building | 300 | 32 | 18 | 0 | 91 | 1.61 | 54.7% | 33.7% |

Images and sequences behind each category's 300 crops:

| Crop list | Category | Crops | Source images | UAV sequences | Max crops from one image |
|---|---|---|---|---|---|
| p1_dota | large-vehicle | 300 | 68 | — | 42 |
| p1_dota | small-vehicle | 300 | 119 | — | 43 |
| p1_visdrone_heldout | large-vehicle | 300 | 241 | 66 | 5 |
| p1_visdrone_heldout | small-vehicle | 300 | 244 | 82 | 4 |
| p2_xview | bus | 300 | 210 | — | 5 |
| p2_xview | car | 300 | 233 | — | 6 |
| p2_xview | truck | 300 | 220 | — | 8 |
| p3_xview_building | building | 300 | 127 | — | 9 |
| p2_visdrone | bus | 300 | 257 | 47 | 3 |
| p2_visdrone | car | 300 | 289 | 84 | 3 |
| p2_visdrone | truck | 300 | 272 | 56 | 3 |
| p3_uavid_building | building | 300 | 164 | 27 | 9 |

### D2. Cached embeddings versus re-encoding of the replayed crops, by JPEG decoder

Source: `results/paper_facts/crop_identity_conda_pillow_11.1.0_jpeg9.json` (conda-built Pillow 11.1.0, IJG libjpeg 9: the original decoder) and `crop_identity_pip_pillow_11.1.0.json` (pip wheel, libjpeg-turbo 3.1.0; pixel-identical to the pip wheel of 11.3.0). "Own row" = the cached row with the same index. "Optimal matching" = one-to-one assignment maximising total cosine. "Misaligned control" = median cosine after deliberately shifting rows within each label. Labels matched the cached label files for every list and encoder under both decoders, and raw crop counts before sampling equal the August logs for all six lists.

| Crop list | Encoder | n | Original decoder: min cosine to own row | Original decoder: share > 0.999999 | Original decoder: optimal matching = own row | Pip decoder: median cosine | Pip decoder: min cosine | Pip decoder: nearest = own row | Pip decoder: optimal matching = own row | Misaligned control (median) |
|---|---|---|---|---|---|---|---|---|---|---|
| p1_dota | CLIP ViT-B/32 | 600 | 0.9999998 | 100.0% | 600 / 600 | 1.000000 | 1.000000 | 600 / 600 | 600 / 600 | 0.8739 |
| p1_dota | RemoteCLIP ViT-L/14 | 600 | 0.9999998 | 100.0% | 600 / 600 | 1.000000 | 1.000000 | 600 / 600 | 600 / 600 | 0.9028 |
| p1_dota | Fine-tuned CLIP | 600 | 0.9999998 | 100.0% | 600 / 600 | 1.000000 | 1.000000 | 600 / 600 | 600 / 600 | 0.9559 |
| p1_visdrone_heldout | CLIP ViT-B/32 | 600 | 0.9999999 | 100.0% | 600 / 600 | 0.994002 | 0.885595 | 600 / 600 | 600 / 600 | 0.8065 |
| p1_visdrone_heldout | RemoteCLIP ViT-L/14 | 600 | 0.9999998 | 100.0% | 600 / 600 | 0.995598 | 0.948248 | 600 / 600 | 600 / 600 | 0.8689 |
| p1_visdrone_heldout | Fine-tuned CLIP | 600 | 0.9999999 | 100.0% | 600 / 600 | 0.999185 | 0.978814 | 592 / 600 | 600 / 600 | 0.9735 |
| p2_visdrone | CLIP ViT-B/32 | 1200 | 0.9999998 | 100.0% | 1200 / 1200 | 0.993757 | 0.883326 | 1198 / 1200 | 1200 / 1200 | 0.8030 |
| p2_visdrone | RemoteCLIP ViT-L/14 | 1200 | 0.9999998 | 100.0% | 1200 / 1200 | 0.995289 | 0.906302 | 1196 / 1200 | 1200 / 1200 | 0.8664 |
| p2_xview | CLIP ViT-B/32 | 925 | 0.9999998 | 100.0% | 925 / 925 | 0.996743 | 0.957889 | 918 / 925 | 925 / 925 | 0.9195 |
| p2_xview | RemoteCLIP ViT-L/14 | 925 | 0.9999998 | 100.0% | 925 / 925 | 0.996564 | 0.970542 | 919 / 925 | 925 / 925 | 0.9373 |
| p3_uavid_building | CLIP ViT-B/32 | 300 | 0.9999998 | 100.0% | 300 / 300 | 1.000000 | 1.000000 | 300 / 300 | 300 / 300 | 0.6993 |
| p3_uavid_building | RemoteCLIP ViT-L/14 | 300 | 0.9999999 | 100.0% | 300 / 300 | 1.000000 | 1.000000 | 300 / 300 | 300 / 300 | 0.7994 |
| p3_xview_building | CLIP ViT-B/32 | 300 | 0.9999999 | 100.0% | 298 / 300 | 0.997250 | 0.926665 | 298 / 300 | 300 / 300 | 0.8821 |
| p3_xview_building | RemoteCLIP ViT-L/14 | 300 | 0.9999999 | 100.0% | 300 / 300 | 0.997201 | 0.919159 | 299 / 300 | 300 / 300 | 0.9129 |

Totals under the original decoder: 9050 crop–encoder pairs; lowest cosine to own row 0.99999975; largest element-wise difference 1.79e-06.

In P3 xView building (CLIP), the optimal matching swaps rows 94 and 210. Those two crops are exact duplicates (identical embeddings), so the swap is a tie, not a mismatch.

Raw per-class crop counts before sampling (replay = August log in every case):

| Crop list | Counts |
|---|---|
| p1_dota | {'large-vehicle': 3033, 'small-vehicle': 6537} |
| p1_visdrone_heldout | {'small-vehicle': 22368, 'large-vehicle': 7341} |
| p2_visdrone | {'car': 109567, 'van': 20534, 'truck': 11297, 'bus': 5233} |
| p2_xview | {'car': 1269, 'truck': 3694, 'bus': 903, 'van': 25} |
| p3_uavid_building | {'building': 1199} |
| p3_xview_building | {'building': 3711} |

Exact duplicate crops among the evaluated samples (cosine > 0.999999 between two different cached rows, under every encoder). Source: `results/paper_facts/exact_duplicate_crops.json`.

| Crop list | Evaluated crops | Duplicate pairs | Pairs (rows; chips; boxes) |
|---|---|---|---|
| p1_dota | 600 | 0 | — |
| p1_visdrone_heldout | 600 | 0 | — |
| p2_xview | 900 | 2 | rows 180 & 357 (bus), xview_chip_109 [386, 336, 402, 374] and xview_chip_105 [386, 416, 402, 454]; rows 804 & 860 (bus), xview_chip_4821 [476, 137, 512, 155] and xview_chip_4831 [476, 89, 512, 107] |
| p2_visdrone | 900 | 0 | — |
| p3_xview_building | 300 | 1 | rows 94 & 210 (building), xview_chip_110 [398, 134, 512, 251] and xview_chip_106 [398, 214, 512, 331] |
| p3_uavid_building | 300 | 0 | — |

### D3. Image-level permutation tests

Source: `results/drift_stats/drift_stats.json` → `tasks.<task>.permutation`. 2,000 permutations; p = (1 + count of null ≥ observed) / 2001.

| Task | Category | Encoder | Images (overhead / UAV) | Observed L2 | Null 95th pct | Null 99.9th pct | Null maximum | p |
|---|---|---|---|---|---|---|---|---|
| P1 | small-vehicle | CLIP ViT-B/32 | 119 / 244 | 0.2460 | 0.0644 | 0.0823 | 0.0923 | 0.0004998 |
| P1 | small-vehicle | RemoteCLIP ViT-L/14 | 119 / 244 | 0.1892 | 0.0639 | 0.0903 | 0.1057 | 0.0004998 |
| P1 | small-vehicle | Fine-tuned CLIP | 119 / 244 | 0.1585 | 0.0438 | 0.0665 | 0.0690 | 0.0004998 |
| P1 | large-vehicle | CLIP ViT-B/32 | 68 / 241 | 0.2596 | 0.1097 | 0.1412 | 0.1476 | 0.0004998 |
| P1 | large-vehicle | RemoteCLIP ViT-L/14 | 68 / 241 | 0.2276 | 0.1003 | 0.1378 | 0.1483 | 0.0004998 |
| P1 | large-vehicle | Fine-tuned CLIP | 68 / 241 | 0.0955 | 0.0608 | 0.0833 | 0.0938 | 0.0004998 |
| P2+P3 | car | CLIP ViT-B/32 | 233 / 289 | 0.3564 | 0.0512 | 0.0713 | 0.0922 | 0.0004998 |
| P2+P3 | car | RemoteCLIP ViT-L/14 | 233 / 289 | 0.3024 | 0.0457 | 0.0732 | 0.0765 | 0.0004998 |
| P2+P3 | bus | CLIP ViT-B/32 | 210 / 257 | 0.3256 | 0.0538 | 0.0750 | 0.0850 | 0.0004998 |
| P2+P3 | bus | RemoteCLIP ViT-L/14 | 210 / 257 | 0.2802 | 0.0471 | 0.0621 | 0.0657 | 0.0004998 |
| P2+P3 | truck | CLIP ViT-B/32 | 220 / 272 | 0.2837 | 0.0498 | 0.0680 | 0.0722 | 0.0004998 |
| P2+P3 | truck | RemoteCLIP ViT-L/14 | 220 / 272 | 0.2366 | 0.0432 | 0.0587 | 0.0599 | 0.0004998 |
| P2+P3 | building | CLIP ViT-B/32 | 127 / 164 | 0.5417 | 0.0987 | 0.1505 | 0.1729 | 0.0004998 |
| P2+P3 | building | RemoteCLIP ViT-L/14 | 127 / 164 | 0.3783 | 0.0759 | 0.1119 | 0.1214 | 0.0004998 |

Extended run (fine-tuned CLIP, large-vehicle): 20000 permutations, seed 7, p = 0.0003; observed L2 0.09545, null 99.9th percentile 0.08554, null maximum 0.09904.

### D4. Drift intervals: image-level versus sequence-level resampling

Source: `results/drift_stats/drift_stats.json` → `tasks.<task>.bootstrap.image` and `.sequence`. 95% percentile intervals, 2,000 replicates. "Sequence" resamples UAV crops by sequence; overhead crops are resampled by image in both.

| Task | Category | Encoder | L2 (point) | L2 image CI | L2 sequence CI | Ratio (point) | Ratio image CI | Ratio sequence CI |
|---|---|---|---|---|---|---|---|---|
| P1 | small-vehicle | CLIP ViT-B/32 | 0.2460 | [0.2240, 0.2798] | [0.2203, 0.2796] | 0.6498 | [0.5962, 0.7415] | [0.5847, 0.7406] |
| P1 | small-vehicle | RemoteCLIP ViT-L/14 | 0.1892 | [0.1711, 0.2238] | [0.1685, 0.2227] | 0.5627 | [0.5114, 0.6713] | [0.5032, 0.6719] |
| P1 | small-vehicle | Fine-tuned CLIP | 0.1585 | [0.1158, 0.2070] | [0.1162, 0.2074] | 0.8165 | [0.6302, 1.0390] | [0.6262, 1.0416] |
| P1 | large-vehicle | CLIP ViT-B/32 | 0.2596 | [0.2368, 0.2947] | [0.2318, 0.3009] | 0.6235 | [0.5731, 0.7108] | [0.5634, 0.7240] |
| P1 | large-vehicle | RemoteCLIP ViT-L/14 | 0.2276 | [0.2069, 0.2596] | [0.2047, 0.2605] | 0.6252 | [0.5743, 0.7200] | [0.5707, 0.7212] |
| P1 | large-vehicle | Fine-tuned CLIP | 0.0955 | [0.0793, 0.1220] | [0.0792, 0.1257] | 0.4132 | [0.3384, 0.5386] | [0.3386, 0.5523] |
| P2+P3 | car | CLIP ViT-B/32 | 0.3564 | [0.3317, 0.3821] | [0.3158, 0.3980] | 0.9931 | [0.9306, 1.0628] | [0.8943, 1.1050] |
| P2+P3 | car | RemoteCLIP ViT-L/14 | 0.3024 | [0.2784, 0.3277] | [0.2640, 0.3402] | 0.9719 | [0.8953, 1.0557] | [0.8603, 1.0958] |
| P2+P3 | bus | CLIP ViT-B/32 | 0.3256 | [0.3038, 0.3502] | [0.3058, 0.3650] | 0.8443 | [0.7923, 0.9061] | [0.7937, 0.9305] |
| P2+P3 | bus | RemoteCLIP ViT-L/14 | 0.2802 | [0.2599, 0.3034] | [0.2599, 0.3176] | 0.8491 | [0.7942, 0.9172] | [0.7955, 0.9463] |
| P2+P3 | truck | CLIP ViT-B/32 | 0.2837 | [0.2637, 0.3079] | [0.2427, 0.3283] | 0.7550 | [0.7103, 0.8133] | [0.6694, 0.8580] |
| P2+P3 | truck | RemoteCLIP ViT-L/14 | 0.2366 | [0.2201, 0.2576] | [0.2065, 0.2726] | 0.7500 | [0.7041, 0.8091] | [0.6751, 0.8461] |
| P2+P3 | building | CLIP ViT-B/32 | 0.5417 | [0.5134, 0.5733] | [0.5163, 0.5788] | 1.2219 | [1.1452, 1.3185] | [1.1535, 1.3300] |
| P2+P3 | building | RemoteCLIP ViT-L/14 | 0.3783 | [0.3554, 0.4045] | [0.3578, 0.4072] | 0.9809 | [0.9009, 1.0823] | [0.9069, 1.0961] |

Paired change in the drift ratio relative to CLIP (percent), same replicates:

| Task | Category | Encoder | Point | Image CI | Sequence CI |
|---|---|---|---|---|---|
| P1 | small-vehicle | RemoteCLIP ViT-L/14 | -13.40 | [-18.54, -6.16] | [-18.93, -6.04] |
| P1 | small-vehicle | Fine-tuned CLIP | +25.65 | [-1.48, +52.18] | [-2.63, +54.76] |
| P1 | large-vehicle | RemoteCLIP ViT-L/14 | +0.26 | [-7.99, +11.04] | [-8.03, +11.18] |
| P1 | large-vehicle | Fine-tuned CLIP | -33.73 | [-45.40, -17.59] | [-45.52, -13.86] |
| P2+P3 | car | RemoteCLIP ViT-L/14 | -2.13 | [-6.85, +2.38] | [-8.02, +3.54] |
| P2+P3 | bus | RemoteCLIP ViT-L/14 | +0.57 | [-3.68, +5.03] | [-4.20, +5.86] |
| P2+P3 | truck | RemoteCLIP ViT-L/14 | -0.67 | [-4.74, +3.95] | [-5.43, +4.12] |
| P2+P3 | building | RemoteCLIP ViT-L/14 | -19.72 | [-23.17, -15.94] | [-23.21, -15.48] |

Category-spacing-normalised drift Y_c (pooled task):

| Category | Encoder | Y (point) | Image CI | Sequence CI | Y minus CLIP's Y: image CI | Y minus CLIP's Y: sequence CI |
|---|---|---|---|---|---|---|
| car | CLIP ViT-B/32 | 1.9505 | [1.652, 2.210] | [1.486, 2.322] | — | — |
| car | RemoteCLIP ViT-L/14 | 2.0908 | [1.780, 2.339] | [1.603, 2.438] | [-0.030, +0.298] | [-0.093, +0.340] |
| bus | CLIP ViT-B/32 | 1.8065 | [1.482, 2.091] | [1.486, 2.248] | — | — |
| bus | RemoteCLIP ViT-L/14 | 2.3024 | [1.944, 2.584] | [1.899, 2.733] | [+0.290, +0.672] | [+0.273, +0.674] |
| truck | CLIP ViT-B/32 | 1.4966 | [1.211, 1.795] | [1.006, 2.001] | — | — |
| truck | RemoteCLIP ViT-L/14 | 1.8080 | [1.433, 2.176] | [1.229, 2.385] | [+0.096, +0.516] | [+0.057, +0.537] |
| building | CLIP ViT-B/32 | 2.3189 | [2.175, 2.422] | [2.153, 2.446] | — | — |
| building | RemoteCLIP ViT-L/14 | 2.1233 | [1.872, 2.325] | [1.864, 2.341] | [-0.351, -0.048] | [-0.363, -0.038] |

### D5. Classification and text alignment: image-level versus sequence-level

Source: `results/classification/classification_bootstrap.json`. Recall gap = overhead − UAV, percentage points. Δm_c = overhead margin − UAV margin.

| Task | Category | Encoder | Recall overhead | Recall UAV | Gap (point) | Gap image CI | Gap sequence CI | Δm_c (point) | Δm_c image CI | Δm_c sequence CI |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 | small-vehicle | CLIP ViT-B/32 | 99.00% | 99.33% | -0.33 | [-2.03, +1.04] | [-1.93, +1.12] | -0.00040 | [-0.00469, +0.00392] | [-0.00492, +0.00423] |
| P1 | small-vehicle | RemoteCLIP ViT-L/14 | 81.33% | 91.33% | -10.00 | [-15.71, -3.79] | [-15.83, -3.06] | -0.00708 | [-0.00849, -0.00561] | [-0.00856, -0.00545] |
| P1 | small-vehicle | Fine-tuned CLIP | 86.33% | 96.33% | -10.00 | [-15.18, -5.22] | [-15.41, -5.67] | -0.01861 | [-0.02553, -0.01261] | [-0.02536, -0.01275] |
| P1 | large-vehicle | CLIP ViT-B/32 | 3.33% | 9.67% | -6.33 | [-10.20, -2.22] | [-10.53, -1.98] | -0.00386 | [-0.00744, +0.00027] | [-0.00750, +0.00041] |
| P1 | large-vehicle | RemoteCLIP ViT-L/14 | 52.33% | 42.67% | +9.67 | [-4.92, +23.63] | [-5.99, +24.37] | +0.00279 | [-0.00280, +0.00795] | [-0.00312, +0.00874] |
| P1 | large-vehicle | Fine-tuned CLIP | 83.33% | 72.00% | +11.33 | [+3.20, +19.07] | [+2.39, +20.23] | +0.01316 | [+0.00492, +0.02060] | [+0.00387, +0.02243] |
| P2+P3 | car | CLIP ViT-B/32 | 92.00% | 87.67% | +4.33 | [-0.72, +9.11] | [-1.24, +10.33] | -0.01419 | [-0.01652, -0.01193] | [-0.01767, -0.01030] |
| P2+P3 | car | RemoteCLIP ViT-L/14 | 8.33% | 63.33% | -55.00 | [-61.30, -48.85] | [-63.45, -46.01] | -0.02949 | [-0.03283, -0.02636] | [-0.03378, -0.02485] |
| P2+P3 | bus | CLIP ViT-B/32 | 1.00% | 41.33% | -40.33 | [-46.55, -34.56] | [-51.01, -35.04] | -0.02120 | [-0.02451, -0.01796] | [-0.02751, -0.01809] |
| P2+P3 | bus | RemoteCLIP ViT-L/14 | 1.67% | 41.00% | -39.33 | [-44.95, -33.47] | [-45.30, -31.62] | -0.02890 | [-0.03346, -0.02454] | [-0.03444, -0.02233] |
| P2+P3 | truck | CLIP ViT-B/32 | 10.00% | 30.33% | -20.33 | [-26.38, -14.22] | [-27.38, -13.29] | -0.01008 | [-0.01239, -0.00784] | [-0.01351, -0.00683] |
| P2+P3 | truck | RemoteCLIP ViT-L/14 | 0.67% | 11.67% | -11.00 | [-15.00, -7.29] | [-16.99, -5.56] | -0.00175 | [-0.00530, +0.00159] | [-0.00618, +0.00251] |
| P2+P3 | building | CLIP ViT-B/32 | 28.67% | 81.67% | -53.00 | [-59.61, -46.40] | [-60.61, -45.62] | -0.03501 | [-0.03853, -0.03168] | [-0.04064, -0.02997] |
| P2+P3 | building | RemoteCLIP ViT-L/14 | 98.33% | 91.33% | +7.00 | [+3.60, +10.81] | [+2.32, +13.69] | -0.01645 | [-0.02089, -0.01182] | [-0.02188, -0.01036] |

Anchor similarities and margins at the centroid (point estimates):

| Task | Category | Encoder | a overhead | a UAV | A_c | Margin overhead | Margin UAV | Nearest other label (overhead / UAV) |
|---|---|---|---|---|---|---|---|---|
| P1 | small-vehicle | CLIP ViT-B/32 | 0.24276 | 0.26601 | 0.02325 | +0.02104 | +0.02144 | large-vehicle / large-vehicle |
| P1 | small-vehicle | RemoteCLIP ViT-L/14 | 0.23584 | 0.23581 | 0.00003 | +0.00554 | +0.01262 | large-vehicle / large-vehicle |
| P1 | small-vehicle | Fine-tuned CLIP | 0.28737 | 0.29750 | 0.01013 | +0.03490 | +0.05350 | large-vehicle / large-vehicle |
| P1 | large-vehicle | CLIP ViT-B/32 | 0.21998 | 0.25063 | 0.03065 | -0.01758 | -0.01373 | small-vehicle / small-vehicle |
| P1 | large-vehicle | RemoteCLIP ViT-L/14 | 0.22706 | 0.23485 | 0.00778 | +0.00612 | +0.00333 | small-vehicle / small-vehicle |
| P1 | large-vehicle | Fine-tuned CLIP | 0.30599 | 0.29991 | 0.00608 | +0.04151 | +0.02835 | small-vehicle / small-vehicle |
| P2+P3 | car | CLIP ViT-B/32 | 0.23480 | 0.26812 | 0.03331 | +0.00849 | +0.02269 | truck / truck |
| P2+P3 | car | RemoteCLIP ViT-L/14 | 0.20618 | 0.24284 | 0.03666 | -0.01194 | +0.01755 | building / building |
| P2+P3 | bus | CLIP ViT-B/32 | 0.22188 | 0.26118 | 0.03930 | -0.01193 | +0.00927 | car / car |
| P2+P3 | bus | RemoteCLIP ViT-L/14 | 0.19794 | 0.24563 | 0.04769 | -0.02508 | +0.00381 | building / building |
| P2+P3 | truck | CLIP ViT-B/32 | 0.22592 | 0.25585 | 0.02993 | -0.00731 | +0.00277 | car / car |
| P2+P3 | truck | RemoteCLIP ViT-L/14 | 0.20466 | 0.22763 | 0.02298 | -0.01692 | -0.01517 | building / building |
| P2+P3 | building | CLIP ViT-B/32 | 0.22555 | 0.28365 | 0.05810 | -0.00386 | +0.03115 | car / car |
| P2+P3 | building | RemoteCLIP ViT-L/14 | 0.23388 | 0.27126 | 0.03738 | +0.02283 | +0.03928 | car / car |

Confusion matrices (rows = true label, columns = predicted label, in the task's category order):

- **P1** (order: small-vehicle, large-vehicle)
  - CLIP ViT-B/32, overhead: `[[297, 3], [290, 10]]`
  - CLIP ViT-B/32, uav: `[[298, 2], [271, 29]]`
  - RemoteCLIP ViT-L/14, overhead: `[[244, 56], [143, 157]]`
  - RemoteCLIP ViT-L/14, uav: `[[274, 26], [172, 128]]`
  - Fine-tuned CLIP, overhead: `[[259, 41], [50, 250]]`
  - Fine-tuned CLIP, uav: `[[289, 11], [84, 216]]`
- **P2+P3** (order: car, bus, truck, building)
  - CLIP ViT-B/32, overhead: `[[276, 0, 19, 5], [280, 3, 13, 4], [257, 0, 30, 13], [117, 2, 95, 86]]`
  - CLIP ViT-B/32, uav: `[[263, 14, 21, 2], [116, 124, 28, 32], [133, 47, 91, 29], [34, 19, 2, 245]]`
  - RemoteCLIP ViT-L/14, overhead: `[[25, 2, 0, 273], [12, 5, 1, 282], [12, 2, 2, 284], [5, 0, 0, 295]]`
  - RemoteCLIP ViT-L/14, uav: `[[190, 7, 0, 103], [36, 123, 3, 138], [49, 20, 35, 196], [10, 10, 6, 274]]`

### D6. Accuracy, accuracy gaps and paired encoder differences

Source: `results/classification/classification_bootstrap.json`; `results/paper_facts/facts.json` → `accuracy_gaps`, `paired_encoder_accuracy_difference`. Gap = overhead accuracy − UAV accuracy, percentage points.

| Task | Encoder | Accuracy overhead | Accuracy UAV | Macro-F1 overhead | Macro-F1 UAV | Gap (point) | Gap image CI | Gap sequence CI |
|---|---|---|---|---|---|---|---|---|
| P1 | CLIP ViT-B/32 | 51.17% | 54.50% | 0.3668 | 0.4305 | -3.33 | [-15.97, +9.11] | [-16.44, +10.27] |
| P1 | RemoteCLIP ViT-L/14 | 66.83% | 67.00% | 0.6612 | 0.6492 | -0.17 | [-7.24, +7.28] | [-7.97, +7.71] |
| P1 | Fine-tuned CLIP | 84.83% | 84.17% | 0.8483 | 0.8393 | +0.67 | [-3.97, +5.32] | [-4.68, +5.64] |
| P2+P3 | CLIP ViT-B/32 | 32.92% | 60.25% | 0.2553 | 0.5829 | -27.33 | [-31.64, -23.28] | [-35.42, -20.83] |
| P2+P3 | RemoteCLIP ViT-L/14 | 27.25% | 51.83% | 0.1496 | 0.4825 | -24.58 | [-30.09, -19.27] | [-34.28, -16.28] |

Paired difference between encoder gaps (this encoder's gap minus CLIP's gap, same replicates):

| Task | Encoder | Point | Image CI | Sequence CI |
|---|---|---|---|---|
| P1 | RemoteCLIP ViT-L/14 | +3.17 | [-10.12, +17.71] | [-10.58, +17.78] |
| P1 | Fine-tuned CLIP | +4.00 | [-8.48, +16.50] | [-8.91, +16.17] |
| P2+P3 | RemoteCLIP ViT-L/14 | +2.75 | [-3.45, +8.66] | [-4.27, +10.13] |

Paired accuracy difference between encoders within one domain (encoder minus CLIP; image bootstrap, 4,000 replicates, seed 11, computed 14 October from `predictions.csv`):

| Task | Domain | Comparison | Point (pp) | 95% image CI (pp) |
|---|---|---|---|---|
| P1 | overhead | RemoteCLIP − CLIP | +15.67 | [+2.51, +29.47] |
| P1 | uav | RemoteCLIP − CLIP | +12.50 | [+8.69, +16.45] |
| P1 | overhead | Fine-tuned CLIP − CLIP | +33.67 | [+22.24, +45.02] |
| P1 | uav | Fine-tuned CLIP − CLIP | +29.67 | [+25.52, +33.87] |
| P2+P3 | overhead | RemoteCLIP − CLIP | -5.67 | [-10.89, -0.58] |
| P2+P3 | uav | RemoteCLIP − CLIP | -8.42 | [-11.62, -5.31] |

### D7. Correlation inputs and values

Source: `results/joined/joined_table.csv` (rows) and `results/joined/association.json` (stored values); recomputed in `results/paper_facts/facts.json` → `correlations`. Spearman rank correlation via `scipy.stats.spearmanr`. No ties in any column.

| Encoder | Category | Cosine displacement | Recall overhead | Recall UAV | Recall gap (overhead − UAV, pp) | Absolute recall gap | A_c | Δm_c | Rank of cosine displacement (pooled) | Rank of absolute recall gap (pooled) |
|---|---|---|---|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | car | 0.071774 | 92.00% | 87.67% | +4.33 | 4.33 | 0.03331 | -0.01419 | 6 | 1 |
| RemoteCLIP ViT-L/14 | car | 0.050350 | 8.33% | 63.33% | -55.00 | 55.00 | 0.03666 | -0.02949 | 4 | 8 |
| CLIP ViT-B/32 | bus | 0.060278 | 1.00% | 41.33% | -40.33 | 40.33 | 0.03930 | -0.02120 | 5 | 6 |
| RemoteCLIP ViT-L/14 | bus | 0.043389 | 1.67% | 41.00% | -39.33 | 39.33 | 0.04769 | -0.02890 | 2 | 5 |
| CLIP ViT-B/32 | truck | 0.045092 | 10.00% | 30.33% | -20.33 | 20.33 | 0.02993 | -0.01008 | 3 | 4 |
| RemoteCLIP ViT-L/14 | truck | 0.030340 | 0.67% | 11.67% | -11.00 | 11.00 | 0.02298 | -0.00175 | 1 | 3 |
| CLIP ViT-B/32 | building | 0.182164 | 28.67% | 81.67% | -53.00 | 53.00 | 0.05810 | -0.03501 | 8 | 7 |
| RemoteCLIP ViT-L/14 | building | 0.084660 | 98.33% | 91.33% | +7.00 | 7.00 | 0.03738 | -0.01645 | 7 | 2 |

| Correlation of cosine displacement with… | Pooled, 8 rows | CLIP, 4 rows | RemoteCLIP, 4 rows |
|---|---|---|---|
| absolute recall gap (the reported 0.02) | 0.0238095 | 0.40 | -0.20 |
| signed recall gap (overhead − UAV) | 0.0000000 | -0.40 | 0.20 |
| A_c | 0.5238095 | 0.80 | 0.40 |
| absolute Δm_c | 0.4761905 | 0.80 | 0.40 |

With 4 rows and no ties, Spearman's coefficient can only take the values 0, ±0.2, ±0.4, ±0.6, ±0.8 and ±1.

### D8. DOTA platform metadata of the sampled images

Source: lines 1–2 of each `data/raw/DOTA/labelTxt/<image>.txt`; `results/paper_facts/facts.json` → `dota_metadata`.

- Sampled images: 149; `imagesource` values: {'GoogleEarth': 149}.
- Sampled crops by label and `imagesource`: {'small-vehicle': {'GoogleEarth': 300}, 'large-vehicle': {'GoogleEarth': 300}}.
- All 469 local images by `imagesource`: {'GoogleEarth': 469}.
- GSD is present for all sampled images (missing: 0). Minimum, median, maximum as stored: [1.34110450745e-06, 0.130310363461, 0.50790386164]. The minimum is the single value in `P0256.txt`; excluding it, the range is 0.1070 to 0.5079 m/pixel (median 0.1305, n = 148).

### D9. VisDrone filename-prefix groups

Source: `results/paper_facts/visdrone_prefix_groups.json`. File names have the form `<prefix>_<second field>_d_<id>.jpg`. Resemblance is the mean correlation of 16×16 grey thumbnails over randomly drawn image pairs.

| Kind of group | Groups | Images | Mean thumbnail correlation within a group |
|---|---|---|---|
| Second field varies (clip-like) | 175 | 2,105 | 0.347 (adjacent frames 0.383; first vs last frame 0.300) |
| Second field always `00000` (still-image sets) | 33 | 4,366 | 0.097 |
| Reference: pairs from different groups | — | — | 0.100 |

| Evaluated crop list | Evaluated crops | From clip-like groups | From still-image sets | Groups used (clip-like / still-image) | Largest share of crops in one group |
|---|---|---|---|---|---|
| p1_visdrone_heldout | 600 | 226 | 374 | 65 / 33 | 9.2% |
| p2_visdrone | 900 | 254 | 646 | 75 / 32 | 13.7% |

### D10. CLIP weight files

Source: `results/paper_facts/clip_weight_files.json`.

- Tensors in `pytorch_model.bin`: 400; in `model.safetensors`: 400; names only in one file: 0.
- Tensors compared: 400; not bit-identical: 0; values compared: 151,277,440; maximum absolute difference: 0.0.
- Transformers 4.49.0 log lines on loading:
  - `loading weights file pytorch_model.bin from cache at C:\Users\user\.cache\huggingface\hub\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268\pytorch_model.bin`
- Loaded model equals the `.bin` tensors for every parameter: True.

---

## E. Questions that need the author's answer

1. **RemoteCLIP download source.** The local checkpoint (dated 28 April 2026) is byte-identical to the official `chendelong/RemoteCLIP` file. Where it was downloaded from in April is not recorded. Is citing the official Hugging Face repository acceptable?
2. **Installs outside the logged sessions.** Between 13 and 20 August, were any packages installed or upgraded in the `hf` environment by hand, outside the Claude Code sessions? The records before and after agree, but only the author can rule this out.
3. **Which script and environment produced the manuscript's Fig. 2** (the two-panel "percent of CLIP" bar chart)? No script in this repository generates that exact figure, so its plotting versions cannot be established from these files. Fig. 3 (t-SNE) corresponds to `final_premst_results/make_final_figures.py`, run on 20 September in the original environment.
4. **DOTA release name.** The local labels contain only the 15 DOTA-v1.0 classes and number 1,411 files, but no file names the release. Was DOTA-v1.0 downloaded?
5. **VisDrone naming.** The filename prefix groups have been examined (A5): 175 behave like video clips and 33 are sets of still images. If the author has an official VisDrone document describing the file naming, it would let the paper cite the convention; without one, the grouping should be described as filename-prefix groups.
6. **UAVid components.** Was any visual inspection of the sampled building components done outside this repository? None is recorded here.
7. **`gsd` of DOTA image `P0256`.** Its label file gives `1.34110450745e-06`. This is the dataset's own value; should it be treated as missing?

---

## Where I searched for items marked "Not located"

- **Repository:** every tracked and untracked file, the full git history (15 commits; `requirements.txt` was empty in all of them and no environment file was ever committed), `configs/`, `results/`, `notebooks/` (cell outputs and metadata), `models/` (no download metadata folders), and file modification times in `embeddings/` and `models/`.
- **Session transcripts:** Transcripts A and B above, plus `D:\C drive data\Kabir Oberoi\.claude\projects\c--Users-Kabir-Oberoi-Desktop-Jarvis\` (file names only).
- **Restored pre-reset profile `D:\C drive data\Kabir Oberoi\`:** `.conda\environments.txt` (lists `C:\anaconda\envs\hf`), `.condarc`, `.cache\huggingface\` (hub and datasets), `.ipython\profile_default\history.sqlite` (123 sessions; no version print from this project), `.bash_history` (3 lines), `.jupyter\`, `AppData\Local\conda\`.
- **Not present on this machine:** `C:\anaconda\` (the original installation, including its `conda-meta` history), `AppData\Roaming\` and the PowerShell command history. A full `pip freeze` or `conda list` of the original environment could therefore not be read.
