# Clarifications for the paper: items 7–10

Written 14 October 2026 in answer to four open points raised on `PROJECT_FACTS_FOR_PAPER.md`. Each was settled by a direct test on the project's files, not by argument. The fact sheet has been updated to match (sections A1–A3, A5, C, D2, D9, D10). Nothing in the manuscript, the cached embeddings, the checkpoints or the earlier results was changed.

All four are now resolved. In one line each:

| Item | Question | Answer |
|---|---|---|
| 9 | Do the reconstructed crop IDs match the cached embedding rows? | **Yes, every row, exactly.** |
| 10 | Where do the JPEG re-encoding differences come from, and do they matter? | **From the JPEG decoding library, not from crop correspondence. They do not touch any reported number.** |
| 8 | Which CLIP weight file was loaded, and does it matter? | **`pytorch_model.bin`. It does not matter: the two files are bit-identical.** |
| 7 | Do VisDrone filename prefixes identify real videos? | **Only partly. Do not call the analysis "video-sequence resampling" for VisDrone.** |

---

## Item 9 — Crop IDs and cached rows (essential)

**Answer.** Every reconstructed crop is the crop behind the cached row with the same index. This is now established for all rows, not "mostly".

**Evidence.**

1. **Exact reproduction.** Re-encoding all 3,925 crops with the original JPEG decoder (see item 10) reproduces the cached embedding of every crop under every encoder: 9,050 crop–encoder pairs, all with cosine above 0.999999 to their own cached row. The lowest is 0.99999975 and the largest element-wise difference is 1.8e-06, which is float32 rounding.
2. **The earlier ambiguous cases disappear.** "592 of 600 nearest to their own row" (fine-tuned CLIP, P1 VisDrone) was measured with a different JPEG decoder. With the original decoder it is 600 of 600. The same holds for the other partial counts (1198/1200, 1196/1200, 918/925, 919/925, 299/300).
3. **A one-to-one matching test.** Even under the non-original decoder, the optimal one-to-one assignment between re-encoded crops and cached rows is the identity for every row of every list and encoder. No relabelling of rows fits the cache better.
4. **Raw counts equal the original logs.** The per-class crop counts before sampling, recomputed in October, equal the counts printed by the August runs for all six lists (for example 22,368 small-vehicle and 7,341 large-vehicle crops in held-out VisDrone; 1,269 / 3,694 / 903 / 25 for xView car / truck / bus / van).
5. **Labels.** The replayed label sequence equals the cached label file for every list and encoder.
6. **Control.** Deliberately misaligning rows within each label drops the median cosine to 0.70–0.97, so the test can tell correct rows from wrong ones.

**One residual detail, fully explained.** Under the original decoder, six crops are not *uniquely* nearest to their own row. They form three pairs of exact duplicates: the same object cut from two overlapping xView chips, with identical size and identical embeddings.

| List | Rows | Label | Chips |
|---|---|---|---|
| P2 xView | 180 and 357 | bus | 109 and 105 |
| P2 xView | 804 and 860 | bus | 4821 and 4831 |
| P3 xView | 94 and 210 | building | 110 and 106 |

For these, "nearest row" is a tie between two identical rows. No other list contains an exact duplicate.

**What the paper can state.** The image groups used for resampling are exact. A crop manifest links every cached embedding to its source image and box, and re-encoding with the original decoding library reproduces all cached embeddings to floating-point precision.

**A small additional limitation worth one clause.** xView chips overlap, so the same object can appear in two chips; 3 such duplicate pairs occur among the 1,200 evaluated overhead crops of the pooled task. Chip-level resampling treats the two copies as independent.

**Files.** `results/paper_facts/crop_identity_conda_pillow_11.1.0_jpeg9.json` and `.log`; `crop_identity_pip_pillow_11.1.0.json` and `.log`; `exact_duplicate_crops.json`; script `verify_crop_identity.py`. Fact sheet table D2.

---

## Item 10 — JPEG re-encoding differences (cause and consequence)

**Answer.** The differences come from the JPEG decoding library. They are not a sign of wrong crop correspondence, and they do not affect any reported result.

**The cause, isolated by test.** My earlier statement that the Pillow *version* was responsible was wrong. Three decoders were compared on the same files:

| Decoder | JPEG library | Result |
|---|---|---|
| Pillow 11.3.0, pip wheel (the October environment) | libjpeg-turbo 3.1.1 | JPEG-sourced crops re-encode at median cosine 0.9938–0.9992 |
| Pillow 11.1.0, pip wheel | libjpeg-turbo 3.1.0 | Pixel-identical to the row above; same cosines |
| Pillow 11.1.0, conda build | IJG libjpeg 9 (no libjpeg-turbo) | **Every cached embedding reproduced to float precision** |

- The two pip wheels decode the same JPEG files to identical pixels, so changing Pillow from 11.1.0 to 11.3.0 changes nothing.
- The conda build decodes them differently: over 10 VisDrone images, 25.9% of pixels differ from the libjpeg-turbo decode, by at most 57 grey levels and 0.25 on average per channel.
- The original environment was a conda environment (its torch version string has no `+cu118` suffix), and conda's Pillow 11.1.0 builds depend on `jpeg >=9e`. The test used `jpeg 9f`; whether the original was 9e or 9f was not recorded.
- PNG-sourced crops (DOTA, UAVid) reproduce exactly under all three, as expected.

**Do the differences affect the image-grouped statistics?** No.

- Every reported number (Tables II, III and V, the permutation tests, the bootstrap intervals, A_c and Δm_c) is computed from the **cached** embeddings. No re-encoded embedding enters any of them.
- The image groups come from the crop manifest, whose row identity is exact (item 9).
- So the decoder difference affects only someone who *re-extracts* the embeddings with a libjpeg-turbo build of Pillow. In that case the drift values move slightly: L2 displacement by at most 0.009 and the drift ratio by at most 0.025, all inside the reported bootstrap intervals (`results/crop_manifests/manifest_report.json` → `reencode_drift_sensitivity`).

**What the paper can state.** The embeddings were computed with Pillow 11.1.0 built against IJG libjpeg 9. Re-extraction with a libjpeg-turbo build of Pillow changes JPEG-derived embeddings slightly (median cosine 0.994–0.999 to the originals) and moves displacement values by less than the reported interval widths.

**Files.** Same as item 9. Fact sheet rows A1 ("JPEG decoding library behind Pillow") and section C ("Pixel decoding").

---

## Item 8 — Which CLIP weight file was loaded (reproducibility)

**Answer.** `pytorch_model.bin` from revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` of `openai/clip-vit-base-patch32`. The question is moot in any case, because the `.safetensors` copy holds exactly the same weights.

**Evidence.**

1. **Bit-identical weights.** `pytorch_model.bin` and `model.safetensors` each contain 400 tensors with the same names. All 400 are bit-identical: 151,277,440 values compared, maximum absolute difference 0.0.
2. **What Transformers 4.49.0 loads.** With the original library version and the same cache layout, `CLIPModel.from_pretrained` logs "loading weights file pytorch_model.bin from cache at …\snapshots\3d74acf9…\pytorch_model.bin". The loaded model equals the `.bin` tensors for every parameter.
3. **Why a `.safetensors` file exists in the cache.** The same load prints "Attempting to create safetensors variant / Safetensors PR exists". This background check downloads `model.safetensors` into a second snapshot (`c237dc49a33fc61debc9276459120b7eac67e7ef`). It happened in the original cache on 13 August and again in October.
4. **End-to-end confirmation.** Embeddings recomputed from `pytorch_model.bin` reproduce all cached CLIP and fine-tuned-CLIP embeddings to float precision (item 9).

**Checkpoint identity to record.**

| Encoder | Identity |
|---|---|
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; `pytorch_model.bin` sha256 `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f` |
| RemoteCLIP ViT-L/14 | `RemoteCLIP-ViT-L-14.pt`, sha256 `fcc2a7e21e171f4ffcb7a9c0206b8b74ac0c9eb83c67b576958b7a4ed6c8cecb`; byte-identical to the file in `chendelong/RemoteCLIP` |
| Fine-tuned CLIP | `finetuned_clip_visdrone_heldout_epoch3.pt`, sha256 `3c150faac15620b4edce39abfe50d4191f029df85e45ad4a03f17fb10be0ba2a` |

**The one thing not recorded.** The August run did not print the weight-file name. That is why points 1–4 are given instead of a log line from that run.

**Files.** `results/paper_facts/clip_weight_files.json`. Fact sheet rows in A2 and section D10.

---

## Item 7 — VisDrone filename prefixes (only for sequence-based claims)

**Answer.** The prefix is a video identifier for some groups and not for others. The analysis that used it should not be described as video-sequence resampling for VisDrone.

**What the prefixes group.** VisDrone-DET file names have the form `<prefix>_<second field>_d_<id>.jpg`. The 6,471 training images fall into 208 prefix groups of two kinds:

| Kind | Groups | Images | Second field | Do images in a group resemble each other? |
|---|---|---|---|---|
| Clip-like | 175 | 2,105 | Varies (for example 1, 460, 571, 1001…), like frame numbers | Yes: thumbnail correlation 0.35 within a group, 0.38 for adjacent frames |
| Still-image sets | 33 | 4,366 | Always `00000` | No: 0.10, the same as for images from different groups (0.10) |

- The still-image sets have prefixes 9999937 to 9999997 and hold 13 to 434 images each.
- They supply most of the evaluated VisDrone crops: 374 of 600 in P1 and 646 of 900 in P2.
- No dataset document describing the naming was found. The reading above rests on the file names and on image resemblance.

**What the "sequence-level" analysis therefore is.**

- **VisDrone:** resampling of filename-prefix groups, a grouping coarser than the image. It is video-level only for the clip-like groups.
- **UAVid:** genuine video-sequence resampling. The dataset's readme documents the `seqN` sequences, and all 300 building crops come from 27 of them.

**Does this change any result?** No. A coarser grouping can only widen the intervals, and it did so modestly: the sequence-level intervals are a median of 1.09 times (drift) and 1.16 times (classification) as wide as the image-level ones, at most 1.94 times. No interval that excludes zero at image level includes zero at the coarser level, and every Y_c lower bound stays above 1.

**What the paper can state.** Image-level resampling is the primary analysis. As a sensitivity check, UAV crops were also resampled by coarser groups (UAVid video sequences; VisDrone filename-prefix groups, of which only some correspond to video clips), with similar conclusions and somewhat wider intervals.

**What the paper should avoid.** "Sequence IDs are inferred from file names" as a description of all VisDrone groups, and any claim of video-sequence resampling for VisDrone as a whole.

**Files.** `results/paper_facts/visdrone_prefix_groups.json`. Fact sheet rows in A5 and section D9.

---

## Corrections to earlier statements

| Earlier statement | Correction |
|---|---|
| "That the Pillow version is the cause is an inference… not tested in isolation." | Tested. The Pillow version is not the cause; the JPEG library is (conda build with IJG libjpeg 9 against pip wheels with libjpeg-turbo). |
| "592/600 matched their own row" and similar partial counts | These were measured with the non-original decoder. With the original decoder every row matches. |
| "It did not establish whether the original run loaded `.bin` or `.safetensors`, or verify that both contain identical weights." | Both now established: `.bin` is what Transformers 4.49.0 loads, and the two files are bit-identical. |
| "Replayed crop counts equal the original logs" (fact sheet A3) | This row was stated before it had been checked against a replay. It has now been checked and holds for all six lists. |
| "The sequence-level UAV sensitivity analysis" | For VisDrone it is a filename-prefix-group analysis; see item 7. |
