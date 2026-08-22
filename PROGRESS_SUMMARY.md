# Capstone Progress Summary — Semantic Drift in Aerial Imagery

*Prepared for mid-semester evaluation. Written in plain language for use in the report.*

## What the project is about

AI models that understand both images and text (like CLIP) can look at a picture and match it to a word like "car" or "truck" without ever being trained on that exact picture. This works well normally, but our project asks: **does the model still recognize the same object correctly when the picture comes from a very different altitude?** A car photographed from a satellite (straight down, blurry, tiny) looks very different from a car photographed by a low-flying drone (angled, clear, large). If the AI's internal understanding of "car" shifts too much between these two views, that's a problem — we call this shift **semantic drift**.

We measure drift by taking many pictures of the same object type from two datasets — DOTA (satellite images) and VisDrone (drone images) — turning each picture into a numerical "fingerprint" (embedding) using an AI model, and then measuring how far apart the average fingerprint for "car" is between the satellite set and the drone set. A bigger distance means more drift.

## What we have done so far

### 1. Moved the project from a cloud notebook to a local, working setup
The project originally ran on a cloud service (Lightning AI Studio). We moved everything to a local Windows machine with its own GPU, set up the correct software environment, and got the datasets (DOTA and VisDrone) downloaded and ready to use locally.

### 2. Recovered and carefully checked the earlier cloud work
Some result files from the original cloud experiments were saved but the code that produced two of them was missing. We tracked down the missing notebook and went through it in detail to understand exactly how those numbers were produced.

### 3. Reproduced the original finding ourselves, locally
We re-ran the original small-scale experiment (using a general-purpose AI model called CLIP) on our own machine and got results in the same range as the original — confirming the basic finding (semantic drift is real and measurable) wasn't a one-off fluke of the cloud environment.

### 4. Found a serious flaw in one of the earlier comparisons — and explained why
One of the earlier cloud experiments tried to compare a satellite-specialized AI model against a separately fine-tuned drone-specialized AI model. On closer inspection, this comparison was invalid: it compared two *different* AI models rather than testing one model consistently, and used a mathematical shortcut (PCA) to force incompatible results to line up, which doesn't produce a meaningful number. We identified exactly why this was wrong and made sure not to repeat the mistake.

### 5. Built a correct, trustworthy version of that comparison (our main result so far)
We tested two AI models properly this time: a general-purpose model (CLIP) and a model pretrained specifically on satellite/remote-sensing imagery (RemoteCLIP). Each model was applied identically to *both* the satellite and drone images — a fair, apples-to-apples test. Using real ground-truth labels (not guesses), we found:

- **RemoteCLIP (satellite-specialized) shows noticeably less drift than plain CLIP**, for both vehicle types tested.
- This supports the idea that training an AI model on domain-relevant imagery helps it stay more consistent across altitudes — though drift isn't eliminated, just reduced.

### 6. Added a third comparison — then caught and fixed two more issues with it
To make the comparison fair in both directions (not just testing a satellite-specialized model), we fine-tuned our own version of CLIP specifically on drone imagery, then tested it the same consistent way on both datasets. This gave us a three-way comparison instead of two, and the first version showed a very large improvement for the drone-specialized model.

Before trusting that number, we checked it carefully and found two real problems:

- **A vocabulary problem:** VisDrone's "small vehicle" category was accidentally grouping bicycles together with cars. Satellite images can't even resolve something as small as a bicycle, so this was silently comparing two visually different object types as if they were one.
- **A data leakage problem:** the model had been fine-tuned using *every* available drone image, and then tested using a sample drawn from that same pool. That means part of what looked like "the model understands drone imagery well" was really just "the model remembers the exact pictures it was already shown during training" — not a fair test.

We fixed both: removed bicycles from the mapping, and split the drone dataset into a training portion (80%) and a completely separate, untouched testing portion (20%) *before* doing anything else, so the model is only ever evaluated on images it has never seen.

- **After both fixes, the drone-fine-tuned model still shows the lowest drift of all three** — the original finding holds up.
- For large vehicles, the corrected number is almost identical to the original one, meaning that result was already trustworthy and not a memorization artifact.
- For small vehicles, the improvement is real but somewhat smaller than first measured — exactly what you'd expect once memorization is ruled out.

We think this is worth presenting explicitly: catching two of our own methodology mistakes and fixing them before reporting the number is part of what makes this result defensible.

### 7. Fixed the biggest limitation: only 2 shared object categories
DOTA and VisDrone only share **two** object categories in common (small vehicles and large vehicles), which limits how much we can say about drift across more object types. We connected a third dataset, **xView** (a large satellite imagery dataset with much finer vehicle categories), and found a way to access it without downloading the full ~20GB — pulling only the images we need, on demand, using on-demand streaming instead.

Comparing xView against VisDrone the same way as before (same AI model, both directions, real labels), we got **three well-populated shared categories: car, bus, and truck** (a fourth, van, turned out too rare in xView's own annotations to trust, and was correctly excluded rather than measured on too few samples). The result matched our earlier finding exactly: **the satellite-specialized model showed less drift than the general-purpose model on all three categories** — meaningful confirmation, because this is a completely different satellite dataset from DOTA, so it shows the finding isn't specific to one dataset pairing.

### 8. Pushed further: added a 4th category using a second drone dataset
VisDrone has no "building" category at all — it's a pure traffic/pedestrian dataset. To get a 4th shared category, we brought in **UAVid**, a different drone dataset that labels buildings, and paired it with xView's building category (VisDrone still supplies car/bus/truck; UAVid supplies only building — a deliberate substitution, not a full dataset swap).

This required a different extraction method than everything before it: UAVid provides its labels as painted-in colored regions on the image (pixel-by-pixel), not as boxes around each object like every other dataset we used, so we built a way to automatically find the boundaries of each colored region and crop around it.

**The building result was the most striking one in the whole project:**

- Building drift was the **largest of any category measured anywhere in this project** — larger than any vehicle type, in either dataset pairing.
- The satellite-specialized model reduced that drift by **30%**, the biggest improvement of any category measured.

This makes intuitive sense: a building looks almost nothing alike from directly overhead (just a roof outline) versus from a low-flying drone at an angle (full facade, windows, visible structure) — a far bigger viewpoint change than any vehicle undergoes, since a car or truck keeps roughly the same recognizable shape and color from any altitude. This is arguably the clearest single piece of evidence for why semantic drift matters.

### 9. Planned how to combine this with a teammate's work
A teammate is combining all three datasets and applying SAM (an AI model that finds object regions in images) to enable real object *detection*, not just classification of pre-cropped images. We planned out how their detection work and our drift-measurement work fit together into a single, complete open-vocabulary object detection pipeline — SAM finds *where* objects are, our models identify *what* they are and measure how consistent that identification is across altitudes.

## Key results so far (numbers)

Drift score = distance between the average "fingerprint" of the same object type in satellite vs. drone images. **Lower = more consistent = less drift.**

| Object type | Plain CLIP | Satellite-specialized (RemoteCLIP) | Drone-specialized (fine-tuned) |
|---|---|---|---|
| Small vehicle | 0.246 | 0.189 (−23%) | 0.159 (−36%) |
| Large vehicle | 0.260 | 0.228 (−12%) | 0.096 (−63%) |

(All three models tested on the same 300 images per category per dataset, on data the fine-tuned model never saw during training, for a fair comparison. These are the corrected, held-out numbers — see item 6 above.)

**Richer vocabulary: xView vs. VisDrone/UAVid (4 categories, see items 7–8 above)**

| Category | Plain CLIP | Satellite-specialized (RemoteCLIP) | UAV-side source |
|---|---|---|---|
| Car | 0.356 | 0.302 (−15%) | VisDrone |
| Bus | 0.326 | 0.280 (−14%) | VisDrone |
| Truck | 0.284 | 0.237 (−17%) | VisDrone |
| Building | 0.542 | 0.378 (−30%) | UAVid |

(300 images per category per dataset. Building compares against UAVid instead of VisDrone, since VisDrone has no building category — see item 8.)

## What's next

- Double-check the drone-specialized result across a few different random samples of the held-out data, to see how much the exact numbers move around
- Scale up the comparisons using the full datasets rather than the current sample sizes
- Recover the 4th vehicle category (van) with a larger xView scan, now that the pipeline is proven
- Combine with the teammate's SAM-based detection work to get real detection performance numbers (not just embedding-distance numbers) and see if drift actually predicts detection failures
- Eventually build a model component that actively reduces this drift, rather than just measuring it

---

## Suggested figures and result files for the report

*(This section is a guide for assembling the report — not meant to be copied into it directly.)*

**Most important — include these:**

1. **`results/remoteclip_vs_clip/three_way_drift_comparison_heldout.png`** — the main result: side-by-side bar charts comparing all three models' drift scores, on the corrected, leakage-free data. This is the single most complete figure and should be the centerpiece. (Note the `_heldout` suffix — there's an earlier, superseded version of this file without that suffix; use the `_heldout` one, it's the corrected result.)
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
