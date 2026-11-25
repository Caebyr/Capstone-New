# The Capstone Project, Explained

**Semantic drift of vision-language embeddings between overhead and drone imagery**
Capstone CPG No. 141, Thapar Institute of Engineering and Technology · Mentor: Dr. Geeta Kasana
Team: Anoop Singh Sachdev · Arnav Gupta · Kabir Oberoi · Khushpreet Kaur · Siddhant Saxena

This is a plain-English reference for the final presentation, for revision, and for anyone joining the work. It covers what the project asks, every concept it uses, everything we did in the order we did it, every mistake we found and fixed, every result, and what we can and cannot claim.

Numbers come from the result files in `results/`. Where the paper submitted to RAIT 2026 words something differently, the paper's wording wins for the paper, but the numbers here are the same ones.

*Last updated: 25 November 2026.*

---

## Contents

1. [The project in one minute](#1-the-project-in-one-minute)
2. [The ideas you need, explained simply](#2-the-ideas-you-need-explained-simply)
3. [The datasets](#3-the-datasets)
4. [The three models](#4-the-three-models)
5. [The story, step by step](#5-the-story-step-by-step)
6. [The results, explained](#6-the-results-explained)
7. [What we can and cannot claim](#7-what-we-can-and-cannot-claim)
8. [Mistakes we caught and fixed](#8-mistakes-we-caught-and-fixed)
9. [Limitations](#9-limitations)
10. [Where the project goes next](#10-where-the-project-goes-next)
11. [Questions an examiner might ask](#11-questions-an-examiner-might-ask)
12. [Where everything lives](#12-where-everything-lives)
13. [Timeline of changes](#13-timeline-of-changes)

---

## 1. The project in one minute

Modern AI models such as CLIP can look at a picture and match it to a word, like "car", without ever having been trained on that exact kind of picture. This is what makes **open-vocabulary object detection** possible: you can ask a detector to find "fire trucks" or "solar panels" even if nobody labelled those for it.

That ability rests on one assumption: the model's internal idea of "car" stays the same whatever the picture looks like. Aerial imagery tests that assumption hard.

- A car seen from a **satellite or high aircraft** is a tiny, blurry, straight-down rectangle of a few dozen pixels.
- A car seen from a **low-flying drone** is larger, sharper and often seen at an angle, with windows and wheels visible.

**Our question:** when the same model looks at cars (or buses, trucks, buildings) from these two kinds of platform, does it place them in the same region of its internal "meaning space"? Or does the category's position move? We call that movement **drift**.

**What we found, in four sentences:**

1. The model's idea of each category moves a lot between overhead and drone images, for all six categories and all three models we tested. The movement is far larger than chance could produce.
2. For four categories, a car's overhead version sits further from its drone version than a car sits from a bus. The move is bigger than the gap between two different categories.
3. A remote-sensing model (RemoteCLIP) moves less in raw distance, but once you account for how tightly it bunches its examples, it is better only for small vehicles and buildings.
4. When we asked the models to name the objects, both did much worse on overhead crops (about 25–27 points lower accuracy). The model with less drift did *not* recognise objects better, so **less drift does not automatically mean better recognition**.

---

## 2. The ideas you need, explained simply

### 2.1 Embedding: a picture turned into a list of numbers

A model like CLIP turns any image into a list of numbers, for example 512 numbers for CLIP ViT-B/32 and 768 for RemoteCLIP. That list is called an **embedding**. Think of it as the image's position on a huge map with 512 directions instead of two.

> **Analogy:** a library where every book is shelved by its meaning, not its title. Books about similar things end up on nearby shelves. An embedding is the shelf address of a picture.

The key property is that **pictures with similar meaning get nearby addresses**. Two photos of cars sit close together, and a photo of a car sits further from a photo of a building.

CLIP also turns **text** into an address on the same map. The words "a photo of a car" land near photos of cars. That is how CLIP can name things it was never specifically trained on: it checks which text address is closest to the image's address.

We always **normalise** embeddings to length 1. Every address then lies on the surface of a sphere, and only the direction matters.

### 2.2 Open-vocabulary detection

A normal detector is trained on a fixed list of labels and can only ever find those. An **open-vocabulary detector** uses a model like CLIP to name each detected object by comparing it with any text you give it. It has two jobs:

1. **Find** where objects are (draw boxes).
2. **Name** what each box contains, using the image–text comparison.

Our work looks at the foundations of job 2. If the model's idea of "car" moves between platforms, the naming step may fail.

### 2.3 Ground-truth crops

Datasets come with **ground truth**: boxes drawn by humans around every object, each with its correct label. A **ground-truth crop** is the small image cut out from inside one of those boxes.

We always use ground-truth crops, never a detector's guesses. That way, any difference we measure is about how the model *sees* the object, not about whether a detector found it correctly.

> **Analogy:** testing whether someone can identify birds. Instead of walking them through a forest (detection), we hand them tightly cropped photos of birds and ask them to name each one. It is an easier test, but it isolates the naming skill.

### 2.4 Centroid: the "average address" of a category

Take the 300 overhead car crops, embed them all, and average their addresses. The result is the **centroid**, the centre of the overhead-car cloud. Do the same for 300 drone car crops and you get the drone-car centroid.

> **Analogy:** the average location of everyone in a city who supports one football team. You can compute it for two cities and compare.

### 2.5 Drift: how far the centroid moves

**Drift** is the distance between the overhead centroid and the drone centroid of the *same* category, measured with the *same* model. We report it two ways:

- **L2 distance:** the straight-line distance between the two centroids.
- **Cosine distance:** how different their directions are (0 means identical direction). It is smaller in absolute value, but it tells the same story.

**Important rule:** both domains must be embedded by **the same model**. Comparing addresses from two different models is like comparing a street address in Delhi with one in London as if they were on the same map. They are not. (We found and fixed exactly this mistake; see Section 8.)

### 2.6 Spread, radius and the drift ratio

Some models bunch all examples of a category tightly together; others scatter them. A tightly bunching model will show smaller distances for everything, drift included, simply because its whole map is more compact.

- The **radius** of a category in one domain is the average distance from its examples to their centroid: how spread out the cloud is.
- The **drift ratio** is drift divided by the average radius of the two domains.

> **Analogy:** two cities, one drawn on a small map and one on a large map. A 2 cm gap on the small map may be a bigger real distance than a 3 cm gap on the large one. The drift ratio rescales each model by its own spread, so the comparison is fair.

A ratio near 1 means the cloud moved by about as much as the cloud is wide.

### 2.7 Between-category spacing: the yardstick (Y_c)

How big is "big" drift? We compare it with the distance between *different* categories in the same domain: how far "car" sits from "bus", "truck" and "building".

**Y_c** = (cross-domain drift of a category) ÷ (average distance between different categories).

If Y_c is above 1, the overhead car has moved further from the drone car than "car" sits from a *different category*. In other words, the platform change matters more to the model than the object's identity does.

> **Analogy:** if your accent changed so much between two cities that you sounded more different from yourself than two different people normally sound from each other, that would be a big change.

### 2.8 Permutation test: "could this be luck?"

Any two random samples have slightly different averages, so a non-zero drift on its own proves nothing. The **permutation test** checks whether the drift is bigger than chance would produce:

1. Pool all images from both domains.
2. Randomly reassign whole images to "overhead" and "drone", keeping the original counts.
3. Recompute the drift. Because the labels are now random, any drift here is pure chance.
4. Repeat 2,000 times.

If the real drift is larger than all 2,000 chance drifts, the chance of getting it by luck is below 1 in 2,000 (p < 0.001).

We shuffle **whole images**, not single crops, because crops from the same image are related: they share lighting, the camera and the scene.

> **Analogy:** to test whether a coin is biased, you compare its results with thousands of results from a fair coin. If it beats every fair-coin run, luck is a poor explanation.

### 2.9 Bootstrap: "how wobbly is this number?"

The **bootstrap** gives error bars (95% confidence intervals):

1. Draw a new pretend sample by picking *source images* at random, with replacement, from the real ones.
2. Recompute the number.
3. Repeat 2,000 times. The middle 95% of the results is the interval.

We used the *same* random draw for every model, so the differences between models are compared fairly ("paired"). We also redid it by grouping drone images into videos or file-name groups, to check that related frames don't make us overconfident. The conclusions did not change.

### 2.10 Zero-shot crop classification

For each crop, we give the model a few candidate labels, written as "a photo of a car", "a photo of a bus", and so on. The crop gets the label whose text address is closest to its image address. Nothing is trained: this is **zero-shot**.

**Accuracy** is the share of crops named correctly. **Recall** for a category is the share of that category's crops named correctly.

### 2.11 Text alignment: A_c and the margin Δm_c

These connect drift to the text side:

- **a_c**: how close a category's centroid is to its own text, for example how close the overhead-car centroid is to "a photo of a car".
- **A_c**: how much that closeness changes between overhead and drone.
- **Margin m_c**: how much closer the centroid is to the *correct* text than to the best *wrong* text. A positive margin means the average example would be named correctly.
- **Δm_c**: overhead margin minus drone margin. Negative means the correct label is harder to pick on overhead images.

> **Analogy:** in a quiz, the margin is how far ahead the right answer is of the most tempting wrong answer. A shrinking margin means the question has become more confusing.

---

## 3. The datasets

| Dataset | Platform | What we used | Why |
|---|---|---|---|
| **DOTA** | Overhead (Google Earth imagery) | 469 images (first training archive); small-vehicle and large-vehicle | Standard overhead benchmark. Only two vehicle classes overlap with drones. |
| **VisDrone** | Drone | 6,471 training images; car, van, bus, truck | Standard drone benchmark, rich vehicle labels |
| **xView** | Overhead (satellite) | First 5,000 training chips streamed from Hugging Face (500 for buildings); car, bus, truck, building | Fine-grained vehicle labels, so we could go beyond DOTA's two classes. Streamed, so no 20 GB download. |
| **UAVid** | Drone | Train and validation images; building only | VisDrone has no building class. UAVid gives pixel masks, so we built boxes from the coloured regions. |

**Three comparisons** were measured:

- **P1:** DOTA vs VisDrone, for small-vehicle and large-vehicle, with three models.
- **P2:** xView vs VisDrone, for car, bus and truck, with two models.
- **P3:** xView vs UAVid, for building, with two models.

**Sampling:** 300 crops per category per domain, random seed 42. For P1, VisDrone was first split 80/20 by image (seed 1234), and all three models were evaluated only on the held-out 20%.

**Label mapping (P1):**
- VisDrone car → small-vehicle.
- VisDrone van, truck and bus → large-vehicle.
- Bicycle is deliberately *not* used (see Section 8).

**Why van is missing from P2:** xView's closest class, Pickup Truck, gave only 25 crops in 5,000 images, below our 30-sample minimum.

---

## 4. The three models

| Model | What it is | Analogy |
|---|---|---|
| **CLIP ViT-B/32** (OpenAI) | General-purpose model trained on web photos and captions. 512-number embeddings. | A well-read generalist who has mostly seen ground-level photos |
| **RemoteCLIP ViT-L/14** | A larger CLIP trained further on remote-sensing (satellite/aerial) images and captions. 768-number embeddings. Used as downloaded, unchanged. | A specialist trained on aerial maps |
| **CLIP fine-tuned on VisDrone** (ours) | CLIP B/32 with its last two image layers retrained on 167,699 drone crops from 5,177 VisDrone images, using the prompt "a drone photo of a {class}". 3 epochs, learning rate 1e-5, batch 8. The text side was frozen. | The generalist after a short course on drone imagery |

The fine-tuned model saw only the 80% training part of VisDrone; we measured it on the 20% it never saw. Because it was trained on drone vehicle labels, its drone-side results are **not zero-shot**, and we always say so.

For the drift measurements, all three are used only as **image** encoders. Text is used only in the classification experiment.

---

## 5. The story, step by step

### Step 1: Phase-1 on the cloud (before August 2026)

The first experiment ran on Lightning AI Studio with CLIP ViT-B/32. It started with about 50 images per dataset to check the pipeline, then grew to about 300. Whole images were embedded and grouped by **the model's own guess** of what each image showed.

**Finding:** the same category landed in different places depending on the domain. This was the first sign of drift.

**Weakness, found later:** the groups were the model's guesses, not true labels, so the per-category numbers were noisy.

### Step 2: Moving to a local machine and reproducing it (13 Aug)

We moved everything to a local Windows laptop (RTX 3050, 4 GB GPU), set up the `hf` environment, downloaded DOTA and VisDrone, and re-ran Phase-1 end to end (`notebooks/01_phase1_reproduction.ipynb`). The results were in the same range as the cloud run, so the effect was not an accident of the cloud setup.

> **Why this matters:** a result you can only get on one machine isn't trustworthy. Reproducing it first is standard research practice.

### Step 3: Finding an invalid comparison (14 Aug)

A second cloud notebook (`model.ipynb`) had compared RemoteCLIP on DOTA against a fine-tuned CLIP on VisDrone. On inspection it was invalid:

- It used **two different models**, one per domain, so the addresses lived on two different maps.
- To force them together it fitted **two separate PCA projections** (a compression step), one per dataset. The resulting "distances" have no meaning.

> **Analogy:** measuring the distance between a point on a map of Delhi and a point on a map of London by laying the two paper maps on top of each other.

We dropped those numbers and never cite them.

### Step 4: Doing it correctly (14 Aug)

We rebuilt the comparison properly:

- the **same model** embeds both domains;
- **ground-truth** labels and boxes only;
- no PCA.

`run_remoteclip_vs_clip.py` does this for CLIP and RemoteCLIP. The reusable code went into `src/dataset.py` (crop extraction) and `src/embeddings.py` (model loading and encoding).

Practical lesson: GPU-heavy runs hung inside Jupyter but ran at full speed as plain scripts. Since then, heavy jobs are scripts and notebooks only load the saved results.

### Step 5: Fine-tuning, then catching two bugs (19–20 Aug)

We fine-tuned CLIP on VisDrone to make a "drone specialist" for a three-way comparison. The first result looked dramatic: a 63% drift reduction for large vehicles. Before trusting it, we checked and found two real problems:

1. **The bicycle bug.** VisDrone bicycles had been grouped with cars as "small-vehicle". DOTA's small vehicles are cars seen from far above, and bicycles can't even be resolved at that height. About 5.8% of the drone small-vehicle group was bicycles, quietly mixing two different objects. **Fix:** bicycle removed from the mapping.
2. **Train/test leakage.** The model had been fine-tuned on *all* VisDrone images and then tested on a sample from the *same* images. Part of its good score was memory, not understanding.
   > **Analogy:** giving students the exam questions as homework, then praising their exam results.

   **Fix:** split VisDrone 80/20 by image *before* doing anything, train only on the 80%, and test **all three models** only on the held-out 20%.

After both fixes, the fine-tuned model still had the lowest raw drift:
- **Large-vehicle** barely changed (0.0964 → 0.0955), so that result had not been inflated.
- **Small-vehicle** rose (0.1412 → 0.1585), so it had been somewhat inflated, as expected.

### Step 6: More categories with xView and UAVid (20 Aug)

DOTA only has small-vehicle and large-vehicle in common with drones, and adding more DOTA images would not add categories. So:

- **xView** (satellite) was added for car, bus and truck. It is streamed from Hugging Face, which avoids a 20 GB download.
- **UAVid** (drone) was added for buildings. Its labels are coloured pixel masks rather than boxes, so we found each connected coloured region and boxed it.

Building showed the **largest drift of anything measured**. That makes intuitive sense: from overhead, a building is a roof outline; from a drone, it is a facade with windows.

### Step 7: Mid-semester presentation and supporting analyses (21–22 Aug)

We wrote the presentation material (`docs/midsem/PROGRESS_SUMMARY.md`, `docs/midsem/PRESENTATION_SCRIPT.md`) and added:

- the **yardstick plots**: drift compared with the distance between categories;
- **drift-ratio plots**: drift compared with the spread inside each domain;
- **PCA/t-SNE pictures** using true labels. These are pictures only, never evidence: t-SNE distorts distances.
- a version of the main figure scaled as "% of CLIP", easier for a non-technical audience.

### Step 8: Technical report and final figures (19–21 Sep)

We wrote `docs/TECHNICAL_REPORT.md`, a complete factual extraction of the code, data and results. Every stored drift number was recomputed from the saved embeddings, and all 20 comparisons matched to four decimals. We also produced the four paper figures in `final_premst_results/`.

### Step 9: Laptop reset #1 and a new repository (early Oct)

Malware forced a Windows reset. The project survived on the D drive. All work was pushed to a new GitHub repository, `Caebyr/Capstone-New`, and the environment was rebuilt (`environment.yml`, `requirements.txt`).

### Step 10: The RAIT 2026 paper review and new experiments (12–13 Oct)

A teammate wrote a 6-page IEEE paper for RAIT 2026: *"Toward Measuring Visual Drift of Vision-Language Embeddings Between Overhead and UAV Imagery"*. We reviewed it against the code and results and produced two edit lists:

- **`docs/paper/RAIT_PAPER_CORRECTIONS.md`**: wording-only fixes. These were:
  - a clearer abstract;
  - an ambiguous label-mapping sentence;
  - why van is missing;
  - a heading that promised a "protocol correction" that wasn't described;
  - removing the non-finding "displacement is non-zero";
  - trimming repeated caveats.
- **`docs/paper/RAIT_PAPER_NEW_RESULTS.md`**: text for four new experiments, run on exactly the same crops and saved embeddings:
  1. **Crop manifests** (`build_crop_manifests.py`). The original runs hadn't recorded which crop each saved embedding came from. Because cropping is deterministic, we replayed it and wrote, for every row, the source image, box, label and group. We then re-encoded the crops and checked that they matched the saved rows.
  2. **Permutation tests and bootstrap intervals** (`drift_stats.py`), resampling whole images (Sections 2.8–2.9).
  3. **Zero-shot crop classification** (`classify_crops.py`), with one fixed prompt for every model and domain (Section 2.10).
  4. **A_c and Δm_c** (Section 2.11). The first draft had defined these but not computed them.

  `join_results.py` puts drift, accuracy and text alignment side by side.

  We deliberately skipped a multi-seed run before the deadline: the image-level error bars answer the same reviewer question better.

### Step 11: Pinning down the facts (13–14 Oct)

The paper editor needed exact facts, so we collected them in `docs/paper/PROJECT_FACTS_FOR_PAPER.md` and `docs/paper/CLARIFICATIONS_FOR_PAPER.md`:

- **Original environment recovered** from old session transcripts: Python 3.9.25, PyTorch 2.5.1 (CUDA 11.8), Transformers 4.49.0, OpenCLIP 3.3.0, NumPy 1.26.4, Pillow 11.1.0.
- **Every saved embedding was matched to its crop exactly.** Small differences on JPEG images came from the *JPEG decoding library*, not from wrong crops. Decoding with the original library (IJG libjpeg 9, as in the original conda build) reproduced every saved embedding to floating-point precision.
  > **Analogy:** two printers printing the same photo with very slightly different colours. The photo is the same; the printer differs.
- **CLIP weights:** the exact file and revision were identified. The two copies in the cache are bit-identical.
- **VisDrone file-name groups:** some are video clips and some are sets of unrelated still images, so we call them "file-name groups", not "videos". UAVid's sequences are real videos.
- **Other facts:** no evaluated crop is smaller than 16 pixels; all DOTA images used are Google Earth imagery; the RemoteCLIP file is byte-identical to the official release.

### Step 12: Submission (by 19 Oct)

The paper was submitted and is **under review**. The submitted PDF is `docs/paper/RAIT_paper_submitted.pdf`. Six authors: the five team members and Dr. Geeta Kasana.

How the paper is organised, and where each piece comes from in the repo:

| Paper | Content | Repo source |
|---|---|---|
| Table I | The three comparison pairs P1–P3 | `src/dataset.py`, `src/crop_sets.py` |
| Fig. 1 | Pipeline diagram (drawn for the paper) | — |
| Eqs. (1)–(4) | Centroid, radius, L2/cosine drift, drift ratio, B_c and Y_c | `src/stats.py`, `drift_stats.py` |
| Eqs. (5)–(6) | a_c, A_c, margin m_c, Δm_c | `classify_crops.py` |
| Table II | Drift per category and encoder, with radius | `final_premst_results/results_table.md` |
| Fig. 2 | Raw L2 and drift ratio as % of CLIP (drawn for the paper) | same numbers as `results_table.md` |
| Table III | Y_c | `results/drift_stats/drift_stats_table.md` |
| Table IV | Four-way crop classification and Δm_c | `results/classification/classification_bootstrap_table.md` |
| Fig. 3 | Joint t-SNE | `final_premst_results/fig2_tsne_building_domain_split.*` (21 Sep render) |
| §III "Implementation details" | Software versions | `docs/paper/PROJECT_FACTS_FOR_PAPER.md` |
| §IV p-values and intervals | Permutation and bootstrap | `results/drift_stats/`, `results/classification/` |

The paper words the main finding as: *"reduced visual embedding drift does not necessarily correspond to improved recognition performance."*

### Step 13: Laptop reset #2 and cleanup (25 Nov)

There was a second reset, and the new Windows user is `Pc`. We reinstalled Miniconda and recreated `hf` with the same pinned versions, then verified it:

- GPU visible;
- all packages at the pinned versions;
- checkpoint checksums correct;
- all three models load the way the project code loads them;
- a smoke test reproduced the saved embeddings exactly for PNG images, and within the known JPEG-decoder range for JPEG images.

We then cleaned up the repository:
- **Removed:** empty placeholders, the first leaky fine-tuning run's checkpoints and embeddings, unused intermediate checkpoints, the broken seed script, and an archive that duplicated extracted files.
- **Archived:** superseded results, moved to `results/archive/`.
- **Moved:** documents into `docs/`, and the midsem plot scripts into `scripts/plots/`.
- **Restored:** the 21 Sep paper figures, which a later re-run had overwritten.
- **Updated:** outdated text in the docs.

---

## 6. The results, explained

All values come from 300 crops per category per domain, seed 42. Brackets are 95% image-bootstrap intervals.

### 6.1 How far each category moves

| Comparison | Category | CLIP L2 | RemoteCLIP L2 | Fine-tuned L2 |
|---|---|---|---|---|
| DOTA vs VisDrone | small-vehicle | 0.246 | 0.189 (−23%) | 0.159 (−36%) |
| DOTA vs VisDrone | large-vehicle | 0.260 | 0.228 (−12%) | 0.096 (−63%) |
| xView vs VisDrone | car | 0.356 | 0.302 (−15%) | — |
| xView vs VisDrone | bus | 0.326 | 0.280 (−14%) | — |
| xView vs VisDrone | truck | 0.284 | 0.237 (−17%) | — |
| xView vs UAVid | building | 0.542 | 0.378 (−30%) | — |

**In plain words:** every category moves. Building moves most, and RemoteCLIP has a smaller raw move than CLIP for all six categories.

### 6.2 Is the movement real? Yes.

In all 14 category–model combinations, the real drift beat every one of 2,000 random reshuffles (p < 0.001). The closest case was fine-tuned large-vehicle, which we re-ran with 20,000 reshuffles: p = 0.0003. Drift is not a fluke of which crops were picked.

### 6.3 Is it big? Bigger than the gap between categories.

| Category (xView vs drone) | CLIP Y_c | RemoteCLIP Y_c |
|---|---|---|
| car | 1.95 [1.65, 2.21] | 2.09 [1.78, 2.34] |
| bus | 1.81 [1.48, 2.09] | 2.30 [1.94, 2.58] |
| truck | 1.50 [1.21, 1.80] | 1.81 [1.43, 2.18] |
| building | 2.32 [2.17, 2.42] | 2.12 [1.87, 2.32] |

All eight values are 1.50–2.32, and every interval is above 1. **A car seen from above is further from a car seen by a drone than "car" is from "bus".** To the model, the change of platform matters more than the change of object.

Caveat: car, bus and truck are naturally close to each other, which keeps the yardstick small, and the yardstick mixes two drone datasets.

### 6.4 Is RemoteCLIP really better? Only partly.

RemoteCLIP's clouds are 11–16% tighter than CLIP's in each domain. Once drift is divided by spread (the drift ratio, Section 2.6):

- **Small-vehicle:** RemoteCLIP's ratio is 13.4% lower than CLIP's, and the interval excludes zero.
- **Building:** 19.7% lower, and the interval excludes zero.
- **Large-vehicle, car, bus and truck:** within about 2% of CLIP, and the intervals include zero, so there is no real difference.

**Plain words:** much of RemoteCLIP's "less drift" comes from it bunching everything more tightly, not from aligning the two platforms better. This is why the paper says raw comparisons must always be read alongside normalised ones.

### 6.5 What about the fine-tuned model?

It has the lowest raw drift, but it also shrinks its clouds by roughly 40–50%. Its drift ratio:

- **Small-vehicle:** rises to 0.8165 (CLIP 0.6498), but the interval for the change (−1.5% to +52.2%) includes zero, so it is inconclusive.
- **Large-vehicle:** falls to 0.4132 (CLIP 0.6235), a genuine improvement relative to spread.

### 6.6 Can the models name the objects? Much worse from overhead.

**Four-way task (car, bus, truck, building), xView vs VisDrone/UAVid:**

| Model | Overhead accuracy | Drone accuracy | Gap |
|---|---|---|---|
| CLIP | 32.9% | 60.2% | −27.3 points [−31.6, −23.3] |
| RemoteCLIP | 27.3% | 51.8% | −24.6 points [−30.1, −19.3] |

Random guessing among four labels gives 25%, so the overhead accuracies are barely above guessing.

What goes wrong is striking:

- **CLIP** calls almost every overhead vehicle a "car": 813 of 900 overhead vehicle crops.
- **RemoteCLIP** calls almost every overhead vehicle a "building": 839 of 900.

The margin for the correct label (Δm_c) is smaller on the overhead side in all eight category–model cases, and the interval excludes zero in seven.

**Two-way task (small vs large vehicle), DOTA vs VisDrone:** no model shows a significant accuracy gap. CLIP is weak at this task on both platforms; it labels 98% of overhead crops "small vehicle". The fine-tuned model reaches 84.8% overhead and 84.2% drone accuracy, but it was trained on this very task.

**Plain words:** the shift we measure comes with a large loss in recognition on overhead images. Yet **RemoteCLIP, which drifts less, does not recognise better**. Smaller drift does not mean better recognition.

### 6.7 Does more drift mean more accuracy loss? Not across categories.

Across the four categories, we ranked drift against accuracy loss. The rank correlation is about 0.02, essentially none. Drift does rank with the text-alignment changes (Spearman 0.8 for CLIP and 0.4 for RemoteCLIP against A_c and |Δm_c|). With only four categories, these correlations are descriptive, not proof.

**Takeaway:** drift tracks how the image moves relative to the text, but drift size alone does not predict how badly recognition fails.

---

## 7. What we can and cannot claim

| Statement | Status |
|---|---|
| Drift is larger than sampling chance | **Supported**: all 14 cells beat 2,000 image-level permutations |
| Drift is larger than the gap between categories | **Supported** for four categories and two models (all intervals above 1) |
| RemoteCLIP has lower raw drift than CLIP | **Supported** for all six categories |
| RemoteCLIP has lower *normalised* drift | **Only** for small-vehicle and building |
| Fine-tuning reduces drift through better alignment | **Unresolved**: tighter clusters; small-vehicle change inconclusive |
| Drift comes with a loss of crop recognition | **Supported** for the four-way task (25–27 points lower overhead); not for the two-way task |
| Less drift means better recognition | **Not supported**: RemoteCLIP is no better overhead |
| More drift predicts more accuracy loss | **Not supported** across four categories (ρ ≈ 0.02) |
| Drift is caused by altitude | **Not supported**: sensor, resolution, geography and dataset all change together |
| Drift predicts *detection* failure | **Not tested**: no detector was run |
| Stable across seeds and full datasets | **Not tested**: one seed, image-level resampling only |

**Words to avoid:**
- "Detection accuracy" for the classification results. Say "ground-truth crop classification".
- "Drift causes errors". Say "drift accompanies errors".
- "Altitude drift" as a proven cause. Say "a dataset-level change of platform".

---

## 8. Mistakes we caught and fixed

These are worth presenting openly. Catching your own mistakes before reporting is what makes results believable.

| # | Mistake | How we noticed | Fix | Effect |
|---|---|---|---|---|
| 1 | Cross-model PCA comparison (two models, two separately fitted PCAs) | Reading the recovered cloud notebook | Same model for both domains, no PCA | Old numbers discarded |
| 2 | Phase-1 grouped images by the model's guessed labels | Code review | All later work uses ground-truth labels | Phase-1 cited only as "drift exists" |
| 3 | Bicycles counted as small vehicles | Checked label composition (~5.8% bicycles) | Bicycle removed from mapping | Cleaner small-vehicle category |
| 4 | Fine-tuned model tested on images it was trained on | Checking before the midsem | 80/20 split by image before cropping; all models tested on the held-out 20% | Small-vehicle result shrank; large-vehicle held |
| 5 | Saved embeddings had no record of which crop they came from | Paper review | Rebuilt crop manifests and verified them row by row | Enabled image-level statistics |
| 6 | Paper listed the rebuilt environment's versions, not the original's | Fact-checking | Original versions recovered from session logs | Factual fix in the paper |
| 7 | "Video-sequence resampling" for VisDrone was inaccurate | Checking file-name groups | Called "file-name groups"; only UAVid has true videos | Wording fix, no number changed |
| 8 | The seed-check script (`verify_drift_seeds.py`) used the leaky checkpoint and all VisDrone images | Technical report audit | Never used for any result; removed in the Nov cleanup. A correct multi-seed script is future work. | Open item |

---

## 9. Limitations

- **Platform and dataset are tangled.** Overhead vs drone is also DOTA/xView vs VisDrone/UAVid: different cameras, cities, years and labelling styles. We cannot say how much is altitude.
- **Pilot scale.** 300 crops per category, one sampling seed, 469 DOTA images, 5,000 xView chips.
- **The DOTA sample is clumpy.** The 300 crops per class come from only 68–119 images, with up to 43 from a single image. Resampling by image handles this, but it limits how much independent evidence there is.
- **One prompt.** RemoteCLIP is normally used with remote-sensing prompts. Its "everything is a building" habit might partly be the prompt.
- **Classification is not detection.** There is no box finding, no background and no extra classes.
- **Two drone datasets are pooled.** VisDrone supplies the vehicles and UAVid the buildings.
- **xView chips can't be grouped by scene**, and three exact duplicate crop pairs exist from overlapping chips.
- **Centroids are a first-order summary.** Two clouds can have the same centre but different shapes.

---

## 10. Where the project goes next

The planned next stage, from the 12 Oct review and the project plan, in order of value:

1. **Same-domain baseline.** Measure drift between two datasets from the *same* platform: DOTA vs xView (both overhead) and VisDrone vs UAVid or UAVDT (both drone). If overhead-vs-drone drift is much larger than this, the effect is more than just "different dataset". This addresses the biggest weakness.
2. **Same-backbone comparison.** Add OpenAI CLIP ViT-L/14, the same architecture as RemoteCLIP, so the remote-sensing training is the only difference.
3. **Controlled altitude and resolution tests.** UAVDT labels flight altitude, so we can test altitude within one dataset. We can also shrink drone crops to overhead resolution and re-measure.
4. **Real detection.** Use the teammate's SAM-based pipeline, or an off-the-shelf open-vocabulary detector, to test whether per-category detection loss follows our measures.
5. **More prompts and encoders, multiple seeds, full datasets** (DOTA parts 2–3 are on disk, still zipped).
6. **Distribution-level measures** (shape, not just centre) and a fine-tuned model for P2/P3.
7. **Longer term:** an altitude-aware alignment module that *reduces* drift.

---

## 11. Questions an examiner might ask

**"Isn't any two samples' average different? Why is non-zero drift interesting?"**
It isn't on its own, and that's why we ran the permutation test. Drift beat all 2,000 random reshufflings in every case.

**"Your cosine numbers look tiny (0.03–0.18). Is that meaningful?"**
Embeddings live on a sphere where everything is fairly close, so absolute values are small. The meaningful comparison is with the gap between categories: drift is 1.5–2.3 times larger.

**"RemoteCLIP has less drift, so it's better for aerial work, right?"**
Not necessarily. Part of its lower drift is tighter clustering. After normalising, it wins only for two categories, and on overhead crops it recognises objects no better than CLIP: it calls most vehicles buildings.

**"Did you test a detector?"**
No. We tested ground-truth crop classification, which is the easier naming step with perfect boxes. Even there, overhead recognition is barely above chance for the four-way task. Detection is the planned next step.

**"Can you say altitude causes the drift?"**
No. The datasets differ in sensor, resolution, location and labelling as well as altitude. A same-domain baseline and altitude-labelled data (UAVDT) are the planned tests.

**"Why fine-tune only the last layers?"**
To adapt the model to drone imagery cheaply on a 4 GB GPU while keeping its general knowledge, and with the text side frozen so the text addresses stay identical to CLIP's.

**"How do you know the saved embeddings match the crops you describe?"**
We replayed the deterministic crop extraction, re-encoded every crop, and matched each one to its saved row. With the original JPEG library, every row matches to floating-point precision.

**"Why resample whole images, not individual crops?"**
Crops from the same image share lighting, camera and scene, so they aren't independent. Treating them as independent would make the error bars look falsely narrow.

---

## 12. Where everything lives

| What | Where |
|---|---|
| Crop extraction, label mappings, VisDrone split | `src/dataset.py` |
| Model loading and encoding, fine-tuning helpers | `src/embeddings.py` |
| Crop lists, tasks and model names for the October analyses | `src/crop_sets.py`, `src/stats.py` |
| CLIP vs RemoteCLIP, DOTA vs VisDrone | `run_remoteclip_vs_clip.py` → `results/remoteclip_vs_clip/` |
| Fine-tuning and three-way comparison | `notebooks/04_finetune_visdrone_clip.ipynb` |
| xView vs VisDrone; UAVid buildings | `run_xview_vs_visdrone.py`, `run_uavid_building.py` → `results/xview_vs_visdrone/` |
| Crop manifests | `build_crop_manifests.py` → `results/crop_manifests/` |
| Permutation tests and bootstrap | `drift_stats.py` → `results/drift_stats/` |
| Crop classification and A_c/Δm_c | `classify_crops.py` → `results/classification/` |
| Joined table and correlations | `join_results.py` → `results/joined/` |
| Paper figures | `final_premst_results/` (`make_final_figures.py`) |
| Midsem plots (yardstick, drift ratio, PCA/t-SNE) | `scripts/plots/` → `results/category_yardstick/`, `results/pca_tsne/` |
| Superseded results (don't cite) | `results/archive/` |
| The submitted paper | `docs/paper/RAIT_paper_submitted.pdf` |
| Facts for the paper | `docs/paper/PROJECT_FACTS_FOR_PAPER.md`, `docs/paper/CLARIFICATIONS_FOR_PAPER.md`, `results/paper_facts/` |
| Paper edit instructions | `docs/paper/RAIT_PAPER_CORRECTIONS.md`, `docs/paper/RAIT_PAPER_NEW_RESULTS.md` |
| Working log and project rules | `docs/PROJECT_CONTEXT.md` |
| Full technical extraction (Sept) | `docs/TECHNICAL_REPORT.md` |
| Midsem material | `docs/midsem/` |
| How the October numbers were made | `results/NEW_EXPERIMENTS_2026-10.md` |
| Environment | `environment.yml`, `requirements.txt` (conda env `hf`, Python 3.9) |

Large files are not in git: `data/`, `embeddings/` and `models/`.

---

## 13. Timeline of changes

| Date (2026) | What happened |
|---|---|
| Before Aug | Phase-1 on Lightning AI: CLIP B/32, whole images, model-guessed labels; drift first observed |
| 13 Aug | Moved to a local laptop; `hf` environment; Phase-1 reproduced locally |
| 14 Aug | Cross-model PCA comparison found invalid; redone with one model per comparison and ground-truth labels; `src/` modules written |
| 19 Aug | SAM integration with the teammate's work planned; bicycle mapping bug fixed |
| 19–20 Aug | Train/test leakage found and fixed (80/20 held-out split); three-way comparison re-run |
| 20 Aug | xView (streamed) and UAVid integrated; four-category comparison |
| 21–22 Aug | Mid-semester presentation; yardstick, drift-ratio, normalised and PCA/t-SNE plots |
| 19–21 Sep | Technical report; all numbers re-verified; final paper figures |
| early Oct | Laptop reset #1; new repo `Caebyr/Capstone-New`; environment rebuilt |
| 12 Oct | RAIT paper reviewed; corrections list; crop manifests, permutation tests, bootstrap, crop classification, A_c/Δm_c |
| 13–14 Oct | Fact sheet; original environment recovered; JPEG-decoder cause found; crop identity proven exact |
| by 19 Oct | Paper submitted to RAIT 2026 (under review) |
| 25 Nov | Laptop reset #2; `hf` environment rebuilt and verified; October work committed and pushed; repository cleaned up and reorganised |
