# Corrections for the RAIT 2027 paper "Toward Measuring Visual Drift of Vision-Language Embeddings Between Overhead and UAV Imagery"

> **Update (Oct 12):** new results have been added in a second file, `RAIT_PAPER_NEW_RESULTS.md`. Apply this file first, then that one. That file supersedes change 1 (abstract), the Table IV row in change 10, and the "Out of scope" note about significance tests and A_c/Δm_c. Its item N4a also fixes a factual error in Section III-E (the software version list).

## Instructions for Claude

You are editing the source of a 6-page IEEE conference paper (two-column IEEEtran format) before submission to RAIT 2027. Apply the numbered changes below to the paper source.

**What this list is.** A teammate reviewed the PDF against the project's code and result files. Every change here is wording only: corrections, clarifications and cuts. None of them needs a new experiment or new numbers.

**Rules**

1. **Do not change any number, table value, figure, equation or reference.** All of them were checked against the project's result files and are correct.
2. **Do not add claims** beyond the replacement text given here. In particular, do not add significance tests, confidence intervals, p-values or new citations. Those need new work and are out of scope (see the last section).
3. **Finding the text.** The "Current" text below was copied from the compiled PDF, so the source may differ slightly: LaTeX markup (`--`, `\emph{}`, `$...$`, `~\cite{}`), line breaks, or hyphenation. Match on the distinctive words. If a passage can't be found, skip it and report that.
4. **Keep the length.** RAIT allows 6 pages including references, and the paper is at exactly 6. Changes 1 and 12–15 shorten the text and change 9 lengthens it, so the net change should be about zero. After editing, compile and confirm the PDF is still 6 pages or fewer. If it overflows, use the shorter option given in change 9.
5. **When done**, list each change as applied, skipped (with the reason) or modified (with what you did instead).

**Already verified — do not "fix" these**

- **Fine-tuning prompt:** `"a drone photo of a {}"` is the prompt actually used. It is not a typo for `"a photo of a {}"`.
- **Numbers:** all values in Tables II–III, Fig. 2 and Section IV match the result files, including the percentages (12.3–30.2%, 26.1–53.5%, −13.4%, −19.7%, 35.6%, 63.2%, 62.7%, 88.2%, 49%/44%, 11–16%).
- **References:** references [2], [3], [6] and [9] are real papers with correct arXiv IDs and venues.
- **Split counts:** 5,177 + 1,294 = 6,471 images, and split seed 1234, are correct.

---

## Changes

### 1. Abstract — rewrite to state the findings plainly

**Why.** About 40% of the current abstract is disclaimers, so the findings are hard to see on a skim. The caveats are kept in Section V and Table IV. The current abstract also opens its results with "All sampled centroid displacements are nonzero", which is true of any two finite samples and so is not a finding (see change 10). The new version uses only numbers already in the paper.

**Current:**
> Open-vocabulary detectors depend on visual–text alignment under changes in imaging conditions. We study a visual component of this problem by measuring category-conditioned centroid displacement between overhead and unmanned aerial vehicle (UAV) imagery. The same frozen encoder embeds ground-truth object crops from both domains. We report L2 and cosine displacement, displacement normalized by within-domain spread, and a comparison with between-category spacing. The pilot covers six categories, CLIP and RemoteCLIP, and a CLIP encoder fine-tuned on drone imagery for two of the categories, with 300 crops per category and domain and one sampling seed. All sampled centroid displacements are nonzero; their statistical significance is not assessed. For four categories, cosine displacement is 1.50 to 2.32 times the mean between-category cosine dissimilarity in the pooled dataset comparison. RemoteCLIP has lower raw displacement than CLIP for all six categories, but its spread-normalized ratio differs by at most 2.2% for four categories. These reductions in raw displacement accompany smaller within-domain radii and do not establish improved semantic alignment. Fine-tuned CLIP has the lowest raw displacement for both evaluated categories, while its normalized ratio increases for small vehicles and decreases for large vehicles. The datasets differ in sensor, resolution, geography, and annotation conventions, so the measurements do not isolate altitude. Text-anchored measures are defined but not computed, and detection accuracy is not evaluated.

**Replace with:**
> Open-vocabulary detectors rely on visual–text alignment that may not survive a change of imaging platform. We measure one visual component of this problem: how far a category's embedding centroid moves between overhead and unmanned aerial vehicle (UAV) imagery when one frozen encoder embeds ground-truth object crops from both domains. We report L2 and cosine centroid displacement, displacement normalized by within-domain spread, and displacement relative to between-category spacing. The pilot covers six categories across three dataset pairs (DOTA and xView overhead; VisDrone and UAVid UAV), with CLIP, RemoteCLIP and, for the two DOTA–VisDrone categories, a CLIP encoder fine-tuned on separate VisDrone images, using 300 crops per category and domain and one sampling seed. For four categories, cross-domain cosine displacement is 1.50 to 2.32 times the mean between-category cosine dissimilarity. RemoteCLIP lowers raw displacement for all six categories, but its spread-normalized ratio is within 2.2% of CLIP's for four of them and lower only for small vehicles and buildings, alongside 11–16% smaller within-domain radii. Fine-tuned CLIP gives the lowest raw displacement, while its normalized ratio rises for small vehicles and falls for large vehicles. Raw cross-encoder comparisons therefore depend on within-domain concentration and should be read alongside normalized measures. Because the datasets differ in sensor, resolution, geography and annotation, the measurements describe a dataset-level domain change rather than altitude alone.

### 2. Introduction, paragraph 1 — define "satellite" at first use

**Why.** The title and abstract say "overhead", but the body says "satellite". The paper only explains that "satellite" is shorthand for an overhead proxy domain in Section III-A, after the word has already been used. DOTA contains aerial as well as satellite imagery.

**Current:**
> Satellite and drone (unmanned aerial vehicle, UAV) images differ from ground-level photographs in scale, resolution and viewpoint, and the assumption may not survive the change.

**Replace with:**
> Overhead (satellite or high-altitude aerial) and drone (UAV) images differ from ground-level photographs in scale, resolution and viewpoint, and the assumption may not survive the change.

Leave the rest of the paper's "satellite" / S-subscript usage as it is. The existing shorthand sentence in III-A covers it.

### 3. Introduction, paragraph 2 — novelty claim

**Why.** "Among the aerial detection works we reviewed [2]–[4], we found no…" rests the novelty claim on three papers, which a reviewer will notice. The replacement says what those works do and puts the contribution in applying standard measures, rather than in a literature gap.

**Current:**
> Among the aerial detection works we reviewed [2]–[4], we found no per-category embedding-space displacement measured independently of detection accuracy. This observation is restricted to those reviewed studies and is not a claim that the underlying distance measures are new.

**Replace with:**
> Recent aerial open-vocabulary work [2]–[4] evaluates detection accuracy or proposes alignment objectives; none of these studies reports per-category embedding-space displacement measured independently of detection accuracy. The distance measures themselves are standard; our contribution is to apply them per category across a change of sensing domain.

### 4. Introduction, final paragraph — contribution (iii)

**Why.** As written, the contribution reads like a non-result.

**Current:**
> (iii) an analysis showing that raw and normalized comparisons can disagree without identifying the mechanism responsible.

**Replace with:**
> (iii) evidence that raw displacement comparisons across encoders depend on within-domain concentration, so spread-normalized measures should be reported alongside them.

### 5. Section III-A — fix the ambiguous P1 label mapping

**Why.** "car maps to small-vehicle and van, truck and bus map to large-vehicle" can be read as *car → small-vehicle and van*. The code maps car → small-vehicle and van/truck/bus → large-vehicle.

**Current:**
> for P1, VisDrone car maps to small-vehicle and van, truck and bus map to large-vehicle (VisDrone bicycle is not used);

**Replace with:**
> for P1, VisDrone car maps to small-vehicle, and VisDrone van, truck and bus map to large-vehicle (VisDrone bicycle is not used);

### 6. Section III-A — say why van is missing from P2

**Why.** VisDrone has a van class, but P2 evaluates only car, bus and truck. Reviewers will ask why. In the code, xView's closest class (Pickup Truck) was mapped to van but gave too few crops, so van was excluded.

**Current:**
> with VisDrone's car, bus and truck classes mapping one to one;

**Replace with:**
> with VisDrone's car, bus and truck classes mapping one to one (van is not evaluated in P2 because xView's closest class, Pickup Truck, yields too few crops in the scanned images);

### 7. Section III-A — DOTA archive wording

**Why.** "the first of three image archives" is vague. DOTA's training images are distributed as three archives (part1–part3), and the project used part1 (469 images).

**Current:**
> DOTA [11], [12] (469 images, the first of three image archives)

**Replace with:**
> DOTA [11], [12] (the 469 images in the first of the three DOTA training-image archives)

Note for the author: if you know the release (DOTA-v1.0 or v1.5), name it here. Otherwise leave it as above.

### 8. Section III-B — split the run-on sentence and move the image-encoder statement

**Why.** One sentence chains three clauses with semicolons, and the point that every encoder is used only as an image encoder sits in the middle of the fine-tuning description, where it doesn't belong.

**Current:**
> Each training crop is paired with the prompt "a drone photo of a {}", with the crop's class name filling the blank; CLIP and RemoteCLIP are used only as image encoders in this pilot; no text prompt is applied to them for the drift measures in Section IV.

**Replace with:**
> Each training crop is paired with the prompt "a drone photo of a {}", with the crop's class name filling the blank.

Then add this sentence at the end of the first paragraph of III-B, right after "…and gives 768 dimensions.":
> All three encoders are used only as image encoders for the drift measures in Section IV; no text prompt is applied at measurement time.

(The fine-tuned encoder is also measured with image embeddings only, so "All three" is accurate.)

### 9. Section III-E — the heading promises a "Protocol Correction" that is never described

**Why.** The heading is "Implementation and Protocol Correction", but the section never describes a correction. Two corrections really were made to P1 before the reported numbers were produced. Describing them strengthens the paper, because catching your own data leakage is something reviewers respect.

**Preferred option:** keep the heading, and add this paragraph at the end of III-E:
> Two problems in an earlier run of P1 were corrected before the values reported here were produced. First, VisDrone bicycle had been mapped to small-vehicle; because DOTA's overhead small-vehicle class contains cars rather than bicycles, this mixed visually distinct objects, and bicycle was removed from the mapping. Second, the fine-tuned encoder had been trained on crops from all VisDrone images and then evaluated on crops drawn from the same pool. VisDrone is now split by image (80/20, split seed 1234) before any crop is extracted; fine-tuning uses only the 80% pool, and all three encoders are evaluated on the held-out 20%.

**Shorter option, if the paper overflows 6 pages:** rename the heading to **"E. Implementation"** and don't add the paragraph.

### 10. Section IV-A and Table IV — remove the "nonzero displacement" claim

**Why.** Any two finite samples have different centroids, so "nonzero displacement" is not evidence of anything. No significance test was run, so the honest statement is "not tested". Don't add any test result here; a test is planned as separate work.

**Section IV-A — delete these two sentences:**
> Every evaluated category–encoder combination has a nonzero sampled centroid displacement. This is a descriptive observation, not a rejection of equal population centroids.

The paragraph should then read "Table II lists all values. Under CLIP the cosine drift runs from…".

**Table IV, first row — current:**
> Sampled centroid displacement is nonzero | Observed for six categories; statistical significance not assessed

**Replace with:**
> Displacement exceeds sampling variation | Not tested: no permutation or resampling test was run

### 11. Section IV-B — weak sentence

**Current:**
> The normalized ratio differs (Fig. 2(b)).

**Replace with:**
> The normalized ratio tells a different story (Fig. 2(b)).

### 12. Section IV-B — trim a repeated hedge

**Why.** The same caveat appears in the Discussion and in Table IV. This change saves space.

**Current:**
> Algebraically, these changes largely cancel in the normalized ratio; this does not identify a causal compression mechanism.

**Replace with:**
> Algebraically, these changes largely cancel in the normalized ratio.

### 13. Section IV-C — delete a repeated hedge

**Why.** This repeats Section V, paragraph 2 ("these observations do not identify collapse or improved alignment") and Table IV.

**Delete this sentence (the last in IV-C):**
> These statistics do not establish information loss, representation collapse, or improved visual–text alignment.

### 14. Section IV-D — ambiguous phrase

**Why.** "small vehicle-category spacing" reads as if it refers to the "small-vehicle" category.

**Current:**
> The small vehicle-category spacing and the pooled datasets in the denominator must be considered

**Replace with:**
> The small spacing between vehicle categories and the pooled datasets in the denominator must be considered

### 15. Hedge trims that save space

Each of these repeats a caveat stated elsewhere.

- **Section II-C, Wortsman/WiSE-FT sentence.**
  Current: "This motivates evaluating adaptation beyond in-domain performance, but it does not establish that the smaller radii observed here reflect representation collapse or lost semantic information."
  Replace with: "This motivates evaluating adaptation beyond in-domain performance."
- **Section IV-E, last sentence.**
  Current: "The figure is illustrative and is not independent evidence for the magnitude or significance of drift."
  Replace with: "The figure is illustrative only."
- **Fig. 2 caption.** Delete the final sentence: "Lower values do not establish better semantic alignment." (Section III-C already says this.)

Keep these hedges, which matter: the Introduction's "Visual centroid displacement alone does not establish semantic confusion.", the whole of Section V, Table IV, and the Conclusion.

### 16. Table II caption — say what "Rad." is

**Why.** "RAD. = L2/RATIO" gives an identity, not a definition. Rad. is the mean of the two intra-domain radii.

**Current (it renders in small caps):**
> MEAN INTRA-DOMAIN RADIUS (RAD. = L2/RATIO)

**Replace with:**
> mean intra-domain radius (Rad. = ½(r^(S) + r^(U)) = L2/Ratio)

Use the same math markup as Eq. (1) for r^(S) and r^(U).

### 17. Section III-E — the code availability statement has no link

**Current:**
> The full environment snapshot is included with the source.

**Replace with ONE of these.** Ask the author which applies, and use the first only if the repository is public:
- "Code, configuration and the full environment snapshot are available at https://github.com/Caebyr/Capstone-New."
- If RAIT review is double-blind, or the repo is private: "Code and the full environment snapshot will be made available on publication."

---

## Out of scope — do NOT do these

These were recommended in the review but need new experiments or new sources. They are being handled separately.

- Adding a permutation test, bootstrap confidence intervals or p-values anywhere in the paper.
- Image-level resampling, multi-seed results, or any change to "Stability across seeds… Not tested" in Table IV.
- Adding new citations (e.g. RS5M/GeoRSCLIP, SkyScript, GRAFT).
- Regenerating or editing any figure. Fig. 3 in particular must stay the committed version.
- Changing "satellite" to "overhead" throughout the paper, including the S subscript. Only change 2 is intended.

## Final checks after editing

1. The paper compiles, is 6 pages or fewer including references, and the two-column layout is intact (no table or figure pushed to an extra page).
2. No number in any table, figure or sentence has changed. Compare the old and new PDFs.
3. The abstract is 250 words or fewer (IEEE guideline). The replacement in change 1 is about 220 words.
4. The authors' emails still show their underscores (e.g. `asachdev_be23`). LaTeX needs `\_`.
