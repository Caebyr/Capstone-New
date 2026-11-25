# New results for the RAIT 2026 paper: significance tests, crop classification, text alignment

## Instructions for Claude

You are editing the source of a 6-page IEEE conference paper (IEEEtran, two columns), "Toward Measuring Visual Drift of Vision-Language Embeddings Between Overhead and UAV Imagery". A separate file, `RAIT_PAPER_CORRECTIONS.md`, holds wording-only corrections.

**Order of work**
1. Apply `RAIT_PAPER_CORRECTIONS.md` first, **except** the changes this file marks as superseded (listed below).
2. Then apply this file. Where the two conflict, this file wins.

**What this file is.** New experiments have been run on exactly the crops and cached embeddings behind the paper's Tables II–III:
- image-level permutation tests and bootstrap confidence intervals for the drift measures;
- a zero-shot classifier on the ground-truth crops;
- the text-anchored measures A_c and Δm_c, which the paper currently says are "defined, not computed".

Every number below comes from the project's result files (`results/drift_stats/`, `results/classification/`, `results/joined/`). **Do not change, round differently, or add numbers.** If you need a number that is not here, leave a `% TODO` comment rather than inventing one.

**Rules**
1. **Describe the new experiment as "ground-truth crop classification".** Never call it detection or detection accuracy.
2. **Keep the 6-page limit** (references included). The additions are about half a page; the cuts listed under "Fitting into six pages" should cover it. Compile and check the page count.
3. **Keep the paper's notation.** Match the existing macros for μ, r, D, A_c, Δm_c.
4. **When finished,** list each item as applied, skipped (with the reason), or modified.

### Items in RAIT_PAPER_CORRECTIONS.md that this file supersedes
- **Change 1 (abstract):** use the abstract in item N1 below instead.
- **Change 10, Table IV row only:** use the Table IV rows in item N8. The IV-A deletion in change 10 still applies, then add the sentence from item N4.
- **Change 17 (code availability):** still applies. Item N3 adds a related sentence.
- **The "Out of scope" section:** significance tests, confidence intervals and A_c/Δm_c are now **in scope**, but only through this file. The other out-of-scope items (new citations, figure regeneration, global satellite→overhead rename) still stand.

---

## N1. Abstract (replaces the abstract entirely, about 240 words)

```latex
Open-vocabulary detectors rely on visual--text alignment that may not survive a change of imaging platform. We measure how far a category's embedding centroid moves between overhead and unmanned aerial vehicle (UAV) imagery when one frozen encoder embeds ground-truth object crops from both domains, and whether that movement accompanies a loss of zero-shot category recognition. The pilot covers six categories across three dataset pairs (DOTA and xView overhead; VisDrone and UAVid UAV) with CLIP, RemoteCLIP and, for two categories, a CLIP encoder fine-tuned on separate VisDrone images, using 300 crops per category and domain. For every category and encoder, centroid displacement exceeds all 2,000 image-level permutations, and for four categories it is 1.50 to 2.32 times the mean between-category cosine dissimilarity, with image-bootstrap intervals above 1. RemoteCLIP lowers raw displacement for all six categories, but its spread-normalized ratio is lower only for small vehicles and buildings. In ground-truth crop classification over car, bus, truck and building with a fixed prompt, accuracy is 32.9\% on overhead and 60.2\% on UAV crops for CLIP, and 27.3\% and 51.8\% for RemoteCLIP, and the correct-text margin of the category centroid is smaller on the overhead side in seven of eight cases. Lower displacement therefore does not imply better recognition: RemoteCLIP labels most overhead vehicles as buildings. Across categories, displacement ranks with text-anchor change but not with accuracy loss. Because the datasets differ in sensor, resolution and geography, these effects describe a dataset-level domain change rather than altitude alone.
```

## N2. Introduction

**(a) Paragraph 3, last two sentences.**

Current:
> Text-anchored quantities that could assess changes in category alignment are defined in Section III-D but are not computed. Visual centroid displacement alone does not establish semantic confusion.

Replace with:
```latex
Visual centroid displacement alone does not establish semantic confusion, so we also compute text-anchored quantities and a zero-shot classifier on the same crops (Section III-D) to test whether displacement accompanies a loss of category recognition.
```

**(b) Contributions.** Contribution (iii) was reworded by change 4 of the corrections file. After it, add:
```latex
and (iv) a ground-truth crop-classification and text-alignment analysis on the same crops, showing that lower displacement need not mean better recognition.
```
Also adjust the conjunctions so the list reads (i), (ii), (iii) and (iv).

## N3. Section III-D — retitle and add the classifier

**Heading.** Change "D. Text-Anchored Extensions (Defined, Not Computed)" to:
```latex
\subsection{Text-Anchored Measures and Crop Classification}
```

**Definitions.** Keep Eqs. (5)–(6) and their definitions unchanged.

**Closing sentence.** Current:
> These centroid-level quantities are defined for completeness and have not been computed; they do not determine individual-crop classification accuracy.

Replace with:
```latex
These centroid-level quantities do not determine individual-crop accuracy, which we therefore measure directly. For each encoder, the prompt ``a photo of a \{$c$\}'' is encoded for every candidate category with that encoder's own text tower and unit-normalized, and each crop is assigned the candidate with the highest cosine similarity to its unit-norm image embedding. Both domains use the same candidates, prompt and rule, and nothing is trained or tuned. P1 uses \{small vehicle, large vehicle\}; P2 and P3 are pooled into \{car, bus, truck, building\}, so building is never a single-candidate task, and the two vocabularies are kept separate because their vehicle labels overlap. We report accuracy, per-category recall, and the recall gap (overhead minus UAV). The fine-tuned encoder was trained on VisDrone with the P1 labels, so its UAV results are in-domain rather than zero-shot.
```

## N4. Section III-E — software versions fix and uncertainty method

**(a) Software versions. This is a factual fix.** The version list in III-E (Pillow 11.3.0, OpenCLIP 3.3.0, NumPy 1.26.4, Matplotlib 3.9.4, torchvision 0.20.1) describes the environment rebuilt in October. It is not the one that computed the embeddings in August. Re-decoding the JPEG images today gives slightly different pixels, which proves at least the image library differed.

Current:
> The experiment environment used PyTorch 2.5.1 and torchvision 0.20.1 (both CUDA 11.8 builds), Transformers 4.49.0, OpenCLIP 3.3.0, NumPy 1.26.4, Pillow 11.3.0, Matplotlib 3.9.4, and scikit-learn 1.3.2.

Replace with:
```latex
Embeddings were computed with PyTorch 2.5.1 (CUDA 11.8 build) and Transformers 4.49.0 and cached; all reported analyses use these cached embeddings, and a crop manifest links each embedding to its source image and box.
```

**(b) New paragraph.** Add at the end of III-E, after the protocol-correction paragraph from the corrections file if it was added:
```latex
\emph{Uncertainty.} Crops from one image are correlated, so all resampling is by source image. A permutation test reassigns whole images between the two domains (2,000 permutations; statistic $D^{\mathrm{L2}}_c$). Intervals are 95\% percentile intervals from 2,000 image-bootstrap replicates, drawn independently per domain and shared across encoders so that encoder differences are paired. Resampling UAV crops by video sequence instead gives similar intervals; xView chips cannot be grouped by source scene.
```

## N5. Section IV — add significance and intervals to existing results

**IV-A.** After "Table II lists all values.", add:
```latex
For all 14 category--encoder cells the observed displacement exceeds every one of 2,000 image-level permutations ($p<0.001$; the closest case, fine-tuned large-vehicle, gives $p=0.0003$ with 20,000 permutations).
```

**IV-B.** After the sentence ending "…with larger relative reductions for small-vehicle (−13.4%) and building (−19.7%).", add:
```latex
Paired image-bootstrap intervals for these two ratio changes are $-18.5\%$ to $-6.2\%$ and $-23.2\%$ to $-15.9\%$; for the other four categories they include zero.
```

**IV-C.** Current:
> In small-vehicle the ratio rises to 0.8165 (CLIP 0.6498, RemoteCLIP 0.5627), so displacement fell proportionally less than spread did. In large-vehicle the ratio falls to 0.4132 (CLIP 0.6235, RemoteCLIP 0.6252), so displacement fell more than spread did.

Replace with:
```latex
In small-vehicle the ratio rises to 0.8165 (CLIP 0.6498, RemoteCLIP 0.5627), but the paired interval for the change ($-1.5\%$ to $+52.2\%$) includes zero. In large-vehicle it falls to 0.4132 (CLIP 0.6235, RemoteCLIP 0.6252; change $-45.4\%$ to $-17.6\%$), so displacement fell more than spread did.
```

**IV-D.** After "…is 1.50 to 2.32 times the pooled mean between-category cosine dissimilarity defined in (4).", add:
```latex
Image-bootstrap intervals for $Y_c$ lie above 1 in all eight cells (lowest bound 1.21).
```

Then current:
> For car, bus and truck, RemoteCLIP's ratio is higher than CLIP's, so relative to its own between-category spacing it shows no advantage there.

Replace with:
```latex
RemoteCLIP's ratio is higher than CLIP's for bus and truck (paired intervals exclude zero) but not clearly for car, so relative to its own between-category spacing it shows no advantage for vehicles.
```

## N6. Section IV-F (new) — crop classification and text alignment, with Table V

Add a new subsection at the end of Section IV, after IV-E, or replacing IV-E if Fig. 3 is cut (see "Fitting into six pages"):

```latex
\subsection{Crop Classification and Text Alignment}
Table~\ref{tab:cls} reports ground-truth crop classification for the pooled task. Both encoders recognize UAV crops better than overhead crops: accuracy is 60.2\% against 32.9\% for CLIP and 51.8\% against 27.3\% for RemoteCLIP (image-bootstrap intervals for the gaps: $-31.6$ to $-23.3$ and $-30.1$ to $-19.3$ points), and the two losses are not distinguishable from each other. The loss follows a domain-wide bias rather than per-category drift: overhead vehicles collapse onto one label, car for CLIP (813 of 900) and building for RemoteCLIP (839 of 900). The centroid margin $m_c$ is negative on the overhead side for three categories under each encoder, and $\Delta m_c<0$ in all eight cells (interval excludes zero in seven), so the correct-text margin is smaller for overhead crops. RemoteCLIP's lower raw displacement therefore does not translate into better overhead recognition. In P1 the two-label task is weak for CLIP, which labels 98\% of overhead crops small vehicle; no encoder shows an accuracy gap whose interval excludes zero, and the fine-tuned encoder, trained on this UAV task, reaches 84.8\% overhead and 84.2\% UAV accuracy. Across the four pooled categories, displacement ranks with $A_c$ and $|\Delta m_c|$ (Spearman $\rho=0.8$ for CLIP, $0.4$ for RemoteCLIP) but not with recall loss ($\rho=0.02$ pooled over both encoders); with four categories these rank correlations are descriptive only.
```

Table V. Place it near IV-F; a `table` float in one column. Use `booktabs` if the paper already does, otherwise match Table II's style.

```latex
\begin{table}[t]
\caption{Ground-truth crop classification, pooled P2+P3 task (candidates car, bus, truck, building; prompt ``a photo of a \{$c$\}''). Recall on overhead (O) and UAV (U) crops and centroid margin change $\Delta m_c$ from (6). Same crops and seed as Table~II.}
\label{tab:cls}
\centering
\begin{tabular}{lcccc}
\hline
 & \multicolumn{2}{c}{CLIP ViT-B/32} & \multicolumn{2}{c}{RemoteCLIP ViT-L/14} \\
Category & Recall O / U (\%) & $\Delta m_c$ & Recall O / U (\%) & $\Delta m_c$ \\
\hline
car      & 92.0 / 87.7 & $-0.014$ & \phantom{0}8.3 / 63.3 & $-0.030$ \\
bus      & \phantom{0}1.0 / 41.3 & $-0.021$ & \phantom{0}1.7 / 41.0 & $-0.029$ \\
truck    & 10.0 / 30.3 & $-0.010$ & \phantom{0}0.7 / 11.7 & $-0.002$ \\
building & 28.7 / 81.7 & $-0.035$ & 98.3 / 91.3 & $-0.017$ \\
\hline
Accuracy & 32.9 / 60.2 & & 27.3 / 51.8 & \\
\hline
\end{tabular}
\end{table}
```

## N7. Section V (Discussion and Limitations)

**(a) "Sampling" paragraph.** Current, from "Sampling. All values are point estimates from one seed and 300 crops per category and domain, with no confidence intervals." through "…should not be interpreted as evidence of superiority or statistical equivalence.":

Replace that whole paragraph with:
```latex
\emph{Sampling.} Values come from one sampling seed and 300 crops per category and domain. Intervals reflect image-level resampling of these crops, not new samples from the full datasets. The 300 DOTA crops per category come from only 119 (small-vehicle) and 68 (large-vehicle) of the 469 images, with up to 43 from a single image, so image-level resampling matters; xView chips cannot be grouped by scene, and VisDrone sequence IDs are inferred from file names. Differences of about 2\% between encoders lie inside these intervals and should not be read as superiority or equivalence.
```

**(b) Classification limitations.** Add to the "Coverage and metric" paragraph:
```latex
Crop classification uses one fixed prompt; RemoteCLIP's own evaluations use remote-sensing prompts, so its overhead ``building'' bias may be partly prompt-dependent. It also omits localization, background and unlisted classes, and on the UAV side the pooled task mixes VisDrone vehicles with UAVid buildings.
```

**(c) Discussion paragraph 2.** Current, the sentence beginning "Computing A_c and Δm_c, measuring category separation and covariance structure, and repeating the analysis with image-level resampling would help distinguish these possibilities…":

Replace with:
```latex
Measuring category separation and covariance structure, and repeating the analysis on more seeds and full datasets, would help distinguish these possibilities without guaranteeing a unique mechanism.
```

## N8. Table IV — updated rows

Replace the whole table body with the rows below. They are written as `Statement & Status \\` lines to match the existing two-column layout.

```latex
Displacement exceeds sampling variation & Supported: all 14 cells exceed 2,000 image-level permutations \\
RemoteCLIP has lower raw displacement than CLIP & Supported for all six categories \\
RemoteCLIP has lower normalized displacement & Lower for small-vehicle and building; intervals include zero for the other four \\
Displacement exceeds pooled between-category spacing & Supported for four categories and two encoders (intervals above 1); reference mixes datasets \\
Fine-tuning lowers displacement through better alignment & Unresolved: smaller radii; small-vehicle ratio change not distinguishable from zero \\
Displacement accompanies loss of crop recognition & Supported for the pooled task (overhead accuracy 25--27 points lower); not for P1 \\
Lower displacement means better recognition & Not supported: RemoteCLIP is no better than CLIP on overhead crops \\
Larger displacement predicts larger accuracy loss & Not supported across four categories ($\rho=0.02$); exploratory \\
Text-anchored $A_c$ and $\Delta m_c$ & Computed: overhead margin smaller in seven of eight pooled cells \\
Displacement is caused by altitude & Not supported: sensor, resolution, geography and dataset co-vary \\
Displacement predicts detection loss & Not tested: no detector was run \\
Stability across seeds and full datasets & Not tested; image-level resampling only \\
```

## N9. Section VI (Scope of Claims and Future Work)

Current:
> Future work will compute A_c and Δm_c, extend category and dataset coverage, and estimate uncertainty through repeated sampling and image-level resampling.

Replace with:
```latex
Future work will test several prompts and encoders, extend category and dataset coverage, and repeat sampling over seeds and full datasets.
```

Keep the rest of the paragraph (detection experiments, multi-altitude data).

## N10. Conclusion (replaces the paragraph)

```latex
This pilot measures category-conditioned visual centroid displacement between overhead and UAV crops using a shared frozen encoder. Displacement is significant under image-level permutation for every category and encoder and exceeds a pooled between-category reference for four categories. In a four-category ground-truth crop-classification task the same domain change accompanies a 25 to 27 point accuracy loss on overhead crops and smaller correct-text margins, yet RemoteCLIP's lower displacement does not improve overhead recognition, and displacement magnitude does not rank with accuracy loss across categories. Visual drift thus tracks movement relative to the text anchors but is not by itself a predictor of recognition failure. Controlled multi-altitude data, more prompts and encoders, and detection experiments are needed to isolate altitude and to test the link to detection.
```

---

## Fitting into six pages

The additions (N3, N4b, N5, N6 with Table V) are about half a page. N1, N7a, N9 and the hedge trims in the corrections file recover part of that. If the paper is still over six pages after compiling, cut in this order and recompile after each step:

1. **Remove Fig. 3 (joint t-SNE) and Section IV-E.** The paper itself says the figure is illustrative and not evidence. Put the IV-F text where IV-E was.
2. **Shorten Section II-B (closed-vocabulary domain adaptation)** to its first two sentences.
3. **Use the shorter option of change 9** in the corrections file (rename the III-E heading; no protocol-correction paragraph).

Do not shrink fonts, margins or the reference list.

## Final checks

1. **Every number in the new text matches this file exactly.** The table cells and percentages come straight from the results.
2. **"Detection" is never used** for the classification results, except in statements that detection was **not** tested.
3. **Section III-D no longer says "not computed",** and Table IV no longer says "Defined, not computed".
4. **Page count:** 6 or fewer, including references.
5. **Abstract:** 250 words or fewer.
