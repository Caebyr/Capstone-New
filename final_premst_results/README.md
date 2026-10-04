# final_premst_results — figures for the initial-findings paper

Four paper-ready figures, each as **PNG (300 dpi)** and **PDF (vector, fonts embedded)**, plus the table of absolute values that Fig. 1 needs beside it. All are sized for full text width of a two-column page (about 7 in) — place them at `\textwidth` so the 6–8 pt text stays legible.

| File | What it shows | Where it comes from |
|---|---|---|
| `fig1_drift_vs_clip_baseline` | Drift of RemoteCLIP (and, for DOTA↔VisDrone, the VisDrone-fine-tuned CLIP) as % of the CLIP ViT-B/32 baseline, L2 and cosine on one axis | `results/remoteclip_vs_clip/drift_comparison_3way_heldout.json`, `results/xview_vs_visdrone/drift_comparison_4class.json` |
| `fig2_tsne_building_domain_split` | t-SNE of all 2,400 ground-truth crops per encoder; `building` separates into a satellite cluster and a UAV cluster | cached embeddings in `embeddings/xview_vs_visdrone/` |
| `fig3_cross_altitude_vs_cross_category` | Cross-altitude drift vs. the distance between two different categories (1.5–2.3×) | `results/category_yardstick/yardstick_table.json` |
| `fig4_pipeline` | Measurement pipeline: datasets → crops → encoders → centroids → drift measures → analysis | drawn from the code path (`src/dataset.py`, `src/embeddings.py`, `run_*.py`) |
| `results_table.md` / `.csv` | Absolute L2, cosine, drift ratio, and % of CLIP for every class and encoder | same JSONs as Fig. 1 |
| `make_final_figures.py` | Regenerates everything above | run with the `hf` conda env |

## Suggested captions

**Fig. 1.** Semantic drift of specialised encoders relative to CLIP ViT-B/32 (= 100%). (a) DOTA vs. VisDrone; (b) xView vs. VisDrone (car, bus, truck) and xView vs. UAVid (building). Solid bars: L2 distance between per-domain class centroids; hatched: cosine distance. Lower means less drift. 300 crops per class per domain, single seed.

**Fig. 2.** t-SNE of ground-truth object crops embedded by (a) CLIP ViT-B/32 and (b) RemoteCLIP ViT-L/14, fitted jointly on both domains. Circles: satellite (xView); crosses: UAV (VisDrone, and UAVid for buildings). Buildings form separate satellite and UAV clusters in both encoders. *(Qualitative only; t-SNE distances are not a metric.)*

**Fig. 3.** Cross-altitude drift (cosine distance between the satellite and UAV centroids of the same class) compared with cross-category distance (mean cosine distance from a class centroid to the other classes' centroids at the same altitude). Cross-altitude drift is 1.5–2.3× larger for every class and both encoders.

**Fig. 4.** Overview of the drift-measurement pipeline. †The fine-tuned CLIP is trained on 80% of VisDrone images and evaluated on DOTA↔VisDrone only.

## Numbers behind the text (from `results_table.md`)

- RemoteCLIP reduces L2 drift vs. CLIP by 12.3% (large-vehicle) to 30.2% (building); cosine drift by 26.1% to 53.5%. It is lower on all six classes.
- Fine-tuned CLIP (held-out VisDrone): L2 −35.6% (small-vehicle) / −63.2% (large-vehicle); cosine −62.7% / −88.2%.
- `building` has the largest drift of any class (CLIP L2 0.5417, cosine 0.1822) and the largest reduction under RemoteCLIP.

## What the paper must say alongside these figures

These are properties of the data, not of the plots; a reviewer can find them.

1. **Single seed, 300 crops per class per domain.** No confidence intervals exist. A plain bootstrap over crops would *understate* the uncertainty, because the crops are not independent: the 300 DOTA crops per class come from only 68 (large-vehicle) and 119 (small-vehicle) of the 469 images (up to 43 crops from one image), whereas the VisDrone samples span 241–289 images (figures obtained by replaying the seed-42 sampling; see `TECHNICAL_REPORT.md` §10.1).
2. **Raw drift reductions partly reflect cluster tightening.** RemoteCLIP and especially the fine-tuned model have smaller intra-domain spread than CLIP on both domains, so L2/cosine shrink partly because clusters shrink. The normalised `drift_ratio` (L2 ÷ mean intra-domain radius) does not tell the same story for every case: fine-tuned small-vehicle 0.8165 vs. CLIP 0.6498 and RemoteCLIP 0.5627; RemoteCLIP large-vehicle 0.6252 vs. CLIP 0.6235 (`results_table.md`, column *Drift ratio*). Do not claim "alignment" from Fig. 1 alone.
3. **The fine-tuned model is evaluated on DOTA↔VisDrone only**, on the held-out 20% of VisDrone images (split seed 1234); it was never run on xView or UAVid.
4. **`building` compares xView with UAVid**, not VisDrone (VisDrone has no building class), so that class differs from the vehicle classes in both the class *and* the UAV dataset.
5. **"Satellite vs. UAV" is also "dataset vs. dataset"** (different sensors, resolutions, annotation conventions). Altitude is not isolated.
6. **Scope:** DOTA = 469 images (`part1.zip`; 1,411 label files exist, `part2`/`part3` are not extracted); xView = first 5,000 streamed `train` images (buildings: first 500); UAVid = train + val. Only ground-truth boxes/masks are used — no detector.

## What was changed relative to the earlier figures in `results/`

| Earlier figure | Problem | Fix here |
|---|---|---|
| `normalized_drift_reduction.png` → Fig. 1 | Legend covered bars; no absolute values; no caveats on the figure | Legend moved below the axes, values printed on bars, footnote with n / seed / held-out split / building note, `results_table.*` added |
| `pca_tsne/tsne_xview_visdrone_uavid.png` → Fig. 2 | Randomly kept 900 of 1,200 points per domain, and a *different* subset in each encoder's panel | All 1,200 points per domain in both panels (same procedure, seed 42). **The t-SNE layout therefore differs from the earlier image**; the building split is still visible |
| `yardstick_xview_visdrone_uavid.png` → Fig. 3 | Colours meant something different from the other figures (its blue clashed with CLIP's blue) | Encoder colour on the cross-altitude bars, neutral hatched grey for cross-category; ratio labels enlarged; footnote defines cross-category |
| `diagrams/fig4_1_architecture_preview.png` → Fig. 4 | Thin strip of tiny text on a blank canvas; DOTA/VisDrone only; no formulas | Redrawn as a compact 6-step figure with xView/UAVid, the three encoders, and the drift formulas from `diagrams/fig4_3_drift_computation.drawio`. The `.drawio` sources are unchanged |

Fig. 4 drops the "Heatmap" box of the old diagram: the only heat-map in the repo comes from the whole-image Phase-1 experiment with zero-shot-guessed labels, which is not part of these results.

## Reproduce

```
C:\anaconda\envs\hf\python.exe final_premst_results\make_final_figures.py
```

Needs `results/*.json` (tracked) and, for Fig. 2 only, `embeddings/xview_vs_visdrone/*.npy` (git-ignored — copy them along if you work outside this machine). Runs in about 20 s on CPU. Text is set in Arial when installed, else DejaVu Sans (wider; labels may need nudging).
