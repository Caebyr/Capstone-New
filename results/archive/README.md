# Archived results

Superseded outputs, kept as a record of how the results evolved. **Don't cite these.** The current versions are listed beside each file.

| File | What it is | Superseded by |
|---|---|---|
| `drift_comparison_3way_leaky.json`, `three_way_drift_comparison_leaky.png` | First three-way run (Aug 19). The fine-tuned CLIP was trained and evaluated on the same VisDrone images (train/test leakage), and bicycles were still counted as small vehicles. | `results/remoteclip_vs_clip/drift_comparison_3way_heldout.json`, `three_way_drift_comparison_heldout.png` |
| `xview_vs_visdrone_drift_3class.png` | xView vs VisDrone plot before building was added | `results/xview_vs_visdrone/four_class_drift_comparison.png` |
| `normalized_drift_reduction.png` | Mid-semester "% of CLIP" chart (written here by `scripts/plots/make_normalized_drift_plot.py`) | `final_premst_results/fig1_drift_vs_clip_baseline.*` |

The checkpoints and embeddings of the leaky run were deleted in the November 2026 cleanup; only these small result files remain.
