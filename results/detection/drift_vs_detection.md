# Does embedding drift explain detection loss? (B5)

Drop = UAV minus overhead (positive = worse on overhead), in percentage points. Crop drift (CLIP, RemoteCLIP) is the full-scale A3 cache; OWLv2-space drift uses OWLv2's own class embedding of the patch that best matches each GT box on the detection benchmark. Δm = text margin overhead − UAV (negative = the text query fits the overhead centroid worse).

## OWLv2 embedding space

| Category | n (O / U) | cos drift [CI] | ratio [CI] | ratio, IoU ≥ 0.5 only | best-patch IoU ≥ 0.5 (O / U) | margin O / U | Δm |
|---|---|---|---|---|---|---|---|
| small-vehicle | 3000 / 3000 | 0.0759 [0.0682, 0.0908] | 0.630 [0.595, 0.695] | 0.639 | 99% / 91% | +0.0136 / +0.0168 | -0.0032 |
| large-vehicle | 986 / 2160 | 0.1455 [0.1248, 0.1722] | 0.735 [0.675, 0.807] | 0.734 | 100% / 91% | +0.0046 / -0.0023 | +0.0069 |
| car | 168 / 3000 | 0.1026 [0.0926, 0.1226] | 0.693 [0.656, 0.758] | 0.696 | 88% / 91% | +0.0681 / +0.1125 | -0.0444 |
| bus | 127 / 306 | 0.1448 [0.1153, 0.1803] | 0.804 [0.655, 0.983] | 0.828 | 98% / 93% | +0.0098 / +0.0351 | -0.0253 |
| truck | 273 / 634 | 0.0994 [0.0888, 0.1166] | 0.590 [0.559, 0.647] | 0.585 | 93% / 89% | +0.0192 / +0.0545 | -0.0352 |
| building | 3000 / 389 | 0.1602 [0.1465, 0.1797] | 0.747 [0.705, 0.804] | 0.742 | 95% / 100% | +0.0576 / +0.1207 | -0.0631 |

## Per category: drift vs detection loss

| Category | CLIP ratio | RemoteCLIP ratio | OWLv2-space ratio | CLIP Δm | OWLv2-space Δm | OWLv2 AP50 drop | OWLv2 found drop | OWLv2 named drop | G-DINO AP50 drop | G-DINO found drop | G-DINO named drop |
|---|---|---|---|---|---|---|---|---|---|---|---|
| small-vehicle | 0.579 | 0.526 | 0.630 | -0.0005 | -0.0032 | +21.7 | +27.3 | +0.6 | +37.4 | +28.0 | +0.9 |
| large-vehicle | 0.536 | 0.548 | 0.735 | -0.0030 | +0.0069 | +6.5 | +30.6 | -26.8 | +21.6 | +47.7 | +0.5 |
| car | 1.028 | 0.987 | 0.693 | -0.0145 | -0.0444 | +68.2 | +28.2 | +6.1 | +69.7 | +29.1 | +1.5 |
| bus | 0.799 | 0.815 | 0.804 | -0.0203 | -0.0253 | +46.2 | +20.8 | +29.0 | +23.6 | +27.2 | +24.1 |
| truck | 0.809 | 0.771 | 0.590 | -0.0114 | -0.0352 | +30.7 | +18.4 | +17.4 | +26.8 | +30.7 | +43.6 |
| building | 1.115 | 0.907 | 0.747 | -0.0374 | -0.0631 | +14.5 | +57.2 | +10.4 | +24.2 | +53.3 | +18.4 |

## Spearman rank correlation across the 6 categories (exact p)

| Drift measure | owlv2 AP50 drop | owlv2 found drop | owlv2 named drop | gdino AP50 drop | gdino found drop | gdino named drop |
|---|---|---|---|---|---|---|
| clip cos | +0.26 (0.66) | +0.31 (0.56) | +0.43 (0.42) | +0.20 (0.71) | +0.49 (0.36) | +0.49 (0.36) |
| clip ratio | +0.37 (0.50) | +0.20 (0.71) | +0.49 (0.36) | +0.43 (0.42) | +0.31 (0.56) | +0.54 (0.30) |
| clip delta m | -0.26 (0.66) | -0.26 (0.66) | -0.66 (0.17) | +0.14 (0.80) | -0.20 (0.71) | -0.54 (0.30) |
| remoteclip cos | +0.31 (0.56) | +0.37 (0.50) | +0.49 (0.36) | +0.09 (0.92) | +0.31 (0.56) | +0.43 (0.42) |
| remoteclip ratio | +0.54 (0.30) | +0.26 (0.66) | +0.43 (0.42) | +0.26 (0.66) | +0.14 (0.80) | +0.37 (0.50) |
| remoteclip delta m | -0.77 (0.10) | +0.20 (0.71) | -0.77 (0.10) | -0.20 (0.71) | +0.43 (0.42) | -0.60 (0.24) |
| owlv2space cos | -0.43 (0.42) | +0.71 (0.14) | +0.03 (1.00) | -0.66 (0.17) | +0.60 (0.24) | -0.09 (0.92) |
| owlv2space ratio | -0.09 (0.92) | +0.43 (0.42) | +0.26 (0.66) | -0.60 (0.24) | -0.03 (1.00) | -0.03 (1.00) |
| owlv2space delta m | -0.37 (0.50) | -0.20 (0.71) | -0.49 (0.36) | -0.43 (0.42) | -0.31 (0.56) | -0.54 (0.30) |

With six categories the smallest attainable two-sided p is 2/720 ≈ 0.003, and |rho| ≥ 0.89 is needed for p < 0.05; treat these as descriptive.
