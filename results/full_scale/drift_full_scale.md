# Full-scale and multi-seed check of the paper's drift and classification results

Cache: up to 5,000 crops per category and domain drawn at random from the full datasets (DOTA 1,411 images, VisDrone 6,471, xView 20,881 chips, UAVid train+val). FULL = all cached crops, 95% image-bootstrap CI (1000 reps), image-level permutation p (1000 perms). SEEDS = the paper's protocol (300 per category and domain) for 20 independent seeds: mean [min–max].
JPEG sources were re-decoded with today's Pillow (libjpeg-turbo), so numbers can differ slightly from a pure re-sampling of the August embeddings.

## P1

Crops (overhead / UAV): small-vehicle 5000 / 5000 (399 / 1054 images), large-vehicle 5000 / 5000 (308 / 1025 images)

| Category | Model | Paper L2 / ratio | FULL L2 [CI] | FULL ratio [CI] | perm p | SEEDS L2 | SEEDS ratio |
|---|---|---|---|---|---|---|---|
| small-vehicle | CLIP ViT-B/32 | 0.2460 / 0.6498 | 0.2183 [0.209, 0.231] | 0.5788 [0.554, 0.612] | 0.0010 | 0.224 [0.202–0.251] | 0.595 [0.533–0.668] |
| large-vehicle | CLIP ViT-B/32 | 0.2596 / 0.6235 | 0.2209 [0.211, 0.233] | 0.5359 [0.511, 0.565] | 0.0010 | 0.225 [0.207–0.246] | 0.547 [0.503–0.590] |
| small-vehicle | RemoteCLIP ViT-L/14 | 0.1892 / 0.5627 | 0.1714 [0.162, 0.184] | 0.5263 [0.498, 0.565] | 0.0010 | 0.179 [0.162–0.196] | 0.553 [0.496–0.612] |
| large-vehicle | RemoteCLIP ViT-L/14 | 0.2276 / 0.6252 | 0.1937 [0.186, 0.203] | 0.5484 [0.527, 0.577] | 0.0010 | 0.197 [0.183–0.210] | 0.560 [0.521–0.597] |
| small-vehicle | CLIP fine-tuned on VisDrone | 0.1585 / 0.8165 | 0.1213 [0.108, 0.135] | 0.6578 [0.599, 0.722] | 0.0010 | 0.123 [0.103–0.152] | 0.671 [0.591–0.805] |
| large-vehicle | CLIP fine-tuned on VisDrone | 0.0955 / 0.4132 | 0.0778 [0.072, 0.087] | 0.3359 [0.307, 0.382] | 0.0010 | 0.080 [0.073–0.093] | 0.345 [0.317–0.405] |

Paired ratio change vs CLIP, % [95% CI]: RemoteCLIP ViT-L/14 small-vehicle -9.1 [-11.9, -5.9]; RemoteCLIP ViT-L/14 large-vehicle +2.3 [-0.6, +5.4]; CLIP fine-tuned on VisDrone small-vehicle +13.7 [+3.5, +22.6]; CLIP fine-tuned on VisDrone large-vehicle -37.3 [-42.6, -30.1]

| Model | Paper acc O / U (%) | FULL acc O / U (%) | FULL gap [CI] (pp) | SEEDS acc O | SEEDS acc U | FULL Δm_c per category |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | — | 50.8 / 54.8 | -4.0 [-9.0, +0.8] | 50.5 [49.3–51.8] | 54.6 [52.5–55.5] | small-vehicle -0.0005 [-0.0018, +0.0010], large-vehicle -0.0030 [-0.0041, -0.0018] |
| RemoteCLIP ViT-L/14 | — | 63.1 / 68.0 | -4.9 [-8.0, -1.6] | 63.2 [58.8–67.2] | 68.0 [65.7–70.8] | small-vehicle -0.0058 [-0.0066, -0.0051], large-vehicle -0.0011 [-0.0030, +0.0009] |
| CLIP fine-tuned on VisDrone | — | 83.6 / 83.0 | +0.6 [-1.4, +2.9] | 83.7 [81.2–85.3] | 83.6 [81.8–86.2] | small-vehicle -0.0163 [-0.0199, -0.0131], large-vehicle +0.0092 [+0.0057, +0.0129] |

| Paper claim | Holds at FULL scale | Holds in seeds |
|---|---|---|
| RemoteCLIP raw L2 < CLIP (both P1 cats) | yes | 20/20 |
| RemoteCLIP ratio lower than CLIP for small-vehicle | yes | 20/20 |
| fine-tuned raw L2 < CLIP (both) | yes | 20/20 |
| fine-tuned ratio > CLIP for small-vehicle | yes | 19/20 |
| fine-tuned ratio < CLIP for large-vehicle | yes | 20/20 |

## P1a

Crops (overhead / UAV): small-vehicle 5000 / 5000 (399 / 2888 images), large-vehicle 5000 / 5000 (308 / 2760 images)

| Category | Model | Paper L2 / ratio | FULL L2 [CI] | FULL ratio [CI] | perm p | SEEDS L2 | SEEDS ratio |
|---|---|---|---|---|---|---|---|
| small-vehicle | CLIP ViT-B/32 | — | 0.2226 [0.214, 0.233] | 0.5866 [0.565, 0.616] | 0.0010 | 0.222 [0.196–0.245] | 0.584 [0.518–0.640] |
| large-vehicle | CLIP ViT-B/32 | — | 0.2200 [0.213, 0.230] | 0.5322 [0.513, 0.558] | 0.0010 | 0.226 [0.208–0.242] | 0.546 [0.505–0.577] |
| small-vehicle | RemoteCLIP ViT-L/14 | — | 0.1760 [0.169, 0.187] | 0.5382 [0.515, 0.573] | 0.0010 | 0.174 [0.148–0.197] | 0.530 [0.451–0.600] |
| large-vehicle | RemoteCLIP ViT-L/14 | — | 0.1970 [0.190, 0.206] | 0.5569 [0.538, 0.583] | 0.0010 | 0.201 [0.191–0.220] | 0.568 [0.537–0.614] |

Paired ratio change vs CLIP, % [95% CI]: RemoteCLIP ViT-L/14 small-vehicle -8.3 [-10.9, -5.2]; RemoteCLIP ViT-L/14 large-vehicle +4.6 [+2.1, +7.6]

| Model | Paper acc O / U (%) | FULL acc O / U (%) | FULL gap [CI] (pp) | SEEDS acc O | SEEDS acc U | FULL Δm_c per category |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | — | 50.8 / 54.7 | -3.9 [-9.4, +2.1] | 50.6 [49.8–51.8] | 55.1 [53.2–56.2] | small-vehicle -0.0010 [-0.0025, +0.0003], large-vehicle -0.0026 [-0.0038, -0.0015] |
| RemoteCLIP ViT-L/14 | — | 63.1 / 68.0 | -4.8 [-8.1, -1.4] | 63.2 [57.8–64.8] | 68.0 [64.8–71.7] | small-vehicle -0.0061 [-0.0068, -0.0053], large-vehicle -0.0007 [-0.0024, +0.0012] |

| Paper claim | Holds at FULL scale | Holds in seeds |
|---|---|---|
| RemoteCLIP raw L2 < CLIP (both P1 cats) | yes | 20/20 |
| RemoteCLIP ratio lower than CLIP for small-vehicle | yes | 20/20 |

## P2+P3

Crops (overhead / UAV): car 5000 / 5000 (2051 / 2895 images), bus 3455 / 5000 (1629 / 1849 images), truck 5000 / 5000 (2696 / 2365 images), building 5000 / 1199 (3466 / 269 images)

| Category | Model | Paper L2 / ratio | FULL L2 [CI] | FULL ratio [CI] | FULL Y_c [CI] | perm p | SEEDS L2 | SEEDS ratio | SEEDS Y_c |
|---|---|---|---|---|---|---|---|---|---|
| car | CLIP ViT-B/32 | 0.3564 / 0.9931 | 0.3581 [0.351, 0.365] | 1.0277 [1.008, 1.046] | 1.99 [1.88, 2.08] | 0.0010 | 0.360 [0.337–0.385] | 1.035 [0.984–1.099] | 2.00 [1.70–2.25] |
| bus | CLIP ViT-B/32 | 0.3256 / 0.8443 | 0.3057 [0.296, 0.317] | 0.7991 [0.777, 0.827] | 1.75 [1.61, 1.91] | 0.0010 | 0.307 [0.295–0.319] | 0.803 [0.770–0.828] | 1.75 [1.62–1.89] |
| truck | CLIP ViT-B/32 | 0.2837 / 0.7550 | 0.3082 [0.300, 0.316] | 0.8092 [0.791, 0.829] | 2.00 [1.85, 2.14] | 0.0010 | 0.312 [0.288–0.336] | 0.823 [0.765–0.884] | 2.02 [1.68–2.42] |
| building | CLIP ViT-B/32 | 0.5417 / 1.2219 | 0.4902 [0.478, 0.505] | 1.1146 [1.089, 1.147] | 2.10 [2.05, 2.13] | 0.0010 | 0.491 [0.465–0.508] | 1.121 [1.047–1.169] | 2.10 [1.95–2.24] |
| car | RemoteCLIP ViT-L/14 | 0.3024 / 0.9719 | 0.2982 [0.291, 0.306] | 0.9875 [0.963, 1.012] | 1.94 [1.85, 2.02] | 0.0010 | 0.301 [0.271–0.323] | 0.997 [0.904–1.065] | 1.95 [1.59–2.21] |
| bus | RemoteCLIP ViT-L/14 | 0.2802 / 0.8491 | 0.2692 [0.260, 0.279] | 0.8148 [0.792, 0.843] | 2.19 [2.04, 2.34] | 0.0010 | 0.270 [0.253–0.288] | 0.818 [0.761–0.858] | 2.17 [1.99–2.46] |
| truck | RemoteCLIP ViT-L/14 | 0.2366 / 0.7500 | 0.2495 [0.243, 0.256] | 0.7714 [0.755, 0.790] | 2.20 [2.05, 2.34] | 0.0010 | 0.253 [0.231–0.269] | 0.783 [0.727–0.841] | 2.21 [1.84–2.60] |
| building | RemoteCLIP ViT-L/14 | 0.3783 / 0.9809 | 0.3449 [0.336, 0.356] | 0.9066 [0.885, 0.934] | 1.86 [1.80, 1.91] | 0.0010 | 0.345 [0.325–0.359] | 0.911 [0.849–0.953] | 1.85 [1.63–1.97] |

Paired ratio change vs CLIP, % [95% CI]: RemoteCLIP ViT-L/14 car -3.9 [-5.4, -2.4]; RemoteCLIP ViT-L/14 bus +2.0 [-0.1, +3.9]; RemoteCLIP ViT-L/14 truck -4.7 [-6.0, -3.3]; RemoteCLIP ViT-L/14 building -18.7 [-19.5, -17.7]

| Model | Paper acc O / U (%) | FULL acc O / U (%) | FULL gap [CI] (pp) | SEEDS acc O | SEEDS acc U | FULL Δm_c per category |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | 32.9 / 60.2 | 35.2 / 57.0 | -21.8 [-24.1, -19.4] | 32.4 [30.4–34.0] | 62.2 [60.1–64.8] | car -0.0145 [-0.0151, -0.0138], bus -0.0203 [-0.0216, -0.0191], truck -0.0114 [-0.0122, -0.0106], building -0.0374 [-0.0390, -0.0360] |
| RemoteCLIP ViT-L/14 | 27.3 / 51.8 | 31.1 / 44.6 | -13.5 [-15.4, -11.7] | 28.6 [27.9–29.2] | 53.4 [51.5–54.9] | car -0.0284 [-0.0293, -0.0275], bus -0.0306 [-0.0324, -0.0287], truck -0.0062 [-0.0074, -0.0052], building -0.0079 [-0.0095, -0.0060] |

| Paper claim | Holds at FULL scale | Holds in seeds |
|---|---|---|
| RemoteCLIP raw L2 < CLIP (all 4 P2+P3 cats) | yes | 20/20 |
| RemoteCLIP ratio lower than CLIP for building | yes | 20/20 |
| Y_c > 1 in all 8 cells | yes | 20/20 |
| overhead acc < UAV acc (CLIP and RemoteCLIP) | yes | 20/20 |
| RemoteCLIP overhead acc <= CLIP overhead acc | yes | 20/20 |
| delta_m < 0 in all 8 cells | yes | 20/20 |
| CLIP labels most overhead vehicles 'car' | yes | 20/20 |
| RemoteCLIP labels most overhead vehicles 'building' | yes | 20/20 |
| pooled Spearman(cos drift, |recall gap|) < 0.5 | yes | 20/20 |

## P2van

Crops (overhead / UAV): van 146 / 5000 (128 / 2698 images)

| Category | Model | Paper L2 / ratio | FULL L2 [CI] | FULL ratio [CI] | perm p | SEEDS L2 | SEEDS ratio |
|---|---|---|---|---|---|---|---|
| van | CLIP ViT-B/32 | — | 0.3723 [0.363, 0.383] | 1.0821 [1.049, 1.117] | 0.0010 | 0.375 [0.360–0.396] | 1.090 [1.057–1.138] |
| van | RemoteCLIP ViT-L/14 | — | 0.3123 [0.304, 0.322] | 1.0478 [1.015, 1.087] | 0.0010 | 0.314 [0.297–0.334] | 1.054 [1.000–1.121] |

Paired ratio change vs CLIP, % [95% CI]: RemoteCLIP ViT-L/14 van -3.2 [-6.0, -0.2]


## Note on FULL accuracy

FULL "acc" above pools all cached crops, so classes count in proportion to their crop numbers (e.g. 3,455 xView buses, 1,199 UAVid buildings). The paper's design is balanced (300 per class); the SEEDS columns follow it. Balanced accuracy (mean per-class recall) at FULL scale:

| Task | Model | Balanced acc overhead / UAV (%) | Gap (pp) | Paper |
|---|---|---|---|---|
| P1 | CLIP ViT-B/32 | 50.8 / 54.8 | -4.0 | — |
| P1 | RemoteCLIP ViT-L/14 | 63.1 / 68.0 | -4.9 | — |
| P1 | CLIP fine-tuned on VisDrone | 83.6 / 83.0 | +0.6 | 84.8 / 84.2 |
| P2+P3 | CLIP ViT-B/32 | 32.6 / 61.8 | -29.2 | 32.9 / 60.2 |
| P2+P3 | RemoteCLIP ViT-L/14 | 28.9 / 53.3 | -24.4 | 27.3 / 51.8 |

Overhead vehicle predictions at FULL scale: CLIP labels 11,956 of 13,455 overhead vehicles "car"; RemoteCLIP labels 12,244 of 13,455 "building" (paper: 813 / 839 of 900).
