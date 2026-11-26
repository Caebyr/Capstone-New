# Same-backbone control: CLIP ViT-L/14 vs RemoteCLIP ViT-L/14 (C2)

RemoteCLIP is CLIP ViT-L/14 further trained on remote-sensing image-text pairs. Comparing it with OpenAI's CLIP ViT-L/14 on the same crops isolates the effect of that training; comparing CLIP-L/14 with the paper's CLIP-B/32 isolates model size. Full-scale cache (RemoteCLIP's crops), prompt "a photo of a {}" unless stated. 95% image-bootstrap CIs (500 reps; classification on half of them).

## P1

| Category | CLIP ViT-B/32 ratio [CI] | CLIP ViT-L/14 ratio [CI] | RemoteCLIP ViT-L/14 ratio [CI] | RemoteCLIP vs CLIP-L/14, ratio change % [CI] | CLIP-L/14 vs CLIP-B/32, ratio change % [CI] |
|---|---|---|---|---|---|
| small-vehicle | 0.579 [0.551, 0.611] | 0.637 [0.614, 0.663] | 0.526 [0.498, 0.563] | -17.4 [-20.4, -13.7] | +10.1 [6.9, 13.1] |
| large-vehicle | 0.536 [0.511, 0.565] | 0.657 [0.635, 0.682] | 0.548 [0.527, 0.577] | -16.5 [-18.9, -13.7] | +22.6 [18.3, 26.3] |

| Category | CLIP ViT-B/32 cos drift | CLIP ViT-L/14 cos drift | RemoteCLIP ViT-L/14 cos drift |
|---|---|---|---|
| small-vehicle | 0.0270 | 0.0406 | 0.0162 |
| large-vehicle | 0.0291 | 0.0503 | 0.0214 |

| Encoder | accuracy O / U | balanced accuracy O [CI] | balanced accuracy U [CI] | Δm small-vehicle [CI] | Δm large-vehicle [CI] |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | 50.8 / 54.8 | 50.8 [50.3, 51.3] | 54.8 [54.3, 55.4] | -0.0005 [-0.0020, 0.0009] | -0.0030 [-0.0040, -0.0020] |
| CLIP ViT-L/14 | 54.6 / 59.4 | 54.6 [53.3, 56.1] | 59.4 [58.6, 60.1] | -0.0001 [-0.0019, 0.0018] | -0.0011 [-0.0027, 0.0004] |
| RemoteCLIP ViT-L/14 | 63.1 / 68.0 | 63.1 [60.8, 65.1] | 68.0 [67.0, 69.1] | -0.0058 [-0.0065, -0.0050] | -0.0011 [-0.0028, 0.0009] |

Overhead balanced accuracy, RemoteCLIP − CLIP-L/14: +8.6 pp [6.1, 10.5]; CLIP-L/14 − CLIP-B/32: +3.8 pp [2.6, 5.1].

Most common label given to overhead vehicle crops: CLIP ViT-B/32 'small-vehicle'; CLIP ViT-L/14 'small-vehicle'; RemoteCLIP ViT-L/14 'small-vehicle'.

## P1a

| Category | CLIP ViT-B/32 ratio [CI] | CLIP ViT-L/14 ratio [CI] | RemoteCLIP ViT-L/14 ratio [CI] | RemoteCLIP vs CLIP-L/14, ratio change % [CI] | CLIP-L/14 vs CLIP-B/32, ratio change % [CI] |
|---|---|---|---|---|---|
| small-vehicle | 0.587 [0.563, 0.612] | 0.634 [0.614, 0.655] | 0.538 [0.513, 0.568] | -15.1 [-17.7, -12.1] | +8.0 [5.5, 10.7] |
| large-vehicle | 0.532 [0.509, 0.557] | 0.656 [0.635, 0.677] | 0.557 [0.535, 0.582] | -15.1 [-17.6, -12.5] | +23.2 [19.4, 26.7] |

| Category | CLIP ViT-B/32 cos drift | CLIP ViT-L/14 cos drift | RemoteCLIP ViT-L/14 cos drift |
|---|---|---|---|
| small-vehicle | 0.0281 | 0.0408 | 0.0171 |
| large-vehicle | 0.0289 | 0.0505 | 0.0222 |

| Encoder | accuracy O / U | balanced accuracy O [CI] | balanced accuracy U [CI] | Δm small-vehicle [CI] | Δm large-vehicle [CI] |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | 50.8 / 54.7 | 50.8 [50.3, 51.3] | 54.7 [54.3, 55.2] | -0.0010 [-0.0023, 0.0004] | -0.0026 [-0.0037, -0.0014] |
| CLIP ViT-L/14 | 54.6 / 59.7 | 54.6 [53.3, 56.1] | 59.7 [59.0, 60.4] | -0.0008 [-0.0024, 0.0011] | -0.0011 [-0.0024, 0.0004] |
| RemoteCLIP ViT-L/14 | 63.1 / 68.0 | 63.1 [60.9, 65.7] | 68.0 [67.2, 68.8] | -0.0061 [-0.0069, -0.0053] | -0.0007 [-0.0025, 0.0012] |

Overhead balanced accuracy, RemoteCLIP − CLIP-L/14: +8.6 pp [6.3, 10.7]; CLIP-L/14 − CLIP-B/32: +3.8 pp [2.6, 5.3].

Most common label given to overhead vehicle crops: CLIP ViT-B/32 'small-vehicle'; CLIP ViT-L/14 'small-vehicle'; RemoteCLIP ViT-L/14 'small-vehicle'.

## P2+P3

| Category | CLIP ViT-B/32 ratio [CI] | CLIP ViT-L/14 ratio [CI] | RemoteCLIP ViT-L/14 ratio [CI] | RemoteCLIP vs CLIP-L/14, ratio change % [CI] | CLIP-L/14 vs CLIP-B/32, ratio change % [CI] |
|---|---|---|---|---|---|
| car | 1.028 [1.007, 1.047] | 0.937 [0.920, 0.955] | 0.987 [0.965, 1.013] | +5.4 [3.9, 6.8] | -8.8 [-9.5, -7.9] |
| bus | 0.799 [0.779, 0.825] | 0.781 [0.764, 0.800] | 0.815 [0.793, 0.841] | +4.4 [2.4, 6.4] | -2.3 [-3.8, -0.8] |
| truck | 0.809 [0.791, 0.829] | 0.746 [0.731, 0.761] | 0.771 [0.753, 0.790] | +3.4 [2.0, 4.9] | -7.9 [-8.8, -6.8] |
| building | 1.115 [1.090, 1.148] | 1.111 [1.087, 1.140] | 0.907 [0.883, 0.933] | -18.4 [-19.2, -17.6] | -0.3 [-1.3, 0.6] |

| Category | CLIP ViT-B/32 cos drift | CLIP ViT-L/14 cos drift | RemoteCLIP ViT-L/14 cos drift | CLIP ViT-B/32 Y | CLIP ViT-L/14 Y | RemoteCLIP ViT-L/14 Y |
|---|---|---|---|---|---|---|
| car | 0.0719 | 0.0808 | 0.0486 | 1.99 [1.89, 2.09] | 1.81 [1.74, 1.88] | 1.94 [1.85, 2.02] |
| bus | 0.0528 | 0.0605 | 0.0402 | 1.75 [1.63, 1.91] | 1.66 [1.57, 1.77] | 2.19 [2.07, 2.34] |
| truck | 0.0539 | 0.0546 | 0.0343 | 2.00 [1.86, 2.15] | 1.67 [1.57, 1.77] | 2.20 [2.06, 2.33] |
| building | 0.1471 | 0.1551 | 0.0691 | 2.10 [2.06, 2.13] | 1.97 [1.93, 2.00] | 1.86 [1.80, 1.91] |

| Encoder | accuracy O / U | balanced accuracy O [CI] | balanced accuracy U [CI] | Δm car [CI] | Δm bus [CI] | Δm truck [CI] | Δm building [CI] |
|---|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | 35.2 / 57.0 | 32.6 [32.0, 33.1] | 61.8 [61.0, 62.8] | -0.0145 [-0.0152, -0.0138] | -0.0203 [-0.0216, -0.0191] | -0.0114 [-0.0122, -0.0106] | -0.0374 [-0.0392, -0.0361] |
| CLIP ViT-L/14 | 36.2 / 69.5 | 34.5 [34.0, 35.1] | 73.9 [73.1, 74.7] | -0.0340 [-0.0350, -0.0331] | -0.0308 [-0.0322, -0.0295] | -0.0136 [-0.0147, -0.0124] | -0.0518 [-0.0538, -0.0503] |
| RemoteCLIP ViT-L/14 | 31.1 / 44.6 | 28.9 [28.6, 29.3] | 53.3 [52.4, 54.2] | -0.0284 [-0.0293, -0.0274] | -0.0306 [-0.0321, -0.0291] | -0.0062 [-0.0073, -0.0050] | -0.0079 [-0.0098, -0.0063] |

Overhead balanced accuracy, RemoteCLIP − CLIP-L/14: -5.6 pp [-6.2, -5.0]; CLIP-L/14 − CLIP-B/32: +1.9 pp [1.3, 2.6].

Most common label given to overhead vehicle crops: CLIP ViT-B/32 'car'; CLIP ViT-L/14 'truck'; RemoteCLIP ViT-L/14 'building'.

## P2van

| Category | CLIP ViT-B/32 ratio [CI] | CLIP ViT-L/14 ratio [CI] | RemoteCLIP ViT-L/14 ratio [CI] | RemoteCLIP vs CLIP-L/14, ratio change % [CI] | CLIP-L/14 vs CLIP-B/32, ratio change % [CI] |
|---|---|---|---|---|---|
| van | 1.082 [1.051, 1.119] | 0.984 [0.961, 1.014] | 1.048 [1.015, 1.090] | +6.5 [3.2, 9.8] | -9.1 [-11.0, -6.5] |

| Category | CLIP ViT-B/32 cos drift | CLIP ViT-L/14 cos drift | RemoteCLIP ViT-L/14 cos drift |
|---|---|---|---|
| van | 0.0772 | 0.0824 | 0.0530 |

## Prompt templates (full cache; balanced accuracy overhead / UAV, %)

**P1**

| Template | CLIP ViT-B/32 | CLIP ViT-L/14 | RemoteCLIP ViT-L/14 |
|---|---|---|---|
| `a photo of a {}` | 50.8 / 54.8 | 54.6 / 59.4 | 63.1 / 68.0 |
| `a satellite photo of a {}` | 50.8 / 53.5 | 52.6 / 55.4 | 60.2 / 62.8 |
| `an aerial photo of a {}` | 50.1 / 52.4 | 52.9 / 56.1 | 58.4 / 61.2 |
| `a drone photo of a {}` | 50.6 / 55.9 | 55.8 / 62.4 | 61.5 / 64.2 |
| `{}` | 54.5 / 56.8 | 65.7 / 61.7 | 62.5 / 63.6 |

**P2+P3**

| Template | CLIP ViT-B/32 | CLIP ViT-L/14 | RemoteCLIP ViT-L/14 |
|---|---|---|---|
| `a photo of a {}` | 32.6 / 61.8 | 34.5 / 73.9 | 28.9 / 53.3 |
| `a satellite photo of a {}` | 34.4 / 59.9 | 37.8 / 73.6 | 42.0 / 56.7 |
| `an aerial photo of a {}` | 29.3 / 60.4 | 29.8 / 70.3 | 37.8 / 52.4 |
| `a drone photo of a {}` | 29.4 / 58.9 | 36.5 / 72.5 | 30.9 / 53.8 |
| `{}` | 38.7 / 62.8 | 33.1 / 71.2 | 40.9 / 56.9 |

## Same-domain baseline levels (A4), cosine drift [95% CI]

| Level | Comparison | Category | CLIP ViT-B/32 | CLIP ViT-L/14 | RemoteCLIP ViT-L/14 |
|---|---|---|---|---|---|
| L0 | DOTA part1 vs part2/3 | small-vehicle | 0.0007 [0.0006, 0.0023] | 0.0009 [0.0008, 0.0028] | 0.0007 [0.0006, 0.0023] |
| L0 | DOTA part1 vs part2/3 | large-vehicle | 0.0037 [0.0028, 0.0083] | 0.0048 [0.0036, 0.0103] | 0.0044 [0.0028, 0.0098] |
| L0 | VisDrone train vs held-out images | car | 0.0001 [0.0002, 0.0006] | 0.0002 [0.0002, 0.0007] | 0.0001 [0.0001, 0.0004] |
| L0 | xView first vs second half of chips | car | 0.0006 [0.0004, 0.0011] | 0.0004 [0.0003, 0.0009] | 0.0004 [0.0002, 0.0008] |
| L0 | xView first vs second half of chips | building | 0.0003 [0.0002, 0.0005] | 0.0002 [0.0002, 0.0004] | 0.0002 [0.0002, 0.0004] |
| L1 | DOTA vs xView (overhead vs overhead) | small-vehicle / car | 0.0254 [0.0230, 0.0283] | 0.0380 [0.0346, 0.0413] | 0.0233 [0.0203, 0.0271] |
| L1 | DOTA vs xView (overhead vs overhead) | large-vehicle / bus+truck+van | 0.0306 [0.0276, 0.0347] | 0.0313 [0.0277, 0.0362] | 0.0282 [0.0245, 0.0325] |
| L1 | VisDrone vs UAVid (UAV vs UAV) | car | 0.0265 [0.0234, 0.0310] | 0.0443 [0.0408, 0.0489] | 0.0166 [0.0136, 0.0205] |
| L2 | DOTA vs VisDrone (all images) | small-vehicle | 0.0281 [0.0260, 0.0306] | 0.0408 [0.0386, 0.0435] | 0.0171 [0.0156, 0.0193] |
| L2 | DOTA vs VisDrone (all images) | large-vehicle | 0.0289 [0.0268, 0.0315] | 0.0505 [0.0479, 0.0540] | 0.0222 [0.0204, 0.0242] |
| L2 | xView vs VisDrone | car | 0.0719 [0.0690, 0.0748] | 0.0808 [0.0781, 0.0839] | 0.0486 [0.0464, 0.0509] |
| L2 | xView vs UAVid | car | 0.1083 [0.0996, 0.1210] | 0.1244 [0.1153, 0.1348] | 0.0751 [0.0664, 0.0839] |
| L2 | xView vs UAVid | building | 0.1471 [0.1393, 0.1572] | 0.1551 [0.1481, 0.1620] | 0.0691 [0.0650, 0.0736] |
