# Object-level test: does an object's text margin predict detection? (C1)

Unit = one ground-truth object; 28,000 objects (the B5 sample: 400 seeded images per set). Margin = cos(own class text) − max cos(other class texts) for the object's crop, prompt "a photo of a {}", z-scored. CLIP and RemoteCLIP encode the crop independently of both detectors; OWLv2-internal uses OWLv2's own patch embedding, so for OWLv2's outcomes it is close to circular (shown as a check only). Size = log2 √area and log2 aspect ratio, z-scored. All models include category fixed effects. 95% CIs: image-cluster bootstrap, 500 reps. Found / named: score ≥ 0.1, IoU ≥ 0.5.

## Objects

| Category | overhead | UAV |
|---|---|---|
| small-vehicle | 3116 | 6851 |
| large-vehicle | 986 | 2160 |
| car | 168 | 6851 |
| bus | 127 | 306 |
| truck | 273 | 634 |
| building | 6139 | 389 |

## Raw rates

| Detector | outcome | overhead | UAV |
|---|---|---|---|
| OWLv2 B/16 | found | 41.2% | 87.4% |
| OWLv2 B/16 | named_given_found | 89.3% | 90.9% |
| OWLv2 B/16 | success | 36.8% | 79.4% |
| Grounding DINO tiny | found | 38.3% | 82.4% |
| Grounding DINO tiny | named_given_found | 77.5% | 86.3% |
| Grounding DINO tiny | success | 29.7% | 71.1% |

## 1. Does margin predict detection, holding size, category and domain fixed? (model M2)

Odds ratio per +1 SD of margin. Above 1 = objects whose crop sits nearer their own class text are more often detected.

| Detector | outcome | CLIP | RemoteCLIP | OWLv2-internal (check) |
|---|---|---|---|---|
| OWLv2 B/16 | found | 1.41 [1.26, 1.58] | 1.93 [1.80, 2.10] | 1.35 [1.22, 1.54] |
| OWLv2 B/16 | named_given_found | 2.13 [1.92, 2.43] | 1.77 [1.61, 1.95] | 1218697.10 [247294.42, 9099445.79] |
| OWLv2 B/16 | success | 1.71 [1.55, 1.92] | 1.99 [1.87, 2.13] | 2.77 [2.41, 3.35] |
| Grounding DINO tiny | found | 1.52 [1.40, 1.67] | 1.55 [1.42, 1.71] | 1.13 [0.99, 1.28] |
| Grounding DINO tiny | named_given_found | 1.83 [1.67, 2.03] | 1.67 [1.53, 1.80] | 3.49 [2.90, 4.37] |
| Grounding DINO tiny | success | 1.87 [1.72, 2.01] | 1.74 [1.61, 1.86] | 1.77 [1.55, 2.07] |

OWLv2-internal for OWLv2's own naming is circular by construction (OWLv2 picks the class whose text is nearest that same embedding), hence the huge odds ratio; it only confirms the replay is aligned.

## 2. Within one domain (success ~ category + size + margin)

Rules out margin merely standing in for 'overhead or not'.

| Detector | domain | CLIP | RemoteCLIP | OWLv2-internal (check) |
|---|---|---|---|---|
| OWLv2 B/16 | overhead | 1.12 [0.91, 1.37] | 1.64 [1.43, 1.88] | 1.69 [1.44, 2.08] |
| OWLv2 B/16 | uav | 2.33 [2.14, 2.61] | 2.13 [2.01, 2.28] | 5.22 [4.34, 6.63] |
| Grounding DINO tiny | overhead | 1.38 [1.15, 1.63] | 1.32 [1.16, 1.49] | 1.19 [1.00, 1.48] |
| Grounding DINO tiny | uav | 2.11 [1.93, 2.34] | 1.99 [1.84, 2.16] | 2.81 [2.51, 3.21] |

## 3. How much of the overhead gap is left? (average marginal effect of overhead, percentage points)

M0: category only. M1: + size. M2: + size + margin. Closer to 0 = more of the overhead penalty is accounted for.

| Detector | outcome | M0 | M1 (+ size) | M2 CLIP | M2 RemoteCLIP | M2 OWLv2-internal |
|---|---|---|---|---|---|---|
| OWLv2 B/16 | found | -31.8 [-37.9, -24.1] | -23.4 [-29.2, -15.6] | -23.7 [-29.7, -15.6] | -21.2 [-26.7, -14.0] | -22.7 [-28.6, -14.9] |
| OWLv2 B/16 | named_given_found | +1.0 [-1.2, +3.0] | +2.4 [+0.5, +4.1] | +3.1 [+1.2, +5.0] | +2.5 [+0.5, +4.2] | +0.4 [-0.1, +1.0] |
| OWLv2 B/16 | success | -26.0 [-33.1, -18.5] | -16.1 [-22.5, -8.9] | -16.7 [-23.4, -9.0] | -13.9 [-19.9, -7.3] | -13.9 [-19.8, -7.1] |
| Grounding DINO tiny | found | -31.6 [-36.9, -25.5] | -23.0 [-28.2, -17.1] | -23.7 [-28.6, -17.8] | -21.5 [-26.4, -15.6] | -22.8 [-28.0, -16.8] |
| Grounding DINO tiny | named_given_found | -4.7 [-7.8, -1.9] | -2.1 [-4.8, +0.4] | -2.3 [-5.1, +0.5] | -1.4 [-4.0, +1.1] | -1.1 [-3.6, +1.3] |
| Grounding DINO tiny | success | -27.3 [-32.2, -22.1] | -17.9 [-22.8, -12.4] | -18.9 [-23.8, -13.7] | -15.9 [-20.7, -10.7] | -16.7 [-21.4, -11.4] |

## Size effect (M1), odds ratio per +1 SD of log2 size

| Detector | found | named given found | success |
|---|---|---|---|
| OWLv2 B/16 | 2.16 [1.99, 2.38] | 1.59 [1.44, 1.78] | 2.11 [1.96, 2.32] |
| Grounding DINO tiny | 2.48 [2.28, 2.73] | 1.46 [1.32, 1.63] | 2.23 [2.08, 2.41] |

## Success rate by margin quintile, within domain (descriptive; categories mixed)

| Encoder | detector | domain | Q1 (lowest margin) | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | OWLv2 B/16 | overhead | 24% | 23% | 27% | 48% | 62% |
| CLIP ViT-B/32 | OWLv2 B/16 | uav | 50% | 74% | 83% | 93% | 97% |
| CLIP ViT-B/32 | Grounding DINO tiny | overhead | 20% | 18% | 23% | 35% | 53% |
| CLIP ViT-B/32 | Grounding DINO tiny | uav | 46% | 64% | 68% | 84% | 93% |
| RemoteCLIP ViT-L/14 | OWLv2 B/16 | overhead | 44% | 45% | 26% | 21% | 47% |
| RemoteCLIP ViT-L/14 | OWLv2 B/16 | uav | 58% | 76% | 80% | 88% | 96% |
| RemoteCLIP ViT-L/14 | Grounding DINO tiny | overhead | 27% | 33% | 23% | 21% | 44% |
| RemoteCLIP ViT-L/14 | Grounding DINO tiny | uav | 56% | 63% | 62% | 79% | 96% |

## Median object margin by category and domain

| Category | CLIP ViT-B/32 O / U | RemoteCLIP ViT-L/14 O / U | OWLv2-internal O / U |
|---|---|---|---|
| small-vehicle | +0.022 / +0.017 | +0.007 / +0.011 | +0.012 / +0.014 |
| large-vehicle | -0.014 / -0.013 | -0.004 / -0.001 | +0.005 / -0.003 |
| car | +0.008 / +0.016 | -0.010 / +0.016 | +0.059 / +0.092 |
| bus | -0.010 / -0.002 | -0.029 / -0.002 | +0.005 / +0.021 |
| truck | -0.007 / -0.005 | -0.018 / -0.023 | -0.004 / +0.026 |
| building | -0.008 / +0.024 | +0.024 / +0.036 | +0.039 / +0.084 |
