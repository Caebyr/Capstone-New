# Visual drift with image-level uncertainty

Point estimates reproduce Tables II-III. CIs: 95% cluster bootstrap over source images, matched across models, independent between domains. p: image-level permutation test on L2. Seq-level CIs resample UAV crops by video sequence instead (sensitivity check).

## P1  (images: overhead 149, UAV 435; UAV sequences 98)

| Category | Model | L2 [95% CI] | Ratio [95% CI] | Ratio [seq-level CI] | perm p | L2 / null max |
|---|---|---|---|---|---|---|
| small-vehicle | CLIP ViT-B/32 | 0.2460 [0.224, 0.280] | 0.6498 [0.596, 0.741] | [0.585, 0.741] | 0.0005 | 2.7x |
| small-vehicle | RemoteCLIP ViT-L/14 | 0.1892 [0.171, 0.224] | 0.5627 [0.511, 0.671] | [0.503, 0.672] | 0.0005 | 1.8x |
| small-vehicle | CLIP fine-tuned on VisDrone | 0.1585 [0.116, 0.207] | 0.8165 [0.630, 1.039] | [0.626, 1.042] | 0.0005 | 2.3x |
| large-vehicle | CLIP ViT-B/32 | 0.2596 [0.237, 0.295] | 0.6235 [0.573, 0.711] | [0.563, 0.724] | 0.0005 | 1.8x |
| large-vehicle | RemoteCLIP ViT-L/14 | 0.2276 [0.207, 0.260] | 0.6252 [0.574, 0.720] | [0.571, 0.721] | 0.0005 | 1.5x |
| large-vehicle | CLIP fine-tuned on VisDrone | 0.0955 [0.079, 0.122] | 0.4132 [0.338, 0.539] | [0.339, 0.552] | 0.0005 | 1.0x |

Paired change vs CLIP (% , 95% CI, image-level):

| Category | Model | L2 change | Ratio change | Radius change |
|---|---|---|---|---|
| small-vehicle | RemoteCLIP ViT-L/14 | [-27.5, -17.0] | [-18.5, -6.2] | [-12.7, -10.0] |
| small-vehicle | CLIP fine-tuned on VisDrone | [-51.9, -18.9] | [-1.5, 52.2] | [-52.3, -45.4] |
| large-vehicle | RemoteCLIP ViT-L/14 | [-18.9, -3.5] | [-8.0, 11.0] | [-14.6, -10.6] |
| large-vehicle | CLIP fine-tuned on VisDrone | [-69.0, -54.7] | [-45.4, -17.6] | [-47.2, -41.7] |

## P2+P3  (images: overhead 689, UAV 931; UAV sequences 134)

| Category | Model | L2 [95% CI] | Ratio [95% CI] | Ratio [seq-level CI] | perm p | L2 / null max |
|---|---|---|---|---|---|---|
| car | CLIP ViT-B/32 | 0.3564 [0.332, 0.382] | 0.9931 [0.931, 1.063] | [0.894, 1.105] | 0.0005 | 3.9x |
| car | RemoteCLIP ViT-L/14 | 0.3024 [0.278, 0.328] | 0.9719 [0.895, 1.056] | [0.860, 1.096] | 0.0005 | 4.0x |
| bus | CLIP ViT-B/32 | 0.3256 [0.304, 0.350] | 0.8443 [0.792, 0.906] | [0.794, 0.931] | 0.0005 | 3.8x |
| bus | RemoteCLIP ViT-L/14 | 0.2802 [0.260, 0.303] | 0.8491 [0.794, 0.917] | [0.795, 0.946] | 0.0005 | 4.3x |
| truck | CLIP ViT-B/32 | 0.2837 [0.264, 0.308] | 0.7550 [0.710, 0.813] | [0.669, 0.858] | 0.0005 | 3.9x |
| truck | RemoteCLIP ViT-L/14 | 0.2366 [0.220, 0.258] | 0.7500 [0.704, 0.809] | [0.675, 0.846] | 0.0005 | 3.9x |
| building | CLIP ViT-B/32 | 0.5417 [0.513, 0.573] | 1.2219 [1.145, 1.318] | [1.153, 1.330] | 0.0005 | 3.1x |
| building | RemoteCLIP ViT-L/14 | 0.3783 [0.355, 0.404] | 0.9809 [0.901, 1.082] | [0.907, 1.096] | 0.0005 | 3.1x |

Paired change vs CLIP (% , 95% CI, image-level):

| Category | Model | L2 change | Ratio change | Radius change | Y diff |
|---|---|---|---|---|---|
| car | RemoteCLIP ViT-L/14 | [-18.7, -11.6] | [-6.8, 2.4] | [-14.7, -11.8] | [-0.03, 0.30] |
| bus | RemoteCLIP ViT-L/14 | [-17.4, -10.3] | [-3.7, 5.0] | [-15.7, -13.1] | [0.29, 0.67] |
| truck | RemoteCLIP ViT-L/14 | [-20.1, -12.8] | [-4.7, 4.0] | [-17.4, -14.6] | [0.10, 0.52] |
| building | RemoteCLIP ViT-L/14 | [-32.1, -27.9] | [-23.2, -15.9] | [-14.7, -11.2] | [-0.35, -0.05] |

Spacing-normalized drift Y_c (Table III) with image-level CI:

| Category | CLIP ViT-B/32 | RemoteCLIP ViT-L/14 |
|---|---|---|
| car | 1.95 [1.65, 2.21] | 2.09 [1.78, 2.34] |
| bus | 1.81 [1.48, 2.09] | 2.30 [1.94, 2.58] |
| truck | 1.50 [1.21, 1.80] | 1.81 [1.43, 2.18] |
| building | 2.32 [2.17, 2.42] | 2.12 [1.87, 2.32] |

Note: P1 fine-tuned large-vehicle is the only cell whose observed L2 sits near the 2,000-permutation null maximum
(0.0955 vs 0.0938). Re-run with 20,000 image-level permutations: p = 0.0003 (null 99.9th percentile 0.0855).
