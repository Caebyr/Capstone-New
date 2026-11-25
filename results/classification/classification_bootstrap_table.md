# Ground-truth crop classification and text alignment

Prompt: "a photo of a {}". Zero-shot argmax over the task's candidates. Gap = overhead - UAV (positive = worse on UAV). delta_m > 0 = smaller UAV centroid margin.
Fine-tuned CLIP was trained on the UAV side of P1 (other images) - its UAV numbers are not zero-shot.
A_c is an absolute value, so its bootstrap interval cannot cross 0 and can sit above a near-zero point estimate; read the sign and size of a_overhead - a_UAV from the a columns instead.

## P1  candidates: small-vehicle, large-vehicle

| Model | Acc overhead | Acc UAV | Acc gap (pp) | Macro-F1 overhead | Macro-F1 UAV |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | 51.2% | 54.5% | -3.3 [-16.0, +9.1] | 0.367 | 0.431 |
| RemoteCLIP ViT-L/14 | 66.8% | 67.0% | -0.2 [-7.2, +7.3] | 0.661 | 0.649 |
| CLIP fine-tuned on VisDrone | 84.8% | 84.2% | +0.7 [-4.0, +5.3] | 0.848 | 0.839 |

| Category | Model | Recall overhead | Recall UAV | Recall gap (pp) | a overhead | a UAV | A_c | margin overhead | margin UAV | delta_m |
|---|---|---|---|---|---|---|---|---|---|---|
| small-vehicle | CLIP ViT-B/32 | 99.0% | 99.3% | -0.3 [-2.0, +1.0] | 0.2428 | 0.2660 | 0.0232 | +0.0210 | +0.0214 | -0.0004 [-0.0047, +0.0039] |
| small-vehicle | RemoteCLIP ViT-L/14 | 81.3% | 91.3% | -10.0 [-15.7, -3.8] | 0.2358 | 0.2358 | 0.0000 | +0.0055 | +0.0126 | -0.0071 [-0.0085, -0.0056] |
| small-vehicle | CLIP fine-tuned on VisDrone | 86.3% | 96.3% | -10.0 [-15.2, -5.2] | 0.2874 | 0.2975 | 0.0101 | +0.0349 | +0.0535 | -0.0186 [-0.0255, -0.0126] |
| large-vehicle | CLIP ViT-B/32 | 3.3% | 9.7% | -6.3 [-10.2, -2.2] | 0.2200 | 0.2506 | 0.0307 | -0.0176 | -0.0137 | -0.0039 [-0.0074, +0.0003] |
| large-vehicle | RemoteCLIP ViT-L/14 | 52.3% | 42.7% | +9.7 [-4.9, +23.6] | 0.2271 | 0.2348 | 0.0078 | +0.0061 | +0.0033 | +0.0028 [-0.0028, +0.0079] |
| large-vehicle | CLIP fine-tuned on VisDrone | 83.3% | 72.0% | +11.3 [+3.2, +19.1] | 0.3060 | 0.2999 | 0.0061 | +0.0415 | +0.0283 | +0.0132 [+0.0049, +0.0206] |

Confusion matrices (rows = true, columns = predicted, order: small-vehicle, large-vehicle):

- CLIP ViT-B/32, overhead: [[297, 3], [290, 10]]
- CLIP ViT-B/32, uav: [[298, 2], [271, 29]]
- RemoteCLIP ViT-L/14, overhead: [[244, 56], [143, 157]]
- RemoteCLIP ViT-L/14, uav: [[274, 26], [172, 128]]
- CLIP fine-tuned on VisDrone, overhead: [[259, 41], [50, 250]]
- CLIP fine-tuned on VisDrone, uav: [[289, 11], [84, 216]]

## P2+P3  candidates: car, bus, truck, building

| Model | Acc overhead | Acc UAV | Acc gap (pp) | Macro-F1 overhead | Macro-F1 UAV |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | 32.9% | 60.2% | -27.3 [-31.6, -23.3] | 0.255 | 0.583 |
| RemoteCLIP ViT-L/14 | 27.3% | 51.8% | -24.6 [-30.1, -19.3] | 0.150 | 0.482 |

| Category | Model | Recall overhead | Recall UAV | Recall gap (pp) | a overhead | a UAV | A_c | margin overhead | margin UAV | delta_m |
|---|---|---|---|---|---|---|---|---|---|---|
| car | CLIP ViT-B/32 | 92.0% | 87.7% | +4.3 [-0.7, +9.1] | 0.2348 | 0.2681 | 0.0333 | +0.0085 | +0.0227 | -0.0142 [-0.0165, -0.0119] |
| car | RemoteCLIP ViT-L/14 | 8.3% | 63.3% | -55.0 [-61.3, -48.8] | 0.2062 | 0.2428 | 0.0367 | -0.0119 | +0.0176 | -0.0295 [-0.0328, -0.0264] |
| bus | CLIP ViT-B/32 | 1.0% | 41.3% | -40.3 [-46.6, -34.6] | 0.2219 | 0.2612 | 0.0393 | -0.0119 | +0.0093 | -0.0212 [-0.0245, -0.0180] |
| bus | RemoteCLIP ViT-L/14 | 1.7% | 41.0% | -39.3 [-44.9, -33.5] | 0.1979 | 0.2456 | 0.0477 | -0.0251 | +0.0038 | -0.0289 [-0.0335, -0.0245] |
| truck | CLIP ViT-B/32 | 10.0% | 30.3% | -20.3 [-26.4, -14.2] | 0.2259 | 0.2559 | 0.0299 | -0.0073 | +0.0028 | -0.0101 [-0.0124, -0.0078] |
| truck | RemoteCLIP ViT-L/14 | 0.7% | 11.7% | -11.0 [-15.0, -7.3] | 0.2047 | 0.2276 | 0.0230 | -0.0169 | -0.0152 | -0.0017 [-0.0053, +0.0016] |
| building | CLIP ViT-B/32 | 28.7% | 81.7% | -53.0 [-59.6, -46.4] | 0.2256 | 0.2836 | 0.0581 | -0.0039 | +0.0311 | -0.0350 [-0.0385, -0.0317] |
| building | RemoteCLIP ViT-L/14 | 98.3% | 91.3% | +7.0 [+3.6, +10.8] | 0.2339 | 0.2713 | 0.0374 | +0.0228 | +0.0393 | -0.0165 [-0.0209, -0.0118] |

Confusion matrices (rows = true, columns = predicted, order: car, bus, truck, building):

- CLIP ViT-B/32, overhead: [[276, 0, 19, 5], [280, 3, 13, 4], [257, 0, 30, 13], [117, 2, 95, 86]]
- CLIP ViT-B/32, uav: [[263, 14, 21, 2], [116, 124, 28, 32], [133, 47, 91, 29], [34, 19, 2, 245]]
- RemoteCLIP ViT-L/14, overhead: [[25, 2, 0, 273], [12, 5, 1, 282], [12, 2, 2, 284], [5, 0, 0, 295]]
- RemoteCLIP ViT-L/14, uav: [[190, 7, 0, 103], [36, 123, 3, 138], [49, 20, 35, 196], [10, 10, 6, 274]]
