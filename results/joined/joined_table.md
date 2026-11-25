# Visual drift vs crop classification vs text alignment (joined, exploratory)

All from the same crops (300 per category and domain, seed 42). Brackets: 95% image-level bootstrap CI.
Recall gap = overhead - UAV in percentage points (positive = worse on UAV). delta_m = margin(overhead) - margin(UAV) (positive = smaller UAV margin). Fine-tuned CLIP: supervised on the UAV side of P1, not zero-shot.

## P1

| Category | Model | L2 drift | Cos drift | Ratio | Recall overhead → UAV | Recall gap (pp) | A_c | Margin overhead → UAV | Δm_c |
|---|---|---|---|---|---|---|---|---|---|
| small-vehicle | CLIP ViT-B/32 | 0.246 [0.224, 0.280] | 0.0346 | 0.650 [0.596, 0.741] | 99.0% → 99.3% | -0.3 [-2.0, +1.0] | 0.0232 | +0.0210 → +0.0214 | -0.0004 [-0.0047, +0.0039] |
| small-vehicle | RemoteCLIP ViT-L/14 | 0.189 [0.171, 0.224] | 0.0199 | 0.563 [0.511, 0.671] | 81.3% → 91.3% | -10.0 [-15.7, -3.8] | 0.0000 | +0.0055 → +0.0126 | -0.0071 [-0.0085, -0.0056] |
| small-vehicle | CLIP fine-tuned on VisDrone | 0.159 [0.116, 0.207] | 0.0129 | 0.816 [0.630, 1.039] | 86.3% → 96.3% | -10.0 [-15.2, -5.2] | 0.0101 | +0.0349 → +0.0535 | -0.0186 [-0.0255, -0.0126] |
| large-vehicle | CLIP ViT-B/32 | 0.260 [0.237, 0.295] | 0.0406 | 0.624 [0.573, 0.711] | 3.3% → 9.7% | -6.3 [-10.2, -2.2] | 0.0307 | -0.0176 → -0.0137 | -0.0039 [-0.0074, +0.0003] |
| large-vehicle | RemoteCLIP ViT-L/14 | 0.228 [0.207, 0.260] | 0.0300 | 0.625 [0.574, 0.720] | 52.3% → 42.7% | +9.7 [-4.9, +23.6] | 0.0078 | +0.0061 → +0.0033 | +0.0028 [-0.0028, +0.0079] |
| large-vehicle | CLIP fine-tuned on VisDrone | 0.095 [0.079, 0.122] | 0.0048 | 0.413 [0.338, 0.539] | 83.3% → 72.0% | +11.3 [+3.2, +19.1] | 0.0061 | +0.0415 → +0.0283 | +0.0132 [+0.0049, +0.0206] |

## P2+P3

| Category | Model | L2 drift | Cos drift | Ratio | Y_c | Recall overhead → UAV | Recall gap (pp) | A_c | Margin overhead → UAV | Δm_c |
|---|---|---|---|---|---|---|---|---|---|---|
| car | CLIP ViT-B/32 | 0.356 [0.332, 0.382] | 0.0718 | 0.993 [0.931, 1.063] | 1.95 [1.65, 2.21] | 92.0% → 87.7% | +4.3 [-0.7, +9.1] | 0.0333 | +0.0085 → +0.0227 | -0.0142 [-0.0165, -0.0119] |
| car | RemoteCLIP ViT-L/14 | 0.302 [0.278, 0.328] | 0.0503 | 0.972 [0.895, 1.056] | 2.09 [1.78, 2.34] | 8.3% → 63.3% | -55.0 [-61.3, -48.8] | 0.0367 | -0.0119 → +0.0176 | -0.0295 [-0.0328, -0.0264] |
| bus | CLIP ViT-B/32 | 0.326 [0.304, 0.350] | 0.0603 | 0.844 [0.792, 0.906] | 1.81 [1.48, 2.09] | 1.0% → 41.3% | -40.3 [-46.6, -34.6] | 0.0393 | -0.0119 → +0.0093 | -0.0212 [-0.0245, -0.0180] |
| bus | RemoteCLIP ViT-L/14 | 0.280 [0.260, 0.303] | 0.0434 | 0.849 [0.794, 0.917] | 2.30 [1.94, 2.58] | 1.7% → 41.0% | -39.3 [-44.9, -33.5] | 0.0477 | -0.0251 → +0.0038 | -0.0289 [-0.0335, -0.0245] |
| truck | CLIP ViT-B/32 | 0.284 [0.264, 0.308] | 0.0451 | 0.755 [0.710, 0.813] | 1.50 [1.21, 1.80] | 10.0% → 30.3% | -20.3 [-26.4, -14.2] | 0.0299 | -0.0073 → +0.0028 | -0.0101 [-0.0124, -0.0078] |
| truck | RemoteCLIP ViT-L/14 | 0.237 [0.220, 0.258] | 0.0303 | 0.750 [0.704, 0.809] | 1.81 [1.43, 2.18] | 0.7% → 11.7% | -11.0 [-15.0, -7.3] | 0.0230 | -0.0169 → -0.0152 | -0.0017 [-0.0053, +0.0016] |
| building | CLIP ViT-B/32 | 0.542 [0.513, 0.573] | 0.1822 | 1.222 [1.145, 1.318] | 2.32 [2.17, 2.42] | 28.7% → 81.7% | -53.0 [-59.6, -46.4] | 0.0581 | -0.0039 → +0.0311 | -0.0350 [-0.0385, -0.0317] |
| building | RemoteCLIP ViT-L/14 | 0.378 [0.355, 0.404] | 0.0847 | 0.981 [0.901, 1.082] | 2.12 [1.87, 2.32] | 98.3% → 91.3% | +7.0 [+3.6, +10.8] | 0.0374 | +0.0228 → +0.0393 | -0.0165 [-0.0209, -0.0118] |

## Does larger drift go with a larger gap? (exploratory, within task and model, across categories)

Spearman rank correlation across the task's categories. P1 has 2 categories (no correlation possible); P2+P3 has 4, where |rho| = 1 occurs by chance with probability 1/12 and nothing here is a statistical test.

| Task | Model | drift measure | vs abs(recall gap) | vs A_c | vs abs(Δm_c) |
|---|---|---|---|---|---|
| P2+P3 | CLIP ViT-B/32 | cos | +0.40 | +0.80 | +0.80 |
| P2+P3 | CLIP ViT-B/32 | ratio | +0.40 | +0.80 | +0.80 |
| P2+P3 | CLIP ViT-B/32 | Y | +0.40 | +0.80 | +0.80 |
| P2+P3 | RemoteCLIP ViT-L/14 | cos | -0.20 | +0.40 | +0.40 |
| P2+P3 | RemoteCLIP ViT-L/14 | ratio | -0.20 | +0.40 | +0.40 |
| P2+P3 | RemoteCLIP ViT-L/14 | Y | +0.00 | +1.00 | +0.40 |

Pooled over both encoders (8 category-encoder points), cosine drift vs abs(recall gap) rho = +0.02, vs A_c rho = +0.52, vs abs(Δm_c) rho = +0.48.
