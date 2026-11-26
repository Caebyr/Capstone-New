# Image-quality control: size, blur and colour matched UAV crops (C3)

P2+P3 (xView vs VisDrone / UAVid). The same UAV crops and overhead size donors as A5. S = shrink to the donor's size (A5); B = Gaussian blur calibrated so median crop sharpness matches the overhead crops; C = per-channel mean / std transferred from the donor crop. Prompt "a photo of a {}".

## Calibration

| Category | overhead sharpness | UAV original | UAV shrunk | sigma chosen | UAV shrunk + blur |
|---|---|---|---|---|---|
| car | 0.642 | 0.315 | 0.511 | 0.0 | 0.511 |
| bus | 0.412 | 0.285 | 0.451 | 0.25 | 0.431 |
| truck | 0.447 | 0.275 | 0.412 | 0.0 | 0.412 |
| building | 0.303 | 0.211 | 0.376 | 0.5 | 0.287 |

## CLIP ViT-B/32

Cosine drift, overhead vs UAV variant:

| Category | original | S | S+B | S+C | S+B+C | S+Cu |
|---|---|---|---|---|---|---|
| car | 0.0724 | 0.0233 | 0.0233 | 0.0109 | 0.0109 | 0.0179 |
| bus | 0.0517 | 0.0166 | 0.0165 | 0.0168 | 0.0164 | 0.0166 |
| truck | 0.0558 | 0.0221 | 0.0221 | 0.0123 | 0.0123 | 0.0164 |
| building | 0.1471 | 0.0511 | 0.0479 | 0.0473 | 0.0437 | 0.0461 |

Zero-shot classification on the UAV variant (overhead crops for reference):

| | overhead | original | S | S+B | S+C | S+B+C | S+Cu |
|---|---|---|---|---|---|---|---|
| balanced accuracy | 32.6% | 61.5% | 47.7% | 45.8% | 43.8% | 41.2% | 44.3% |
| accuracy | 35.2% | 59.1% | 46.3% | 45.1% | 43.5% | 41.7% | 43.1% |
| margin car | +0.0098 | +0.0240 | +0.0134 | +0.0134 | +0.0095 | +0.0095 | +0.0124 |
| margin bus | -0.0113 | +0.0083 | -0.0019 | -0.0016 | -0.0065 | -0.0063 | -0.0057 |
| margin truck | -0.0069 | +0.0051 | -0.0032 | -0.0032 | -0.0014 | -0.0014 | -0.0040 |
| margin building | -0.0068 | +0.0306 | +0.0099 | +0.0040 | +0.0049 | -0.0001 | +0.0070 |
| share labelled 'building' | 9% | 19% | 14% | 12% | 13% | 11% | 13% |

Share of the balanced-accuracy gap (UAV original − overhead) closed by each variant: S 48%, S+B 54%, S+C 61%, S+B+C 70%, S+Cu 59%.

Overhead-specific colour effect (S+C minus the UAV-donor placebo S+Cu): +2 points of the gap; car drift 0.0109 vs placebo 0.0179. Whatever the placebo also achieves is generic perturbation, not a move toward the overhead look.

## RemoteCLIP ViT-L/14

Cosine drift, overhead vs UAV variant:

| Category | original | S | S+B | S+C | S+B+C | S+Cu |
|---|---|---|---|---|---|---|
| car | 0.0487 | 0.0182 | 0.0182 | 0.0076 | 0.0076 | 0.0142 |
| bus | 0.0390 | 0.0179 | 0.0181 | 0.0094 | 0.0100 | 0.0136 |
| truck | 0.0350 | 0.0178 | 0.0178 | 0.0059 | 0.0059 | 0.0145 |
| building | 0.0691 | 0.0435 | 0.0472 | 0.0272 | 0.0332 | 0.0368 |

Zero-shot classification on the UAV variant (overhead crops for reference):

| | overhead | original | S | S+B | S+C | S+B+C | S+Cu |
|---|---|---|---|---|---|---|---|
| balanced accuracy | 28.9% | 52.7% | 45.2% | 45.3% | 37.0% | 37.1% | 38.8% |
| accuracy | 31.1% | 48.6% | 39.7% | 39.7% | 30.2% | 30.4% | 32.3% |
| margin car | -0.0098 | +0.0184 | +0.0079 | +0.0079 | -0.0037 | -0.0037 | -0.0011 |
| margin bus | -0.0258 | +0.0038 | -0.0044 | -0.0049 | -0.0170 | -0.0171 | -0.0166 |
| margin truck | -0.0180 | -0.0110 | -0.0141 | -0.0141 | -0.0159 | -0.0159 | -0.0206 |
| margin building | +0.0297 | +0.0376 | +0.0302 | +0.0282 | +0.0325 | +0.0285 | +0.0349 |
| share labelled 'building' | 93% | 53% | 62% | 63% | 81% | 81% | 77% |

Share of the balanced-accuracy gap (UAV original − overhead) closed by each variant: S 32%, S+B 31%, S+C 66%, S+B+C 65%, S+Cu 59%.

Overhead-specific colour effect (S+C minus the UAV-donor placebo S+Cu): +7 points of the gap; car drift 0.0076 vs placebo 0.0142. Whatever the placebo also achieves is generic perturbation, not a move toward the overhead look.

## CLIP ViT-L/14

Cosine drift, overhead vs UAV variant:

| Category | original | S | S+B | S+C | S+B+C | S+Cu |
|---|---|---|---|---|---|---|
| car | 0.0803 | 0.0365 | 0.0365 | 0.0261 | 0.0261 | 0.0328 |
| bus | 0.0587 | 0.0253 | 0.0266 | 0.0229 | 0.0236 | 0.0247 |
| truck | 0.0561 | 0.0256 | 0.0256 | 0.0173 | 0.0173 | 0.0221 |
| building | 0.1551 | 0.0700 | 0.0713 | 0.0543 | 0.0550 | 0.0599 |

Zero-shot classification on the UAV variant (overhead crops for reference):

| | overhead | original | S | S+B | S+C | S+B+C | S+Cu |
|---|---|---|---|---|---|---|---|
| balanced accuracy | 34.5% | 73.6% | 63.4% | 62.7% | 55.6% | 54.6% | 59.5% |
| accuracy | 36.2% | 71.5% | 61.3% | 60.9% | 54.8% | 54.2% | 57.5% |
| margin car | -0.0125 | +0.0213 | +0.0069 | +0.0069 | -0.0004 | -0.0004 | +0.0040 |
| margin bus | -0.0063 | +0.0234 | +0.0099 | +0.0100 | +0.0039 | +0.0040 | +0.0080 |
| margin truck | +0.0090 | +0.0232 | +0.0110 | +0.0110 | +0.0119 | +0.0119 | +0.0093 |
| margin building | -0.0065 | +0.0454 | +0.0306 | +0.0283 | +0.0171 | +0.0130 | +0.0272 |
| share labelled 'building' | 8% | 21% | 18% | 17% | 13% | 13% | 16% |

Share of the balanced-accuracy gap (UAV original − overhead) closed by each variant: S 26%, S+B 28%, S+C 46%, S+B+C 49%, S+Cu 36%.

Overhead-specific colour effect (S+C minus the UAV-donor placebo S+Cu): +10 points of the gap; car drift 0.0261 vs placebo 0.0328. Whatever the placebo also achieves is generic perturbation, not a move toward the overhead look.
