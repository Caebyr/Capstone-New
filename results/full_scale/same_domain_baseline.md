# Same-domain baseline: is platform drift larger than dataset drift?

L0 = two image sets of one dataset (noise floor); L1 = two datasets from the same platform; L2 = overhead vs UAV (the paper's kind of comparison). Values: full cache [95% image-bootstrap CI]; matched = mean [min–max] over 20 draws of 300 crops per side.

Caveats: DOTA small/large-vehicle vs xView car / bus+truck+van are approximate category matches; UAVid car boxes are mask components, so adjacent parked cars can merge into one crop.

## Cosine drift

| Level | Comparison | Category | n (A / B) | CLIP full [CI] | CLIP matched-300 | RemoteCLIP full [CI] | RemoteCLIP matched-300 |
|---|---|---|---|---|---|---|---|
| L0 | DOTA part1 vs part2/3 | small-vehicle | 1382 / 3618 | 0.0007 [0.0006, 0.0022] | 0.0010 [0.0006–0.0015] | 0.0007 [0.0005, 0.0024] | 0.0009 [0.0007–0.0013] |
| L0 | DOTA part1 vs part2/3 | large-vehicle | 1044 / 3956 | 0.0037 [0.0029, 0.0085] | 0.0041 [0.0032–0.0058] | 0.0044 [0.0028, 0.0097] | 0.0047 [0.0035–0.0070] |
| L0 | VisDrone train vs held-out images | car | 3993 / 1007 | 0.0001 [0.0002, 0.0006] | 0.0007 [0.0004–0.0020] | 0.0001 [0.0001, 0.0004] | 0.0006 [0.0003–0.0013] |
| L0 | xView first vs second half of chips | car | 1615 / 3385 | 0.0006 [0.0003, 0.0011] | 0.0009 [0.0005–0.0014] | 0.0004 [0.0002, 0.0008] | 0.0005 [0.0003–0.0010] |
| L0 | xView first vs second half of chips | building | 2499 / 2501 | 0.0003 [0.0002, 0.0005] | 0.0007 [0.0003–0.0010] | 0.0002 [0.0002, 0.0004] | 0.0006 [0.0003–0.0009] |
| L1 | DOTA vs xView (overhead vs overhead) | small-vehicle / car | 5000 / 5000 | 0.0254 [0.0229, 0.0281] | 0.0252 [0.0217–0.0282] | 0.0233 [0.0200, 0.0270] | 0.0234 [0.0200–0.0251] |
| L1 | DOTA vs xView (overhead vs overhead) | large-vehicle / bus+truck+van | 5000 / 8601 | 0.0306 [0.0273, 0.0343] | 0.0310 [0.0256–0.0337] | 0.0282 [0.0245, 0.0324] | 0.0285 [0.0237–0.0328] |
| L1 | VisDrone vs UAVid (UAV vs UAV) | car | 5000 / 391 | 0.0265 [0.0232, 0.0313] | 0.0265 [0.0251–0.0291] | 0.0166 [0.0137, 0.0205] | 0.0168 [0.0151–0.0185] |
| L2 | DOTA vs VisDrone (all images) | small-vehicle | 5000 / 5000 | 0.0281 [0.0260, 0.0308] | 0.0293 [0.0257–0.0358] | 0.0171 [0.0156, 0.0190] | 0.0177 [0.0133–0.0224] |
| L2 | DOTA vs VisDrone (all images) | large-vehicle | 5000 / 5000 | 0.0289 [0.0266, 0.0316] | 0.0298 [0.0228–0.0352] | 0.0222 [0.0207, 0.0243] | 0.0229 [0.0181–0.0287] |
| L2 | xView vs VisDrone | car | 5000 / 5000 | 0.0719 [0.0691, 0.0749] | 0.0735 [0.0666–0.0796] | 0.0486 [0.0462, 0.0511] | 0.0496 [0.0437–0.0595] |
| L2 | xView vs UAVid | car | 5000 / 391 | 0.1083 [0.0980, 0.1189] | 0.1089 [0.1040–0.1135] | 0.0751 [0.0676, 0.0824] | 0.0750 [0.0710–0.0773] |
| L2 | xView vs UAVid | building | 5000 / 1199 | 0.1471 [0.1390, 0.1559] | 0.1449 [0.1259–0.1616] | 0.0691 [0.0655, 0.0735] | 0.0696 [0.0636–0.0810] |

## Spread-normalized ratio

| Level | Comparison | Category | n (A / B) | CLIP full [CI] | CLIP matched-300 | RemoteCLIP full [CI] | RemoteCLIP matched-300 |
|---|---|---|---|---|---|---|---|
| L0 | DOTA part1 vs part2/3 | small-vehicle | 1382 / 3618 | 0.109 [0.098, 0.190] | 0.130 [0.101–0.159] | 0.120 [0.108, 0.225] | 0.141 [0.125–0.168] |
| L0 | DOTA part1 vs part2/3 | large-vehicle | 1044 / 3956 | 0.208 [0.184, 0.320] | 0.218 [0.195–0.264] | 0.262 [0.211, 0.395] | 0.270 [0.233–0.331] |
| L0 | VisDrone train vs held-out images | car | 3993 / 1007 | 0.036 [0.041, 0.076] | 0.078 [0.062–0.135] | 0.034 [0.040, 0.076] | 0.085 [0.065–0.130] |
| L0 | xView first vs second half of chips | car | 1615 / 3385 | 0.125 [0.092, 0.171] | 0.155 [0.115–0.195] | 0.107 [0.079, 0.163] | 0.131 [0.091–0.173] |
| L0 | xView first vs second half of chips | building | 2499 / 2501 | 0.063 [0.053, 0.090] | 0.100 [0.070–0.123] | 0.061 [0.055, 0.088] | 0.101 [0.072–0.132] |
| L1 | DOTA vs xView (overhead vs overhead) | small-vehicle / car | 5000 / 5000 | 0.721 [0.688, 0.760] | 0.720 [0.661–0.762] | 0.779 [0.721, 0.841] | 0.778 [0.720–0.817] |
| L1 | DOTA vs xView (overhead vs overhead) | large-vehicle / bus+truck+van | 5000 / 8601 | 0.681 [0.646, 0.721] | 0.689 [0.623–0.732] | 0.743 [0.697, 0.793] | 0.749 [0.682–0.811] |
| L1 | VisDrone vs UAVid (UAV vs UAV) | car | 5000 / 391 | 0.471 [0.442, 0.512] | 0.472 [0.456–0.496] | 0.452 [0.413, 0.503] | 0.457 [0.429–0.476] |
| L2 | DOTA vs VisDrone (all images) | small-vehicle | 5000 / 5000 | 0.587 [0.564, 0.615] | 0.599 [0.563–0.650] | 0.538 [0.514, 0.568] | 0.545 [0.470–0.611] |
| L2 | DOTA vs VisDrone (all images) | large-vehicle | 5000 / 5000 | 0.532 [0.509, 0.559] | 0.542 [0.480–0.587] | 0.557 [0.536, 0.584] | 0.565 [0.513–0.626] |
| L2 | xView vs VisDrone | car | 5000 / 5000 | 1.028 [1.009, 1.048] | 1.041 [0.989–1.080] | 0.987 [0.962, 1.014] | 1.002 [0.948–1.111] |
| L2 | xView vs UAVid | car | 5000 / 391 | 1.221 [1.160, 1.287] | 1.227 [1.205–1.247] | 1.178 [1.119, 1.239] | 1.177 [1.150–1.205] |
| L2 | xView vs UAVid | building | 5000 / 1199 | 1.115 [1.088, 1.146] | 1.111 [1.034–1.163] | 0.907 [0.885, 0.933] | 0.911 [0.863–0.991] |

