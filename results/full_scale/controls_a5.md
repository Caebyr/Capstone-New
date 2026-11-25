# Controls: resolution matching and prompt sensitivity

## (a) Resolution control

2000 UAV crops per category, each shrunk to the size of a random overhead crop of the same category, then encoded with the same preprocessing. Drift = cosine drift; ratio = spread-normalized ratio. Accuracy = zero-shot crop classification, "a photo of a {}".

### P1 (median shrink factor: small-vehicle 0.74, large-vehicle 1.00)

| Model | Category | cos O vs U | cos O vs U-shrunk | ratio O vs U | ratio O vs U-shrunk | cos U vs U-shrunk |
|---|---|---|---|---|---|---|
| CLIP | small-vehicle | 0.0260 | 0.0110 | 0.567 | 0.396 | 0.0088 |
| CLIP | large-vehicle | 0.0276 | 0.0215 | 0.523 | 0.479 | 0.0027 |
| RemoteCLIP | small-vehicle | 0.0156 | 0.0096 | 0.515 | 0.428 | 0.0057 |
| RemoteCLIP | large-vehicle | 0.0205 | 0.0203 | 0.537 | 0.555 | 0.0018 |
| fine-tuned CLIP | small-vehicle | 0.0073 | 0.0057 | 0.642 | 0.557 | 0.0002 |
| fine-tuned CLIP | large-vehicle | 0.0031 | 0.0032 | 0.329 | 0.332 | 0.0002 |

| Model | Acc overhead | Acc UAV original | Acc UAV shrunk | UAV-shrunk predictions |
|---|---|---|---|---|
| CLIP | 50.8 | 55.3 | 54.4 | small-vehicle 94.2%, large-vehicle 5.8% |
| RemoteCLIP | 63.1 | 68.2 | 66.5 | small-vehicle 72.9%, large-vehicle 27.2% |
| fine-tuned CLIP | 83.6 | 83.5 | 82.0 | small-vehicle 62.6%, large-vehicle 37.4% |

### P2+P3 (median shrink factor: car 0.48, bus 0.49, truck 0.54, building 0.31)

| Model | Category | cos O vs U | cos O vs U-shrunk | ratio O vs U | ratio O vs U-shrunk | cos U vs U-shrunk |
|---|---|---|---|---|---|---|
| CLIP | car | 0.0724 | 0.0233 | 1.033 | 0.702 | 0.0330 |
| CLIP | bus | 0.0517 | 0.0166 | 0.793 | 0.524 | 0.0270 |
| CLIP | truck | 0.0558 | 0.0221 | 0.820 | 0.594 | 0.0225 |
| CLIP | building | 0.1471 | 0.0511 | 1.115 | 0.766 | 0.0641 |
| RemoteCLIP | car | 0.0487 | 0.0182 | 0.992 | 0.697 | 0.0236 |
| RemoteCLIP | bus | 0.0390 | 0.0179 | 0.805 | 0.614 | 0.0199 |
| RemoteCLIP | truck | 0.0350 | 0.0178 | 0.779 | 0.623 | 0.0158 |
| RemoteCLIP | building | 0.0691 | 0.0435 | 0.907 | 0.836 | 0.0345 |

| Model | Acc overhead | Acc UAV original | Acc UAV shrunk | UAV-shrunk predictions |
|---|---|---|---|---|
| CLIP | 35.2 | 59.1 | 46.3 | car 60.8%, bus 16.3%, truck 9.1%, building 13.8% |
| RemoteCLIP | 31.1 | 48.6 | 39.7 | car 26.0%, bus 9.6%, truck 2.1%, building 62.3% |

## (b) Prompt sensitivity

Accuracy overhead / UAV (%), full cache, per prompt template; last column = share of overhead *vehicle* crops predicted as each class for the ensemble.

### P1

| Model | `a photo of a {}` | `a satellite photo of a {}` | `an aerial photo of a {}` | `a drone photo of a {}` | `{}` | ensemble |
|---|---|---|---|---|---|---|
| CLIP | 50.8 / 54.8 | 50.8 / 53.5 | 50.1 / 52.4 | 50.6 / 55.9 | 54.5 / 56.8 | 51.1 / 55.6 |
| RemoteCLIP | 63.1 / 68.0 | 60.2 / 62.8 | 58.4 / 61.2 | 61.5 / 64.2 | 62.5 / 63.6 | 61.6 / 64.4 |
| fine-tuned CLIP | 83.6 / 83.0 | 84.0 / 83.7 | 84.1 / 83.0 | 83.8 / 83.9 | 81.4 / 84.1 | 83.2 / 83.7 |

### P2+P3

| Model | `a photo of a {}` | `a satellite photo of a {}` | `an aerial photo of a {}` | `a drone photo of a {}` | `{}` | ensemble |
|---|---|---|---|---|---|---|
| CLIP | 35.2 / 57.0 | 34.6 / 60.6 | 31.6 / 58.8 | 31.8 / 56.8 | 38.0 / 59.9 | 36.0 / 60.4 |
| RemoteCLIP | 31.1 / 44.6 | 44.1 / 53.4 | 40.3 / 51.8 | 33.2 / 46.3 | 44.0 / 54.0 | 39.4 / 49.9 |

