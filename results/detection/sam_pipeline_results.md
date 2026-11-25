# SAM proposals + zero-shot naming (CLIP / RemoteCLIP) vs open-vocabulary detectors

Same image subset for every pipeline (sizes: {'dota_tiles': 120, 'visdrone_ho_D1': 150, 'visdrone_ho_D2': 150, 'xview_val': 300, 'uavid_tv': 150}). AP50 per category; class-agnostic AP50; found = GT overlapped (IoU >= 0.5) by a detection with score >= 0.1; named = of found, share named correctly.

## dota_tiles

| Pipeline | AP50 small-vehicle | AP50 large-vehicle | agnostic AP50 | found | named |
|---|---|---|---|---|---|
| SAM + CLIP | 14.8 | 0.0 | 13.3 | 22.5 | 89.3 |
| SAM + RemoteCLIP | 7.6 | 2.5 | 9.1 | 23.0 | 77.8 |
| OWLv2 | 33.2 | 16.8 | 34.5 | 59.5 | 91.3 |
| Grounding DINO | 16.4 | 1.5 | 14.3 | 48.0 | 80.1 |

SAM proposal recall (IoU ≥ 0.5, before naming; 81 proposals/image): small-vehicle 51.1% (median short side 29 px), large-vehicle 72.6% (median short side 35 px)

## visdrone_ho_D1

| Pipeline | AP50 small-vehicle | AP50 large-vehicle | agnostic AP50 | found | named |
|---|---|---|---|---|---|
| SAM + CLIP | 30.2 | 2.4 | 32.7 | 42.7 | 79.5 |
| SAM + RemoteCLIP | 17.8 | 14.4 | 20.9 | 25.4 | 86.5 |
| OWLv2 | 58.1 | 27.9 | 71.2 | 87.4 | 85.1 |
| Grounding DINO | 54.8 | 21.9 | 68.5 | 77.4 | 77.1 |

SAM proposal recall (IoU ≥ 0.5, before naming; 106 proposals/image): small-vehicle 54.5% (median short side 35 px), large-vehicle 53.3% (median short side 41 px)

## visdrone_ho_D2

| Pipeline | AP50 car | AP50 bus | AP50 truck | agnostic AP50 | found | named |
|---|---|---|---|---|---|---|
| SAM + CLIP | 36.1 | 10.8 | 8.4 | 26.1 | 43.7 | 91.2 |
| SAM + RemoteCLIP | 35.2 | 20.1 | 7.5 | 23.4 | 39.0 | 95.7 |
| OWLv2 | 79.9 | 46.9 | 36.5 | 64.4 | 88.1 | 95.3 |
| Grounding DINO | 76.4 | 25.8 | 22.4 | 67.1 | 90.5 | 93.1 |

SAM proposal recall (IoU ≥ 0.5, before naming; 106 proposals/image): car 54.5% (median short side 35 px), truck 45.9% (median short side 38 px), bus 61.0% (median short side 48 px)

## xview_val

| Pipeline | AP50 car | AP50 bus | AP50 truck | AP50 building | agnostic AP50 | found | named |
|---|---|---|---|---|---|---|---|
| SAM + CLIP | 0.0 | 0.0 | 0.0 | 2.0 | 2.4 | 4.4 | 47.1 |
| SAM + RemoteCLIP | 0.3 | 0.4 | 0.0 | 4.1 | 3.5 | 4.9 | 98.3 |
| OWLv2 | 13.9 | 6.0 | 7.6 | 16.1 | 11.0 | 29.8 | 83.3 |
| Grounding DINO | 6.2 | 0.9 | 1.2 | 10.0 | 9.8 | 35.8 | 76.1 |

SAM proposal recall (IoU ≥ 0.5, before naming; 72 proposals/image): building 55.6% (median short side 36 px), truck 50.7% (median short side 21 px), bus 74.4% (median short side 22 px), car 41.6% (median short side 17 px)

## uavid_tv

| Pipeline | AP50 car | AP50 building | agnostic AP50 | found | named |
|---|---|---|---|---|---|
| SAM + CLIP | 31.6 | 6.4 | 12.7 | 36.8 | 95.6 |
| SAM + RemoteCLIP | 11.7 | 9.6 | 9.9 | 40.0 | 94.6 |
| OWLv2 | 21.6 | 37.4 | 31.0 | 83.8 | 99.4 |
| Grounding DINO | 29.4 | 39.3 | 25.0 | 83.8 | 100.0 |

SAM proposal recall (IoU ≥ 0.5, before naming; 100 proposals/image): car 83.7% (median short side 37 px), building 52.1% (median short side 62 px)

