# Zero-shot open-vocabulary detection: overhead vs UAV

AP50 / AP: COCO, class-aware. Found: share of GT boxes overlapped (IoU ≥ 0.5) by any detection with score ≥ 0.1. Named: of found boxes, share whose best detection has the right class. Brackets: 95% image-bootstrap CI (500 reps). O = overhead, U = UAV.

## OWLv2 B/16 (ensemble)

| Task | Category | Sets (O / U) | AP50 O / U | AP O / U | Found O [CI] | Found U [CI] | Named O [CI] | Named U [CI] |
|---|---|---|---|---|---|---|---|---|
| D1 | small-vehicle | dota_tiles / visdrone_ho_D1 | 36.9 / 58.6 | 18.8 / 34.0 | 60.3 [50.5, 71.5] | 87.7 [86.7, 88.5] | 97.8 [96.7, 98.7] | 98.4 [98.2, 98.7] |
| D1 | large-vehicle | dota_tiles / visdrone_ho_D1 | 20.6 / 27.1 | 10.6 / 15.6 | 54.6 [44.7, 65.7] | 85.1 [83.8, 86.4] | 73.0 [61.0, 84.0] | 46.3 [44.1, 48.8] |
| D2 | car | xview_val / visdrone_ho_D2 | 10.3 / 78.5 | 3.3 / 47.5 | 60.5 [56.2, 64.4] | 88.7 [87.9, 89.5] | 92.4 [90.0, 94.4] | 98.5 [98.2, 98.7] |
| D2 | bus | xview_val / visdrone_ho_D2 | 6.5 / 52.7 | 2.9 / 34.9 | 66.1 [57.2, 75.5] | 86.9 [84.0, 89.3] | 46.2 [36.1, 55.7] | 75.2 [71.2, 78.6] |
| D2 | truck | xview_val / visdrone_ho_D2 | 8.1 / 38.8 | 3.7 / 22.7 | 62.8 [58.8, 66.1] | 81.2 [79.0, 83.2] | 57.3 [52.5, 62.2] | 74.7 [71.8, 77.5] |
| D2 | building | xview_val / uavid_tv | 18.8 / 33.3 | 8.4 / 17.9 | 28.7 [27.1, 30.4] | 85.9 [83.6, 88.3] | 89.6 [88.2, 90.8] | 100.0 [100.0, 100.0] |

Per set: dota_tiles mean AP50 28.8, class-agnostic AP50 38.4; uavid_tv mean AP50 19.1, class-agnostic AP50 22.6; visdrone_ho_D1 mean AP50 42.9, class-agnostic AP50 70.5; visdrone_ho_D2 mean AP50 56.7, class-agnostic AP50 61.8; xview_val mean AP50 11.0, class-agnostic AP50 12.4

## Grounding DINO tiny

| Task | Category | Sets (O / U) | AP50 O / U | AP O / U | Found O [CI] | Found U [CI] | Named O [CI] | Named U [CI] |
|---|---|---|---|---|---|---|---|---|
| D1 | small-vehicle | dota_tiles / visdrone_ho_D1 | 18.4 / 55.8 | 8.5 / 35.1 | 48.4 [40.4, 58.8] | 76.4 [74.9, 77.9] | 87.9 [83.8, 91.8] | 88.8 [87.4, 90.2] |
| D1 | large-vehicle | dota_tiles / visdrone_ho_D1 | 1.4 / 23.0 | 0.7 / 16.2 | 28.1 [22.0, 36.1] | 75.8 [74.0, 77.6] | 52.3 [39.7, 64.2] | 52.8 [51.1, 54.8] |
| D2 | car | xview_val / visdrone_ho_D2 | 5.7 / 75.5 | 2.1 / 51.4 | 62.2 [58.3, 65.9] | 91.3 [90.6, 92.0] | 96.1 [94.7, 97.4] | 97.6 [97.2, 98.0] |
| D2 | bus | xview_val / visdrone_ho_D2 | 0.9 / 24.5 | 0.4 / 19.6 | 58.1 [45.3, 70.3] | 85.2 [82.1, 88.2] | 16.3 [11.5, 22.4] | 40.3 [35.8, 44.7] |
| D2 | truck | xview_val / visdrone_ho_D2 | 1.0 / 27.8 | 0.5 / 19.3 | 51.9 [48.4, 56.0] | 82.6 [80.2, 84.9] | 23.1 [20.1, 26.3] | 66.7 [63.6, 69.7] |
| D2 | building | xview_val / uavid_tv | 10.8 / 35.0 | 5.1 / 19.2 | 35.2 [33.2, 37.1] | 88.5 [86.2, 90.7] | 81.4 [80.1, 82.7] | 99.9 [99.6, 100.0] |

Per set: dota_tiles mean AP50 9.9, class-agnostic AP50 14.4; uavid_tv mean AP50 20.9, class-agnostic AP50 15.4; visdrone_ho_D1 mean AP50 39.4, class-agnostic AP50 66.3; visdrone_ho_D2 mean AP50 42.6, class-agnostic AP50 66.3; xview_val mean AP50 4.6, class-agnostic AP50 10.4

