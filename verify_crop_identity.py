"""
Settles whether the replayed crops are the SAME crops, in the SAME row order, as the cached embeddings,
and whether the small re-encoding differences on JPEG sources come from the JPEG decoder.

For every crop list and encoder it replays the extraction, re-encodes every crop and reports:
  - raw per-class crop counts before sampling, compared with the counts in the original August run logs;
  - whether the replayed label sequence equals the cached labels;
  - cosine of each re-encoded crop to its own cached row (exact reproduction = 1.0);
  - how many crops are nearest to their own cached row;
  - the optimal one-to-one assignment between re-encoded crops and cached rows (Hungarian method on
    cosine): if it is the identity permutation, no relabelling of rows fits the cache better;
  - a shuffled-row control: the same statistics after deliberately misaligning rows within each label.

--pillow-dir points at a folder holding a different Pillow build; it is put first on sys.path so the
whole pipeline decodes images with that build. --dll-dir adds the native-library folder a conda-built
Pillow needs. The installed environment is not modified. The original embeddings were made with
Pillow 11.1.0 in a conda environment.

Usage (hf env):
  python verify_crop_identity.py --tag pip_pillow_11.3.0                                   # installed pip wheel (libjpeg-turbo)
  python verify_crop_identity.py --pillow-dir <pip --target dir> --tag pip_pillow_11.1.0   # pip wheel 11.1.0 (libjpeg-turbo)
  python verify_crop_identity.py --pillow-dir <dir holding conda's PIL folder> --dll-dir <conda env>\Libraryin --tag conda_pillow_11.1.0_jpeg9
Read-only with respect to embeddings, checkpoints and existing results. Writes results/paper_facts/crop_identity_<tag>.json
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--pillow-dir", default=None)
ap.add_argument("--dll-dir", default=None, help="folder with the native libraries a conda-built Pillow links against (e.g. <env>\Library\bin)")
ap.add_argument("--tag", required=True)
ap.add_argument("--save-embeddings-dir", default=None)
args = ap.parse_args()
if args.dll_dir:
    _dll_handle = os.add_dll_directory(os.path.abspath(args.dll_dir))  # keep the handle alive for the whole run
if args.pillow_dir:
    sys.path.insert(0, os.path.abspath(args.pillow_dir))  # must precede any PIL / torchvision import

import numpy as np
import PIL
import torch
from PIL import features
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_crop_manifests as bcm
from src.crop_sets import LISTS
from src.embeddings import encode_with_clip, encode_with_remoteclip, load_clip, load_remoteclip

# Per-class raw crop counts printed by the original runs on 20 Aug 2026 (results/paper_facts/original_environment_records.txt)
ORIGINAL_LOG_RAW_COUNTS = {
    "p1_dota": {"large-vehicle": 3033, "small-vehicle": 6537},
    "p1_visdrone_heldout": {"small-vehicle": 22368, "large-vehicle": 7341},
    "p2_visdrone": {"car": 109567, "van": 20534, "truck": 11297, "bus": 5233},
    "p2_xview": {"car": 1269, "truck": 3694, "bus": 903, "van": 25},
    "p3_uavid_building": {"building": 1199},
    "p3_xview_building": {"building": 3711},
}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"tag": args.tag, "pillow_version": PIL.__version__, "pillow_path": os.path.dirname(PIL.__file__),
           "libjpeg_turbo": features.version("libjpeg_turbo") if features.check_feature("libjpeg_turbo") else None,
           "jpeg_library_version": features.version("jpg"), "torch": torch.__version__, "lists": {}}
    bcm.log(f"Pillow {PIL.__version__} (libjpeg-turbo: {out['libjpeg_turbo']}; jpeg library API {out['jpeg_library_version']}) from {out['pillow_path']}")

    raw = {}
    lists = bcm.replay_crop_lists(raw)
    clip_model, clip_processor = load_clip(device)
    rc_model, rc_preprocess = load_remoteclip(bcm.REMOTECLIP_CKPT, device)
    ft_model, _ = load_clip(device)
    ft_model.load_state_dict(torch.load(bcm.FINETUNED_CKPT, map_location=device))
    ft_model.eval()
    rng = np.random.default_rng(0)

    for name, crops in lists.items():
        prefixes, has_labels = LISTS[name]
        labels = np.array([c for _, c in crops])
        entry = {"n": len(crops), "raw_counts_replayed": raw[name], "raw_counts_original_log": ORIGINAL_LOG_RAW_COUNTS[name],
                 "raw_counts_equal_original_log": raw[name] == ORIGINAL_LOG_RAW_COUNTS[name], "encoders": {}}
        for model, prefix in prefixes.items():
            cached = np.load(prefix + "_feats.npy").astype(np.float64)
            if model == "remoteclip":
                new, _ = encode_with_remoteclip(crops, rc_model, rc_preprocess, device, batch_size=4, log=None)
            else:
                new, _ = encode_with_clip(crops, clip_model if model == "clip" else ft_model, clip_processor, device, batch_size=4, log=None)
            if args.save_embeddings_dir:
                os.makedirs(args.save_embeddings_dir, exist_ok=True)
                np.save(os.path.join(args.save_embeddings_dir, f"{name}__{model}__{args.tag}.npy"), new)
            new = new.astype(np.float64)
            sims = new @ cached.T
            own = np.diag(sims)
            nn = sims.argmax(axis=1)
            idx = np.arange(len(crops))
            r, c = linear_sum_assignment(-sims)                    # best one-to-one matching of crops to cached rows
            assign = np.empty(len(crops), int)
            assign[r] = c
            # control: misalign rows within each label (labels still match, crops do not)
            perm = idx.copy()
            for lab in np.unique(labels):
                m = np.where(labels == lab)[0]
                perm[m] = np.roll(m, 1)
            shuffled_own = (new * cached[perm]).sum(axis=1)
            mism = []
            for i in np.where(nn != idx)[0]:
                j = int(nn[i])
                mism.append({"row": int(i), "label": labels[i], "source": crops[i][0].info["source"], "box": list(crops[i][0].info["bbox"]),
                             "cos_to_own_row": float(sims[i, i]), "nearest_row": j, "nearest_row_label": labels[j],
                             "cos_to_nearest_row": float(sims[i, j]), "cos_between_the_two_cached_rows": float(cached[i] @ cached[j]),
                             "own_row_rank": int((sims[i] > sims[i, i]).sum()) + 1, "assigned_row_in_optimal_matching": int(assign[i])})
            entry["encoders"][model] = {
                "labels_equal_cached": bool((np.load(prefix + "_labels.npy", allow_pickle=True) == labels).all()) if has_labels else None,
                "cosine_to_own_row": {"min": float(own.min()), "median": float(np.median(own)), "mean": float(own.mean()),
                                      "share_above_0.999999": float((own > 0.999999).mean()), "share_above_0.9999": float((own > 0.9999).mean())},
                "max_abs_elementwise_difference": float(np.abs(new - cached).max()),
                "nearest_neighbour_is_own_row": int((nn == idx).sum()),
                "optimal_one_to_one_assignment_is_own_row": int((assign == idx).sum()),
                "total_cosine_identity": float(own.sum()), "total_cosine_optimal_assignment": float(sims[r, c].sum()),
                "control_misaligned_rows_cosine": {"median": float(np.median(shuffled_own)), "max": float(shuffled_own.max())},
                "rows_not_nearest_to_own": mism,
            }
            e = entry["encoders"][model]
            bcm.log(f"{name}/{model}: own-cos min {own.min():.6f} median {np.median(own):.6f} | exact(>0.999999) {100*e['cosine_to_own_row']['share_above_0.999999']:.1f}% | "
                    f"NN own {e['nearest_neighbour_is_own_row']}/{len(crops)} | assignment own {e['optimal_one_to_one_assignment_is_own_row']}/{len(crops)} | "
                    f"misaligned-control median {np.median(shuffled_own):.4f}")
        bcm.log(f"{name}: raw counts replayed {raw[name]} | equal to original log: {entry['raw_counts_equal_original_log']}")
        out["lists"][name] = entry

    os.makedirs("results/paper_facts", exist_ok=True)
    path = f"results/paper_facts/crop_identity_{args.tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    bcm.log(f"wrote {path}")


if __name__ == "__main__":
    main()
