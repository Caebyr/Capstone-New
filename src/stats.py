"""Centroid drift measures and cluster (image / sequence) resampling shared by drift_stats.py and
classify_crops.py.

Every statistic takes per-crop weights. Weight 1 everywhere reproduces the plain estimate; a cluster
bootstrap draws whole images (or sequences) with replacement and gives each crop the number of times
its cluster was drawn, so correlated crops from one image always move together.
"""
import numpy as np


def centroid(feats, w):
    """Arithmetic (not re-normalized) weighted mean of unit-norm embeddings."""
    return (w[:, None] * feats).sum(0) / w.sum()


def radius(feats, w, mu):
    """Mean (not RMS) distance from each crop to its own domain centroid, Eq. (1)."""
    return float((w * np.linalg.norm(feats - mu, axis=1)).sum() / w.sum())


def cos_sim(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def drift(fs, ws, fu, wu):
    """L2 / cosine centroid displacement, both radii, spread-normalized ratio (Eqs. 1-3)."""
    ms, mu = centroid(fs, ws), centroid(fu, wu)
    rs, ru = radius(fs, ws, ms), radius(fu, wu, mu)
    l2 = float(np.linalg.norm(ms - mu))
    return {"l2": l2, "cos": 1 - cos_sim(ms, mu), "r_overhead": rs, "r_uav": ru,
            "rad": (rs + ru) / 2, "ratio": l2 / ((rs + ru) / 2)}


def spacing_ratio(cents, cos_drift, cats):
    """Y_c of Eq. (4): cosine drift over the pooled mean between-category cosine dissimilarity.
    cents[domain][category] -> centroid."""
    out = {}
    for c in cats:
        b = np.mean([1 - cos_sim(cents[d][c], cents[d][k]) for d in ("overhead", "uav") for k in cats if k != c])
        out[c] = {"B": float(b), "Y": cos_drift[c] / float(b)}
    return out


def cluster_weights(rng, cluster_ids):
    """One bootstrap replicate: draw len(unique clusters) clusters with replacement; crop weight =
    how many times its cluster was drawn."""
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    counts = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
    return counts[inv].astype(np.float64)


def ci(values, level=0.95):
    v = np.asarray([x for x in values if np.isfinite(x)])
    lo, hi = np.quantile(v, [(1 - level) / 2, 1 - (1 - level) / 2])
    return [float(lo), float(hi)]
