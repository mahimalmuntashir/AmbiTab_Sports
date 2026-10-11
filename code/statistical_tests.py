from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(values, n_boot: int = 5000, seed: int = 42):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (
        float(x.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def bootstrap_paired_difference(a, b, n_boot: int = 5000, seed: int = 42):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return (
        float(d.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def paired_permutation_p(a, b, n_perm: int = 10000, seed: int = 42):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed = abs(d.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=len(d))
        if abs((d * signs).mean()) >= observed - 1e-15:
            exceed += 1
    return (exceed + 1) / (n_perm + 1)


def paired_cohen_dz(a, b):
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = d.std(ddof=1)
    if sd == 0:
        return 0.0 if abs(d.mean()) < 1e-15 else float("inf")
    return float(d.mean() / sd)


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.zeros(m)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * pvalues[idx]
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return adjusted
