from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from statistical_tests import (
    bootstrap_mean_ci,
    bootstrap_paired_difference,
    holm_adjust,
    paired_cohen_dz,
    paired_permutation_p,
)


def ensure_human_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = [
        "final_interpretation_validity",
        "final_assumption_quality",
        "final_answer_usefulness",
    ]
    for col in required:
        if col not in out.columns:
            raise KeyError(f"Missing cooperative human field: {col}")
    if "human_total" not in out.columns:
        out["human_total"] = sum(out[c].astype(float) for c in required)
    if "human_score" not in out.columns:
        out["human_score"] = out["human_total"] / 6.0
    return out


def summarize_cooperative(df: pd.DataFrame, n_boot: int = 5000, seed: int = 42) -> pd.DataFrame:
    df = ensure_human_score(df)
    rows = []
    for model, d in df.groupby("model"):
        mean, lo, hi = bootstrap_mean_ci(d["human_score"], n_boot=n_boot, seed=seed)
        rows.append({
            "model": model,
            "n": len(d),
            "human_cooperative_score": mean,
            "ci_low": lo,
            "ci_high": hi,
            "interpretation_validity_norm": d["final_interpretation_validity"].mean() / 2.0,
            "assumption_quality_norm": d["final_assumption_quality"].mean() / 2.0,
            "answer_usefulness_norm": d["final_answer_usefulness"].mean() / 2.0,
        })
    return pd.DataFrame(rows).sort_values("human_cooperative_score", ascending=False)


def pairwise_cooperative(
    df: pd.DataFrame,
    models: list[str],
    n_boot: int = 5000,
    n_perm: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    df = ensure_human_score(df)
    rows = []
    for a_name, b_name in combinations(models, 2):
        a = df[df.model.eq(a_name)][["query_id", "human_score"]].rename(columns={"human_score": "a"})
        b = df[df.model.eq(b_name)][["query_id", "human_score"]].rename(columns={"human_score": "b"})
        paired = a.merge(b, on="query_id", validate="one_to_one")
        diff, lo, hi = bootstrap_paired_difference(paired.a, paired.b, n_boot=n_boot, seed=seed)
        rows.append({
            "model_A": a_name,
            "model_B": b_name,
            "A_minus_B": diff,
            "ci_low": lo,
            "ci_high": hi,
            "permutation_p": paired_permutation_p(paired.a, paired.b, n_perm=n_perm, seed=seed),
            "paired_dz": paired_cohen_dz(paired.a, paired.b),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_adjust(out["permutation_p"].values)
        out["significant_0.05"] = out["p_holm"] < 0.05
    return out
