from __future__ import annotations

import re
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from statistical_tests import bootstrap_paired_difference, holm_adjust

VALID_ACTIONS = ["ANSWER", "INTERPRET", "CLARIFY"]
EXPECTED_ACTION = {
    "unambiguous": "ANSWER",
    "cooperative_ambiguous": "INTERPRET",
    "uncooperative": "CLARIFY",
}


def parse_action(response: object) -> str:
    text = str(response)
    if text.startswith("ERROR:"):
        return "INVALID"
    match = re.search(
        r"(?im)^\s*Action\s*:\s*(ANSWER|INTERPRET|CLARIFY)\b",
        text,
    )
    return match.group(1).upper() if match else "INVALID"


def attach_gold_actions(raw_df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    gold = benchmark_df[["query_id", "final_query_type"]].copy()
    gold["expected_action"] = gold["final_query_type"].map(EXPECTED_ACTION)
    out = raw_df.merge(gold[["query_id", "expected_action"]], on="query_id", validate="many_to_one")
    out["predicted_action"] = out["raw_response"].map(parse_action)
    out["action_correct"] = (out["predicted_action"] == out["expected_action"]).astype(int)
    return out


def _recall(d: pd.DataFrame, label: str) -> float:
    subset = d[d.expected_action.eq(label)]
    return float((subset.predicted_action == label).mean()) if len(subset) else float("nan")


def summarize_actions(eval_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, d in eval_df.groupby("model"):
        r_answer = _recall(d, "ANSWER")
        r_interpret = _recall(d, "INTERPRET")
        r_clarify = _recall(d, "CLARIFY")
        answerable = d.expected_action.isin(["ANSWER", "INTERPRET"])
        uncoop = d.expected_action.eq("CLARIFY")
        rows.append({
            "model": model,
            "n": len(d),
            "action_accuracy": d.action_correct.mean(),
            "balanced_accuracy": np.mean([r_answer, r_interpret, r_clarify]),
            "macro_f1": f1_score(
                d.expected_action,
                d.predicted_action,
                labels=VALID_ACTIONS,
                average="macro",
                zero_division=0,
            ),
            "recall_answer_U": r_answer,
            "recall_interpret_C": r_interpret,
            "appropriate_clarification_rate_X": r_clarify,
            "over_clarification_rate_UC": (d.loc[answerable, "predicted_action"] == "CLARIFY").mean(),
            "unsafe_answer_or_interpret_rate_X": d.loc[uncoop, "predicted_action"].isin(["ANSWER", "INTERPRET"]).mean(),
            "invalid_action_format_rate": d.predicted_action.eq("INVALID").mean(),
        })
    return pd.DataFrame(rows).sort_values("action_accuracy", ascending=False)


def pairwise_action_accuracy(
    eval_df: pd.DataFrame,
    models: list[str],
    n_boot: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for a_name, b_name in combinations(models, 2):
        a = eval_df[eval_df.model.eq(a_name)][["query_id", "action_correct"]].rename(columns={"action_correct": "a"})
        b = eval_df[eval_df.model.eq(b_name)][["query_id", "action_correct"]].rename(columns={"action_correct": "b"})
        paired = a.merge(b, on="query_id", validate="one_to_one")
        n10 = int(((paired.a == 1) & (paired.b == 0)).sum())
        n01 = int(((paired.a == 0) & (paired.b == 1)).sum())
        discordant = n10 + n01
        p = binomtest(min(n10, n01), n=discordant, p=0.5, alternative="two-sided").pvalue if discordant else 1.0
        diff, lo, hi = bootstrap_paired_difference(paired.a, paired.b, n_boot=n_boot, seed=seed)
        rows.append({
            "model_A": a_name,
            "model_B": b_name,
            "A_minus_B": diff,
            "ci_low": lo,
            "ci_high": hi,
            "mcnemar_n10": n10,
            "mcnemar_n01": n01,
            "mcnemar_exact_p": p,
            "matched_odds_ratio": (n10 + 0.5) / (n01 + 0.5),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_adjust(out["mcnemar_exact_p"].values)
        out["significant_0.05"] = out["p_holm"] < 0.05
    return out


def confusion_table(eval_df: pd.DataFrame) -> pd.DataFrame:
    preds = VALID_ACTIONS + ["INVALID"]
    rows = []
    for model, d in eval_df.groupby("model"):
        for expected in VALID_ACTIONS:
            subset = d[d.expected_action.eq(expected)]
            for predicted in preds:
                count = int((subset.predicted_action == predicted).sum())
                rows.append({
                    "model": model,
                    "expected_action": expected,
                    "predicted_action": predicted,
                    "count": count,
                    "row_proportion": count / len(subset) if len(subset) else np.nan,
                })
    return pd.DataFrame(rows)
