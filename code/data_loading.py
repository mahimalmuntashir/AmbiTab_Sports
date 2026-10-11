from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

EXPECTED_TRACKS = {
    "unambiguous": 120,
    "cooperative_ambiguous": 120,
    "uncooperative": 120,
}


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV and normalize column names."""
    df = pd.read_csv(path, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_benchmark(path: str | Path, validate: bool = True) -> pd.DataFrame:
    """Load the AmbiTab-Sports benchmark."""
    df = load_csv(path)
    required = {
        "query_id",
        "query_text",
        "final_query_type",
        "subdomain",
        "gold_answer_1",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Benchmark missing required columns: {sorted(missing)}")

    if validate:
        if df["query_id"].duplicated().any():
            raise ValueError("query_id values must be unique in the benchmark.")
        if len(df) != 360:
            raise ValueError(f"Expected 360 benchmark rows, found {len(df)}.")
        counts = df["final_query_type"].value_counts().to_dict()
        for label, expected_n in EXPECTED_TRACKS.items():
            if counts.get(label, 0) != expected_n:
                raise ValueError(
                    f"Expected {expected_n} rows for {label!r}, found {counts.get(label, 0)}."
                )
    return df


def split_tracks(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Return copies of the three benchmark tracks."""
    return {
        label: df[df["final_query_type"].eq(label)].copy().reset_index(drop=True)
        for label in EXPECTED_TRACKS
    }


def load_cooperative_human_gold(path: str | Path, validate: bool = True) -> pd.DataFrame:
    """Load supplied 480-row cooperative rating records; provenance is unverified."""
    df = load_csv(path)
    required = {
        "query_id",
        "model",
        "final_interpretation_validity",
        "final_assumption_quality",
        "final_answer_usefulness",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Human cooperative gold missing: {sorted(missing)}")

    if "human_total" not in df.columns:
        df["human_total"] = (
            df["final_interpretation_validity"].astype(float)
            + df["final_assumption_quality"].astype(float)
            + df["final_answer_usefulness"].astype(float)
        )
    if "human_score" not in df.columns:
        df["human_score"] = df["human_total"] / 6.0

    if validate:
        if len(df) != 480:
            raise ValueError(f"Expected 480 cooperative response rows, found {len(df)}.")
        for col in [
            "final_interpretation_validity",
            "final_assumption_quality",
            "final_answer_usefulness",
        ]:
            if not df[col].between(0, 2).all():
                raise ValueError(f"Column {col} contains values outside 0-2.")
    return df


def load_execution_human_gold(path: str | Path, validate: bool = True) -> pd.DataFrame:
    """Load supplied Step-1 execution labels; human provenance is unverified."""
    df = load_csv(path)
    # Public ratings are blinded; model identity is recovered from archived
    # response content by associate_execution_labels, not a private ID key.
    required = {"query_id", "raw_response", "human_correct"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Execution human gold missing: {sorted(missing)}")

    if validate:
        if len(df) != 360:
            raise ValueError(f"Expected 360 execution human-gold rows, found {len(df)}.")
        if not pd.to_numeric(df["human_correct"], errors="coerce").isin([0, 1]).all():
            raise ValueError("human_correct must contain only 0/1 labels.")
        if "blind_response_id" in df and df.blind_response_id.duplicated().any():
            raise ValueError("Blinded response IDs must be unique.")
    return df


def associate_execution_labels(raw_df: pd.DataFrame, gold_df: pd.DataFrame, models: list[str]):
    """Match original outputs to recorded binary labels without exposing ID keys.

    One archived literal 'None' output was exported as an empty rating stimulus.
    Match it as empty but preserve the original text and report the normalization.
    Identical model responses can be matched only when recorded labels agree.
    """
    raw = raw_df[raw_df.model.isin(models)].copy()
    labels = gold_df[["query_id", "raw_response", "human_correct"]].copy()
    raw["match_response"] = raw.raw_response.replace({"None": ""})
    labels["match_response"] = labels.raw_response.replace({"None": ""})
    labels["human_correct"] = pd.to_numeric(labels.human_correct, errors="raise").astype(int)
    if labels.groupby(["query_id", "match_response"]).human_correct.nunique().gt(1).any():
        raise ValueError("Conflicting labels for identical response content require a private-key or author resolution.")
    labels = labels[["query_id", "match_response", "human_correct"]].drop_duplicates()
    result = raw.merge(labels, on=["query_id", "match_response"], how="left", validate="many_to_one")
    if result.human_correct.isna().any():
        raise ValueError(f"Unmatched execution responses: {int(result.human_correct.isna().sum())}")
    if result.duplicated(["model", "query_id"]).any():
        raise ValueError("Duplicate model/query execution outputs.")
    result["human_correct"] = result.human_correct.astype(int)
    result["response_export_normalized"] = result.raw_response.eq("None")
    return result.drop(columns="match_response")


def find_first(root: str | Path, names: Iterable[str]) -> Optional[Path]:
    """Find the first named file recursively under root."""
    root = Path(root)
    for name in names:
        hits = list(root.rglob(name))
        if hits:
            return hits[0]
    return None
