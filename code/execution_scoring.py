from __future__ import annotations

import re
import unicodedata
from typing import Dict

import numpy as np
import pandas as pd


def normalize_text(text: object) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9%+\-.\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def parse_gold_numeric(gold: object):
    text = str(gold).strip()
    match = re.match(
        r"^\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(%)?",
        text,
    )
    if not match:
        return None
    token = match.group(1)
    value = float(token.replace(",", ""))
    decimals = len(token.split(".")[1]) if "." in token else 0
    is_percentage = match.group(2) == "%" or "%" in text
    tolerance = 0.0 if (not is_percentage and decimals == 0) else 0.5 * (10 ** -decimals)
    return {
        "value": value,
        "is_percentage": is_percentage,
        "decimals": decimals,
        "tolerance": tolerance,
    }


def extract_numbers(text: object):
    pattern = r"(?<![A-Za-z0-9])([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(%)?"
    out = []
    for m in re.finditer(pattern, str(text)):
        out.append({
            "value": float(m.group(1).replace(",", "")),
            "is_percentage": m.group(2) == "%",
        })
    return out


def numeric_match(response: object, gold: object) -> bool:
    gold_info = parse_gold_numeric(gold)
    if gold_info is None:
        return False
    for candidate in extract_numbers(response):
        if gold_info["is_percentage"] and not candidate["is_percentage"]:
            continue
        if abs(candidate["value"] - gold_info["value"]) <= gold_info["tolerance"] + 1e-12:
            return True
    return False


def entity_match(response: object, gold: object) -> bool:
    r = normalize_text(response)
    g = normalize_text(gold)
    if not g:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(g) + r"(?![a-z0-9])", r) is not None


def identify_task_type(query: object, gold_answer: object) -> str:
    q = str(query).strip().lower()
    if q.startswith("list"):
        return "list"
    if q.startswith("compare"):
        return "compare"
    if parse_gold_numeric(gold_answer) is not None:
        return "numeric"
    return "entity"


def score_execution_row(row: pd.Series) -> pd.Series:
    response = str(row["raw_response"])
    query = row["query_text"]
    gold1 = row["gold_answer_1"]
    gold2 = row.get("gold_answer_2", np.nan)
    task_type = identify_task_type(query, gold1)

    if response.startswith("ERROR:"):
        return pd.Series({
            "task_type": task_type,
            "task_aware_correct": 0,
            "match_reason": "generation_error",
        })

    if task_type == "numeric":
        correct = numeric_match(response, gold1)
        reason = "numeric_match" if correct else "numeric_wrong"
    elif task_type == "entity":
        correct = entity_match(response, gold1)
        reason = "entity_match" if correct else "entity_wrong"
    elif task_type == "compare":
        correct = numeric_match(response, gold1) and numeric_match(response, gold2)
        reason = "comparison_match" if correct else "comparison_wrong"
    else:
        items = [x.strip() for x in str(gold1).split(";") if x.strip()]
        matches = [entity_match(response, item) for item in items]
        correct = bool(items) and all(matches)
        reason = "full_list_match" if correct else "list_incomplete"

    return pd.Series({
        "task_type": task_type,
        "task_aware_correct": int(correct),
        "match_reason": reason,
    })


def score_execution_dataframe(raw_df: pd.DataFrame, track_u: pd.DataFrame) -> pd.DataFrame:
    """Attach benchmark metadata if needed and score all Step-1 rows."""
    lookup_cols = [
        c for c in [
            "query_id", "query_text", "subdomain",
            "gold_answer_1", "gold_answer_2", "gold_answer_3",
        ] if c in track_u.columns
    ]
    base = raw_df.copy()
    for col in ["query_text", "subdomain", "gold_answer_1", "gold_answer_2", "gold_answer_3"]:
        if col in base.columns:
            base = base.drop(columns=[col])
    base = base.merge(
        track_u[lookup_cols].drop_duplicates("query_id"),
        on="query_id",
        how="left",
        validate="many_to_one",
    )
    scores = base.apply(score_execution_row, axis=1)
    return pd.concat([base, scores], axis=1)
