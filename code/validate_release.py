"""Read-only technical validation; this does not establish human or factual validity."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import warnings
from zipfile import ZipFile
import numpy as np
import openpyxl
import pandas as pd
from PIL import Image
import fitz
from data_loading import load_csv, load_benchmark, load_execution_human_gold, associate_execution_labels

MODELS = ["Qwen2.5-7B-Instruct", "Gemma-4-E4B-it", "Mistral-7B-Instruct-v0.2"]

def check(condition, explanation):
    if not condition:
        raise ValueError(explanation)

def sheet_frame(path, sheet, header=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(book[sheet].values)
    book.close()
    return pd.DataFrame(rows[header+1:], columns=rows[header]).fillna("")

def validate(root, compare=None):
    root = Path(root).resolve()
    checked = []
    bench = load_benchmark(root / "benchmark/AmbiTab_Sports_Full_Gold_Benchmark_FINAL.csv")
    check(bench.shape == (360, 63), "Benchmark size/field count differs from release.")
    check(bench.groupby(["subdomain", "final_query_type"]).size().eq(20).all(), "Track/subdomain balance changed.")
    check(bench.benchmark_split.value_counts().to_dict() == {"train": 252, "dev": 54, "test": 54}, "Historical split changed.")
    supplied = load_csv(root / "benchmark/archive/AmbiTab_Sports_Full_Gold_Benchmark_SUPPLIED.csv")
    for col in supplied:
        target = "supplied_verification_status" if col == "verification_status" else col
        check(supplied[col].equals(bench[target]), f"Supplied scientific field changed: {col}")
    final = sheet_frame(root / "benchmark/annotation/AmbiTab_Sports_Final_Construction_Adjudication.xlsx", "Final_Construction_Record")
    pd.testing.assert_frame_equal(final, bench, check_dtype=False)
    dictionary = sheet_frame(root / "benchmark/AmbiTab_Sports_Benchmark_Data_Dictionary.xlsx", "Data Dictionary")
    check(set(dictionary.iloc[:, 0]) == set(bench.columns), "Dictionary does not cover all fields.")
    audit = load_csv(root / "benchmark/execution_gold_source_audit.csv")
    u = bench[bench.final_query_type.eq("unambiguous")]
    check(len(audit) == 120 and set(audit.query_id) == set(u.query_id), "Source audit coverage incomplete.")
    check(set(audit.loc[audit.verification_status.eq("source_review_required"), "query_id"]) ==
          {"TN-U019", "F1-U010", "F1-U008", "F1-U007", "TN-U004", "TN-U008"}, "Source flags changed.")
    check(set(u.verification_status) == {"source_review_required", "supplied_claim_not_independently_verified"}, "Unsupported source certification.")
    checked.append("360-query release, supplied field preservation, workbook/dictionary, 120-source audit")

    raw = load_csv(root / "raw_outputs/step1/step1_raw_outputs_execution_v7.csv")
    gold = load_execution_human_gold(root / "human_evaluation/execution/step1_execution_human_gold_final.csv")
    ex = associate_execution_labels(raw, gold, MODELS)
    counts = ex.groupby("model").human_correct.sum().to_dict()
    check(counts == dict(zip(MODELS, [22, 43, 19])), "Primary correctness totals changed.")
    check(len(ex) == 360 and ex.groupby("model").size().eq(120).all(), "Execution coverage incomplete.")
    check(int(ex.response_export_normalized.sum()) == 1, "Response-export normalization changed.")
    checkpoint = json.loads((root / "checkpoints/step1_execution_fresh_v7.json").read_text())
    for model in MODELS:
        check(checkpoint[model] == raw.loc[raw.model.eq(model), "raw_response"].tolist(), "Execution checkpoint differs from raw output.")
    primary = load_csv(root / "results/step1_execution/step1_execution_human_results_with_ci.csv").set_index("model")
    master = load_csv(root / "results/FINAL_master_results_steps1_3.csv").set_index("model")
    for model in MODELS:
        check(primary.loc[model, "correct"] == counts[model], "Primary summary contains secondary scores.")
        check(np.isclose(master.loc[model, "execution_accuracy"], counts[model]/120), "Master contains stale execution scores.")
        for src, dst in [("ci_low", "execution_ci_low"), ("ci_high", "execution_ci_high")]:
            check(np.isclose(primary.loc[model, src], master.loc[model, dst]), "Master CI mismatch.")
    ca = load_csv(root / "human_evaluation/execution/step1_execution_annotator_A_FILLED.csv")
    cb = load_csv(root / "human_evaluation/execution/step1_execution_annotator_B_FILLED.csv")
    ab = ca.merge(cb, on="blind_response_id", suffixes=("_a", "_b"), validate="one_to_one")
    check(len(ab) == 360 and ab.human_correct_a.eq(ab.human_correct_b).all(), "Execution agreement differs from records.")
    check(ab.raw_response_a.ne(ab.raw_response_b).sum() == 8, "Export discrepancy count changed.")
    checked.append("Blinded response association, raw/checkpoint equality, primary master, eight export discrepancies")

    coop = load_csv(root / "human_evaluation/cooperative/cooperative_human_gold_final_480.csv")
    pd.testing.assert_frame_equal(coop, load_csv(root / "benchmark/cooperative_human_gold_final_480.csv"))
    check(len(coop) == 480 and not coop.duplicated(["query_id", "model"]).any(), "Cooperative coverage/duplicates.")
    check(coop.groupby("model").size().eq(120).all(), "Cooperative per-model coverage.")
    cols = ["final_interpretation_validity", "final_assumption_quality", "final_answer_usefulness"]
    check(all(coop[c].isin([0, 1, 2]).all() for c in cols), "Cooperative score outside rubric.")
    check(np.allclose(coop[cols].sum(axis=1), coop.human_total), "Cooperative total mismatch.")
    check(np.allclose(coop.human_total / 6, coop.human_score), "Cooperative normalization mismatch.")
    check(coop.score_source.value_counts().to_dict() == {"adjudicated": 275, "annotator_agreement": 205}, "Score-source counts changed.")
    action = load_csv(root / "results/step3_action_selection/step3_all360_unified_action_classification.csv")
    check(len(action) == 1440 and action.groupby("model").size().eq(360).all(), "Action coverage incomplete.")
    check(not action.duplicated(["model", "query_id"]).any(), "Duplicate action output.")
    check(set(action.predicted_action) == {"ANSWER", "INTERPRET", "CLARIFY"}, "Action parsing includes invalid labels.")
    baseline = action[action.model.eq("Always-Clarify-Baseline")]
    check(len(baseline) == 360 and baseline.predicted_action.eq("CLARIFY").all(), "Baseline changed.")
    check(np.isclose(baseline.action_correct.mean(), 1/3), "Constant-clarify penalty missing.")
    metrics = load_csv(root / "results/step3_action_selection/step3_primary_unified_action_metrics.csv").set_index("model")
    check(np.isclose(metrics.loc["Always-Clarify-Baseline", "macro_f1"], 1/6), "Baseline macro-F1 changed.")
    cp = json.loads((root / "checkpoints/step3_unified_action360_v7.json").read_text())
    actionraw = load_csv(root / "raw_outputs/step3/step3_raw_outputs_unified_action360_v7.csv")
    for model in MODELS:
        check(cp[model] == actionraw.loc[actionraw.model.eq(model), "raw_response"].tolist(), "Action checkpoint differs from raw output.")
    check(load_csv(root / "results/step2_cooperative/step2_cooperative_human_pairwise_inference.csv").p_holm.ge(.05).all(), "Cooperative significance claim changed.")
    check(load_csv(root / "results/step3_action_selection/step3_unified_action_pairwise_inference.csv").p_holm.ge(.05).all(), "Action significance claim changed.")
    checked.append("480 cooperative records, 1440 action records, constant-clarify penalty, paired non-significance")

    for path in root.rglob("*.ipynb"):
        notebook = json.loads(path.read_text())
        check(set(notebook["metadata"]) <= {"kernelspec", "language_info"}, f"Extra notebook metadata: {path.name}")
        for cell in notebook["cells"]:
            check(not cell.get("metadata"), f"Notebook cell metadata retained: {path.name}")
            if cell["cell_type"] == "code":
                check(cell.get("execution_count") is None and not cell.get("outputs"), f"Notebook outputs retained: {path.name}")
    for path in (root / "code").glob("*.py"):
        ast.parse(path.read_text(), filename=path.name)
    for path in root.rglob("*.xlsx"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            book = openpyxl.load_workbook(path, read_only=True)
        # Some SDK exports omit core properties. openpyxl then supplies its own
        # in-memory default creator; that default is not file metadata.
        with ZipFile(path) as archive:
            has_core_properties = "docProps/core.xml" in archive.namelist()
        if has_core_properties:
            check(book.properties.creator == "Anonymous" and book.properties.lastModifiedBy == "Anonymous", f"Workbook identity metadata: {path.name}")
        else:
            check(book.properties.lastModifiedBy is None, f"Unexpected missing-property identity: {path.name}")
        book.close()
    for path in root.rglob("*.pdf"):
        with fitz.open(path) as pdf:
            check(len(pdf) > 0 and not pdf.is_encrypted, f"Invalid PDF: {path.name}")
            check(pdf.metadata.get("author", "") in {"", "Anonymous"}, f"PDF author metadata: {path.name}")
    for path in root.rglob("*.png"):
        with Image.open(path) as image:
            check(not image.info, f"PNG metadata retained: {path.name}")
    for path in (root / "figures").glob("*.pdf"):
        check(path.read_bytes() == (root / "paper" / path.name).read_bytes(), f"Paper figure differs: {path.name}")
    for filename in ["ambitab_sports_revised.pdf", "response_to_reviewers.pdf", "revision_cover_letter.pdf"]:
        check((root / "paper" / filename).is_file(), f"Missing compiled submission document: {filename}")
    note = json.loads((root / "analysis_run.json").read_text())
    check(not note["independent_human_review_verified"] and not note["factual_gold_independently_verified"] and
          not note["fresh_model_inference_run"], "Unsupported validation/inference certification.")
    checked.append("Clean notebooks, Python syntax, anonymous file metadata, six identical paper figures, three compiled documents")

    compared = 0
    if compare is not None:
        compare = Path(compare)
        if not compare.is_absolute():
            compare = root / compare
        for reference in sorted((root / "results").rglob("*.csv")):
            output = compare / reference.relative_to(root)
            check(output.is_file(), f"Missing reproduced table: {output.name}")
            pd.testing.assert_frame_equal(load_csv(reference), load_csv(output), check_dtype=False, atol=1e-12, rtol=1e-12)
            compared += 1
        for reference in sorted((root / "figures").glob("Fig*.*")):
            output = compare / reference.relative_to(root)
            check(output.is_file(), f"Missing reproduced figure: {output.name}")
            check(hashlib.sha256(output.read_bytes()).digest() == hashlib.sha256(reference.read_bytes()).digest(),
                  f"Reproduced figure bytes differ: {output.name}; check pinned dependencies/fonts.")
        check(json.loads((compare / "analysis_run.json").read_text()) == note, "Reproduction settings/counts/environment differ.")
        checked.append(f"Fresh reproduction: {compared} tables and 12 figure files match")
    repository_url = re.search(r"\\newcommand\*?\{\\RepositoryURL\}\{([^}]*)\}", (root / "paper/submission_settings.tex").read_text())
    return {"technical_checks_passed": True, "checks": checked, "reproduced_tables_compared": compared,
            "independent_human_validation_completed": False, "independent_factual_validation_completed": False,
            "source_review_flags": 6, "historical_checkpoint_shas_recovered": False,
            "public_repository_url_added": bool(repository_url and repository_url.group(1).strip())}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()
    try:
        report = validate(args.root, args.compare)
    except (AssertionError, ValueError, KeyError, FileNotFoundError) as exc:
        print(json.dumps({"technical_checks_passed": False, "error": str(exc)}, indent=2))
        raise SystemExit(1)
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
