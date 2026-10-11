"""CPU-only reproduction from archived evidence; no fresh ratings or inference."""
from __future__ import annotations
import argparse
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score
from data_loading import load_benchmark, load_csv, load_execution_human_gold, associate_execution_labels
from execution_scoring import score_execution_dataframe
from cooperative_scoring import summarize_cooperative, pairwise_cooperative
from action_selection_scoring import attach_gold_actions, summarize_actions, pairwise_action_accuracy, confusion_table
from statistical_tests import bootstrap_mean_ci, bootstrap_paired_difference, holm_adjust

MODELS = ["Qwen2.5-7B-Instruct", "Gemma-4-E4B-it", "Mistral-7B-Instruct-v0.2"]

def save(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.15g", lineterminator="\n")

def execution_summary(df):
    rows = []
    for model, d in df.groupby("model", sort=False):
        mean, lo, hi = bootstrap_mean_ci(d.human_correct)
        rows.append(dict(model=model, n=len(d), correct=int(d.human_correct.sum()), accuracy=mean, ci_low=lo, ci_high=hi))
    return pd.DataFrame(rows).sort_values("accuracy", ascending=False)

def execution_pairs(df):
    rows = []
    for a, b in combinations(MODELS, 2):
        z = df[df.model.eq(a)][["query_id", "human_correct"]].merge(
            df[df.model.eq(b)][["query_id", "human_correct"]], on="query_id", suffixes=("_a", "_b"), validate="one_to_one")
        n10 = int(((z.human_correct_a == 1) & (z.human_correct_b == 0)).sum())
        n01 = int(((z.human_correct_a == 0) & (z.human_correct_b == 1)).sum())
        diff, lo, hi = bootstrap_paired_difference(z.human_correct_a, z.human_correct_b)
        p = binomtest(min(n10, n01), n10+n01, .5).pvalue if n10+n01 else 1.0
        rows.append(dict(model_A=a, model_B=b, A_minus_B=diff, ci_low=lo, ci_high=hi,
                         mcnemar_n10=n10, mcnemar_n01=n01, mcnemar_exact_p=p,
                         matched_odds_ratio=(n10+.5)/(n01+.5)))
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out.mcnemar_exact_p)
    out["significant_0.05"] = out.p_holm < .05
    return out

def run(root: Path, output: Path, figures: bool = True):
    root, output = root.resolve(), output.resolve()
    r = output / "results"
    bench = load_benchmark(root / "benchmark/AmbiTab_Sports_Full_Gold_Benchmark_FINAL.csv")
    assert bench.groupby(["subdomain", "final_query_type"]).size().eq(20).all()
    raw = load_csv(root / "raw_outputs/step1/step1_raw_outputs_execution_v7.csv")
    gold = load_execution_human_gold(root / "human_evaluation/execution/step1_execution_human_gold_final.csv")
    for field in ["query_text", "gold_answer_1", "gold_answer_2", "gold_answer_3"]:
        if not gold[field].astype(str).eq(gold.query_id.map(bench.set_index("query_id")[field]).astype(str)).all():
            raise ValueError(f"Released benchmark differs from execution rating stimulus: {field}")
    ex = associate_execution_labels(raw, gold, MODELS)
    assert len(ex) == 360 and ex.groupby("model").size().eq(120).all()
    assert set(ex.query_id) == set(bench.loc[bench.final_query_type.eq("unambiguous"), "query_id"])
    primary = execution_summary(ex)
    pairs = execution_pairs(ex)
    save(ex, r / "step1_execution/step1_execution_recorded_scored_outputs.csv")
    save(primary, r / "step1_execution/step1_execution_human_results_with_ci.csv")
    save(pairs, r / "step1_execution/step1_execution_human_pairwise_inference.csv")
    a = load_csv(root / "human_evaluation/execution/step1_execution_annotator_A_FILLED.csv")
    b = load_csv(root / "human_evaluation/execution/step1_execution_annotator_B_FILLED.csv")
    ab = a.merge(b, on="blind_response_id", suffixes=("_a", "_b"), validate="one_to_one")
    agreement = pd.DataFrame([dict(n=len(ab), binary_exact_agreement=ab.human_correct_a.eq(ab.human_correct_b).mean(),
                                  response_export_discrepancies=int(ab.raw_response_a.ne(ab.raw_response_b).sum()),
                                  provenance_status="independent_human_review_not_verified")])
    save(agreement, r / "step1_execution/step1_execution_human_agreement.csv")
    save(ab.loc[ab.raw_response_a.ne(ab.raw_response_b), ["query_id_a", "raw_response_a", "raw_response_b", "human_correct_a", "human_correct_b"]],
         r / "step1_execution/step1_response_export_discrepancies.csv")
    automatic = score_execution_dataframe(raw, bench[bench.final_query_type.eq("unambiguous")])
    save(automatic, r / "step1_execution/step1_execution_scored_automatic_secondary.csv")
    auto_summary = execution_summary(automatic.rename(columns={"task_aware_correct": "human_correct"}))
    auto_summary = auto_summary.rename(columns={"correct": "correct_n"})
    save(auto_summary, r / "step1_execution/step1_execution_results_automatic_secondary.csv")
    save(execution_pairs(automatic[automatic.model.isin(MODELS)].rename(columns={"task_aware_correct": "human_correct"})),
         r / "step1_execution/step1_execution_pairwise_automatic_secondary.csv")
    save(automatic[automatic.task_aware_correct.eq(1)], r / "step1_execution/step1_scoring_manual_audit_priority.csv")

    coop = load_csv(root / "human_evaluation/cooperative/cooperative_human_gold_final_480.csv")
    assert len(coop) == 480 and not coop.duplicated(["model", "query_id"]).any()
    assert set(coop.model) == set(MODELS + ["Heuristic-Baseline"])
    assert coop.groupby("model").size().eq(120).all()
    assert set(coop.query_id) == set(bench.loc[bench.final_query_type.eq("cooperative_ambiguous"), "query_id"])
    scorecols = ["final_interpretation_validity", "final_assumption_quality", "final_answer_usefulness"]
    assert all(coop[c].between(0, 2).all() for c in scorecols)
    assert np.allclose(coop[scorecols].sum(axis=1), coop.human_total)
    assert np.allclose(coop.human_total/6, coop.human_score)
    for field in ["query_text", "subdomain", "final_primary_ambiguity_source"]:
        assert coop[field].eq(coop.query_id.map(bench.set_index("query_id")[field])).all()
    cr = summarize_cooperative(coop)
    cp = pairwise_cooperative(coop, MODELS)
    save(cr, r / "step2_cooperative/step2_cooperative_human_results_with_ci.csv")
    save(cp, r / "step2_cooperative/step2_cooperative_human_pairwise_inference.csv")
    for col, name in [("subdomain", "subdomain"), ("final_primary_ambiguity_source", "ambiguity_source")]:
        sub = coop.groupby(["model", col], sort=True).human_score.agg(["count", "mean"]).reset_index()
        save(sub, r / f"step2_cooperative/step2_human_by_{name}.csv")
    ca = load_csv(root / "human_evaluation/cooperative/cooperative_annotator_A_FILLED.csv")
    cb = load_csv(root / "human_evaluation/cooperative/cooperative_annotator_B_FILLED.csv")
    ab = ca.merge(cb, on="blind_response_id", suffixes=("_a", "_b"), validate="one_to_one")
    assert len(ab) == 480 and ab.response_a.eq(ab.response_b).all()
    ar = []
    for col in ["interpretation_validity", "assumption_quality", "answer_usefulness"]:
        ar.append(dict(dimension=col, n=len(ab), exact_agreement=ab[col+"_a"].eq(ab[col+"_b"]).mean(),
                       weighted_cohen_kappa=cohen_kappa_score(ab[col+"_a"], ab[col+"_b"], weights="quadratic"),
                       disagreement_count=int(ab[col+"_a"].ne(ab[col+"_b"]).sum())))
    save(pd.DataFrame(ar), r / "step2_cooperative/step2_pre_adjudication_agreement.csv")

    action_raw = load_csv(root / "raw_outputs/step3/step3_raw_outputs_unified_action360_v7.csv")
    assert len(action_raw) == 1440 and not action_raw.duplicated(["model", "query_id"]).any()
    assert set(action_raw.model) == set(MODELS + ["Always-Clarify-Baseline"])
    assert action_raw.groupby("model").size().eq(360).all()
    assert action_raw.query_text.eq(action_raw.query_id.map(bench.set_index("query_id").query_text)).all()
    action = attach_gold_actions(action_raw, bench)
    action_metrics = summarize_actions(action)
    action_pairs = pairwise_action_accuracy(action, MODELS)
    save(action, r / "step3_action_selection/step3_all360_unified_action_classification.csv")
    save(action_metrics, r / "step3_action_selection/step3_primary_unified_action_metrics.csv")
    save(action_pairs, r / "step3_action_selection/step3_unified_action_pairwise_inference.csv")
    save(confusion_table(action), r / "step3_action_selection/step3_unified_action_confusion_matrix.csv")
    cirows = []
    for model, d in action.groupby("model", sort=False):
        for track in ["ALL", "ANSWER", "INTERPRET", "CLARIFY"]:
            x = d if track == "ALL" else d[d.expected_action.eq(track)]
            mean, lo, hi = bootstrap_mean_ci(x.action_correct)
            cirows.append(dict(model=model, expected_action=track, n=len(x), accuracy=mean, ci_low=lo, ci_high=hi))
    save(pd.DataFrame(cirows), r / "step3_action_selection/step3_action_accuracy_with_ci.csv")
    legacy = load_csv(root / "raw_outputs/step3/step3_legacy_uncooperative_behavior_v7.csv")
    assert len(legacy) == 480 and legacy.groupby("model").size().eq(120).all()
    legacyrows = []
    for weight in [0, .25, .5, .75, 1]:
        for model, d in legacy.groupby("model"):
            value = d.behavior.map({"clarification": 1.0, "abstention": weight}).fillna(0).mean()
            legacyrows.append(dict(abstention_weight=weight, model=model, legacy_weighted_robustness=value))
    save(pd.DataFrame(legacyrows), r / "step3_action_selection/step3_legacy_robustness_weight_sensitivity.csv")

    masterrows = []
    for model in MODELS:
        er = primary.set_index("model").loc[model]
        c = cr.set_index("model").loc[model]
        ac = action_metrics.set_index("model").loc[model]
        masterrows.append(dict(model=model, execution_accuracy=er.accuracy, execution_ci_low=er.ci_low, execution_ci_high=er.ci_high,
          cooperative_human_score=c.human_cooperative_score, cooperative_ci_low=c.ci_low, cooperative_ci_high=c.ci_high,
          interpretation_validity=c.interpretation_validity_norm, assumption_quality=c.assumption_quality_norm, answer_usefulness=c.answer_usefulness_norm,
          action_accuracy=ac.action_accuracy, balanced_action_accuracy=ac.balanced_accuracy, action_macro_f1=ac.macro_f1,
          appropriate_clarification_X=ac.appropriate_clarification_rate_X, over_clarification_UC=ac.over_clarification_rate_UC,
          unsafe_answer_or_interpret_X=ac.unsafe_answer_or_interpret_rate_X))
    save(pd.DataFrame(masterrows), r / "FINAL_master_results_steps1_3.csv")
    candidates = [c for c in bench if "source" in c.lower() or "table" in c.lower()]
    scope_columns = pd.DataFrame([dict(column=c, nonempty_coverage=bench[c].astype(str).ne("").mean(),
                                      evidence_role="annotation_or_reference_metadata_not_model_table_input") for c in candidates])
    save(scope_columns, r / "step4_scope/step4_embedded_grounding_column_audit.csv")
    save(pd.DataFrame([dict(total_queries=len(bench), external_table_evidence_found=False,
          table_evidence_coverage=0.0, grounded_condition_available=False, decision="RECAST_PRE_RETRIEVAL",
          recommended_scope="pre-retrieval interaction decisions in sports-data question answering",
          reason="The archived experiments supplied no authentic source tables to the models; reference URLs are not model evidence inputs.")]),
         r / "step4_scope/step4_grounding_decision.csv")
    scope_note = "Models in the archived experiments received no source tables. Reference URLs in the benchmark support gold-source review and were not supplied as table evidence. Saved execution can reflect parametric knowledge; contamination is unmeasured. No table-grounded condition or fabricated evidence was created.\n"
    (r / "step4_scope/step4_revised_scope_note.txt").write_text(scope_note)
    env = {pkg: version(pkg) for pkg in ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "openpyxl"]}
    note = dict(benchmark_queries=len(bench), execution_ratings=len(gold), cooperative_ratings=len(coop),
                action_outputs=len(action), execution_correct=primary.set_index("model").correct.to_dict(),
                bootstrap_resamples=5000, permutation_resamples=10000, seed=42,
                ordering="Archived CSV row order is retained; groups preserve within-group row order.",
                execution_response_export_normalizations=int(ex.response_export_normalized.sum()),
                independent_human_review_verified=False, factual_gold_independently_verified=False,
                fresh_model_inference_run=False, analysis_environment=env)
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis_run.json").write_text(json.dumps(note, indent=2)+"\n")
    if figures:
        from make_figures import generate
        generate(bench, primary, cr, action_metrics, action, output / "figures")
    return note

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    result = run(args.root, args.output_dir or args.root/"reproduced", not args.no_figures)
    print(json.dumps(result, indent=2))
