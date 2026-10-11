# Reproduction guide

## Saved-response analysis

Python 3.12 and `requirements.txt` reproduce the released analysis. From the repository root:

```bash
python code/reproduce.py
python code/validate_release.py --compare reproduced
python code/verify_manifest.py
```

Outputs go to `reproduced/`; primary release files remain unchanged. The notebook calls the same `run()` function, with root discovery from the current directory/ancestors. Use `--root /path/to/repository` and `--output-dir /path/to/results` when running the script elsewhere.

Inputs are the canonical benchmark, original response CSVs, and recorded ratings. Execution identities are associated by query/response content, with consistent-label checks for identical responses; no model column is required in the blinded file. One literal `None`/empty export difference is documented and flagged. The original response text is retained.

Primary execution uses the recorded binary labels. Automatic task-aware matching remains secondary. Cooperative analysis uses supplied final three-dimension rubric scores; action selection uses the saved common-prompt outputs. No independent human rating or new GPU inference occurs during reproduction.

## Statistics and ordering

Percentile bootstrap uses 5,000 resamples; paired permutation uses 10,000 draws; seed is 42. Within-model/source CSV ordering is retained, and pairs preserve the first model's query ordering. All intervals are regenerated consistently. This changes the archived Mistral execution lower endpoint from 0.091667 to 0.100000 without changing its mean, labels, or significance. Other central claims are unchanged. The archived supplied table remains under `archive/` for traceability.

Holm correction is applied separately to the three LLM pairs within execution, cooperative score, and action accuracy. Baselines and diagnostic subgroups are descriptive. Non-significance is not equivalence.

## Notebooks and inference

The clean canonical notebook performs saved-response analysis on CPU. Sanitized historical notebooks retain their historical code under `notebooks/archive/`; they include Kaggle-specific assumptions and are not the canonical reproduction entry point.

Exact model repository IDs are supplied, but historical commits are unknown. Do not replace an unknown historical SHA with current HEAD. Optional new inference requires the model requirements and appropriate resources; record its actual revisions, runtime, interface, thinking-mode settings, and generation budgets separately.

## Documents

The anonymous LaTeX manuscript, response, and cover-letter draft are under `paper/`. A TeX distribution is required to compile them; the main manuscript must compile before the response's external page references. `code/build_docs.py` recreates the five PDF review guides with `requirements-docs.txt`. Prospective guides do not retroactively certify the archived ratings.

## Integrity and limits

`MANIFEST_SHA256.txt` covers all released files except itself. Verify after extracting the ZIP. Generated `reproduced/` and local interpreter/notebook caches are excluded from release membership checks. `validation/` records technical tests; it is not independent human validation or a certification of all factual references.
