# AmbiTab-Sports

Anonymous review repository for **AmbiTab-Sports: An Ambiguity-Aware Benchmark for Interaction Decisions in Sports Data Question Answering**.

The release includes the complete 360-query benchmark, construction records, recorded response ratings, raw outputs and checkpoints, reproducible analysis, six figure pairs, and anonymous LaTeX manuscript/response sources with compiled PDFs.

Models receive no tables. The experiments assess retrieval-free factual responses, cooperative rubric records, and initial selection among ANSWER, INTERPRET, and CLARIFY. They do not establish grounded table reasoning.

## Reproduce the saved analyses

Use Python 3.12 from a terminal in the extracted repository:

```bash
python -m pip install -r requirements.txt
python code/reproduce.py
python code/validate_release.py --compare reproduced
python code/verify_manifest.py
```

The first command installs analysis dependencies. Reproduction writes to `reproduced/` and leaves the released evidence/tables unchanged. It requires no GPU, network access after installation, model download, or private blinding key. The final two commands check regenerated tables and archive integrity. The canonical notebook under `notebooks/` calls the same implementation; install `requirements-notebook.txt` for a Jupyter interface.

## Evidence and interpretation

The recorded primary execution counts are Gemma 43/120, Qwen 22/120, and Mistral 19/120. The corrected master uses these labels. Automatic execution scoring is secondary; it is not substituted into the master. Paired execution advantages for Gemma are conditional on the recorded labels and references. No corrected LLM differences are significant for cooperative score or overall action accuracy. The strong rank-reversal claim is withdrawn.

**Independent human authorship/re-review of the supplied response ratings is not established.** Earlier AI-assisted scoring records and human-rating descriptions have not been reconciled by documented independent review. Historical filenames containing `human` are retained for compatibility and do not certify provenance. The proposed response-review guides have not been applied to the archived labels. See `docs/Annotation_Provenance.md`.

Execution answers are preserved from the supplied benchmark. Original verification claims are retained in `supplied_verification_status`; current `verification_status` distinguishes those claims from independent validation. Six references/sources need explicit review, detailed in `benchmark/execution_gold_source_audit.csv`. The release does not certify all 120 factual references. See `docs/Gold_Source_Review.md`.

## Files

| Directory | Contents |
|---|---|
| `benchmark/` | Canonical 360-row CSV, 63-field dictionary, schema, source audit, final release workbook, construction A/B sheets, historical master |
| `human_evaluation/` | Supplied blinded execution A/B/final records and cooperative A/B/final records; score workbook and review guides |
| `raw_outputs/` | Original execution/action/legacy response CSVs; cooperative responses are also in the supplied final cooperative corpus |
| `checkpoints/` | Saved model response arrays |
| `results/` | Canonical regenerated primary, secondary, paired, agreement, action, and scope analyses |
| `figures/` | Six publication figure pairs, PDF and PNG |
| `code/` | Saved-response analysis, scoring, figures, validation, manifest checking, optional historical-style model loading |
| `notebooks/` | Clean canonical notebook and sanitized historical notebooks under `archive/` |
| `models/` | Exact model repository IDs, unknown historical revision SHAs, settings, run metadata |
| `paper/` | Anonymous manuscript, point-by-point response, cover-letter draft, PDFs, LaTeX/BibTeX/template dependencies |
| `docs/` | Reproduction, provenance, source-review, scoring/construction guides, repair history |
| `archive/` | Supplied historical results, figures, scripts, and metadata, clearly superseded |
| `validation/` | Technical release checks; these are not independent human or factual validation |

The historical train/dev/test split is 252/54/54. All 360 items are used in the revised zero-shot analyses, with 120 U/C/X queries and 20 queries per subdomain/track cell.

## Manuscript and repository URL

The manuscript and response are anonymous drafts that accurately describe the recorded evidence and remaining scientific limits. Compile from `paper/` with `latexmk -pdf ambitab_sports_revised.tex`, then `latexmk -pdf response_to_reviewers.tex` and `latexmk -pdf revision_cover_letter.tex`. The main manuscript must compile first because the response uses its labels/page references.

Add the actual anonymous review URL to the single `RepositoryURL` value in `paper/submission_settings.tex`, then recompile. After intentional release edits, rerun validation, refresh checksums with `python code/write_manifest.py`, and verify them with `python code/verify_manifest.py`. No public URL or upload is asserted here. Restore author details only in the journal's author-identifying submission version, outside the anonymous repository.

## Optional fresh inference

Install `requirements-models.txt` only for a separate new inference experiment. The historical model commit SHAs and fully locked GPU environment were not recorded. The saved responses can be analyzed exactly; a bit-for-bit historical model rerun is not claimed. New exploratory runs must record actual revisions/settings and be versioned separately. `code/model_inference.py` accepts an explicit revision for new runs.

## Release status

Technical packaging/reproduction is tested. Scientific completion still requires independent response-rating validation, resolution of flagged factual references, and clarification of historical adjudication provenance. The final construction workbook is a CSV release export; the historical master retains 497 open dispute entries. No historical decisions, sources, raters, tables, or checkpoint SHAs were invented.

The MIT license and anonymous citation file accompany this release. Source URLs identify underlying sports statistics; their presence is not a statement that those sources were supplied to the models.
