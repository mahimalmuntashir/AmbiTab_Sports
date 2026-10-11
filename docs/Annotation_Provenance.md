# Annotation provenance

The supplied construction A/B workbooks cover 360 queries each. The final release CSV contains final fields for all queries. The historical master remains an unfinished starter: 497 dispute rows are open, zero final decisions are recorded in its log, and six final track cells are blank. The CSV export does not reconstruct a contemporaneous adjudication chronology.

Execution ratings cover 360 blinded LLM responses; A/B binary labels agree on all rows. Eight response-text fields differ across the A/B exports, with identical incorrect labels. One original Qwen response literally says `None`, whereas its blinded export is empty; the scorer preserves the original and reports a normalization flag. Identical responses across models are associated only when recorded labels agree. No private ID key is needed or published by this association.

Cooperative ratings cover 480 responses. The supplied final corpus marks 205 rows as agreement and 275 as adjudicated. It does not preserve free-text adjudicator rationales. Exact agreement and quadratic kappa are computed before adjudication for each dimension. The final CSV shows model identities; the timing of historical unblinding is not independently established.

Earlier project records describe AI-assisted scoring passes, while supplied filenames/documentation describe human evaluation. No independently documented human re-review reconciles these descriptions. Therefore this release reports **recorded annotations**, not a validated independent human replacement for the rejected automatic judge. Perfect binary agreement and blinded IDs do not prove independence. The original labels are unchanged by the repair.

The new response guides are prospective validation protocols, clearly labeled as such. To establish independent human evidence, actual raters should review identical preserved stimuli independently, document sources/reasons and any AI assistance, and record a separate adjudication process. Anonymous rater IDs are sufficient for public review. Recompute and version the analysis if ratings change; do not overwrite the historical source records or invent old rationales.
