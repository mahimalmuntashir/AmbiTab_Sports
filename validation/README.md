# Technical validation

`release_validation.json` records structural, preservation, metadata, and fresh-reproduction checks. `input_preservation.json` records unchanged raw outputs/checkpoints/annotation CSVs and metadata-only workbook operations. `fresh_extraction_checks.json` records the ZIP extraction, notebook execution, and manuscript compilation checks. The hash manifest inventories every released file.

Run `python code/validate_release.py --compare reproduced` after reproduction. Run `python -m unittest discover -s tests -v` for the response-association regressions and `python code/verify_manifest.py` for archive integrity. Generated `reproduced/` outputs and local build/cache files are excluded from the released-file inventory.
