"""Regressions for the blinded-label association failures in the supplied package."""
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import pandas as pd
from data_loading import associate_execution_labels

class BlindedAssociationTests(unittest.TestCase):
    def test_conflicting_identical_responses_require_resolution(self):
        raw = pd.DataFrame({"query_id": ["q", "q"], "model": ["a", "b"], "raw_response": ["same", "same"]})
        gold = pd.DataFrame({"query_id": ["q", "q"], "raw_response": ["same", "same"], "human_correct": [0, 1]})
        with self.assertRaisesRegex(ValueError, "Conflicting labels"):
            associate_execution_labels(raw, gold, ["a", "b"])

    def test_literal_none_is_preserved_when_matching_empty_export(self):
        raw = pd.DataFrame({"query_id": ["q"], "model": ["a"], "raw_response": ["None"]})
        gold = pd.DataFrame({"query_id": ["q"], "raw_response": [""], "human_correct": [0]})
        out = associate_execution_labels(raw, gold, ["a"])
        self.assertEqual(out.loc[0, "raw_response"], "None")
        self.assertTrue(out.loc[0, "response_export_normalized"])
        self.assertEqual(out.loc[0, "human_correct"], 0)

    def test_unmatched_response_is_rejected(self):
        raw = pd.DataFrame({"query_id": ["q"], "model": ["a"], "raw_response": ["original"]})
        gold = pd.DataFrame({"query_id": ["q"], "raw_response": ["changed"], "human_correct": [1]})
        with self.assertRaisesRegex(ValueError, "Unmatched execution responses"):
            associate_execution_labels(raw, gold, ["a"])

if __name__ == "__main__":
    unittest.main()
