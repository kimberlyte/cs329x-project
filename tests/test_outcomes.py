"""Outcome parsing, rubric boundaries and source/sample aggregation checks."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from data.analysis.outcomes import build_report, success_band, summarize_scores


class OutcomeTests(unittest.TestCase):
    def test_missing_invalid_and_fractional_scores_remain_distinct(self):
        result = summarize_scores([None, "", "bad", "101", "-1", "0.83", "82.0", 100])
        self.assertEqual((result["sessions"], result["scored"], result["missing"], result["invalid"]), (8, 3, 2, 3))
        self.assertEqual(result["fractional_scores"], 1)
        self.assertEqual(result["minimum"], .83)
        self.assertEqual(sum(result["histogram_10_point_counts"]), 3)
        self.assertEqual(sum(result["swe_rubric_bands"].values()), 3)

    def test_original_rubric_boundaries_and_score_100(self):
        expected = [(0, "0-9"), (9.5, "0-9"), (10, "10-29"), (30, "30-49"),
                    (50, "50-69"), (70, "70-89"), (90, "90-100"), (100, "90-100")]
        for value, band in expected:
            with self.subTest(value=value):
                self.assertEqual(success_band(value), band)
        self.assertEqual(success_band(None), "Missing")
        self.assertEqual(success_band(float("inf")), "Invalid")

    def test_aggregate_join_preserves_missing_scores_and_excludes_private_details(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            native = root / "raw/collabskill/leaderboard_data"
            native.mkdir(parents=True)
            (native / "sessions.json").write_text(json.dumps([
                {"session_id": "private-c1", "autograde_overall_score": 88, "autograde_score_id": "d1"},
                {"session_id": "private-c2", "autograde_overall_score": 0},
            ]))
            (native / "scores.json").write_text(json.dumps([
                {"_id": "d1", "detailed_scores": [{"criterion_name": "Correctness & Accuracy", "score": 35,
                                                      "comments": "private grader explanation"}]},
            ]))
            swe = root / "raw/swe-chat/sessions.parquet"
            swe.parent.mkdir(parents=True)
            pd.DataFrame({"session_id": ["private-s1", "private-s2"], "session_success": ["82", None],
                          "summary_outcome": ["private outcome explanation", ""]}).to_parquet(swe)
            config = root / "curation/config"
            config.mkdir(parents=True)
            (config / "download_manifest.json").write_text(json.dumps({"datasets": {
                "collabskill": {"revision": "a"*40}, "swe-chat": {"revision": "b"*40}}}))
            for folder, ids in [("collabskill-curated", ["private-c2"]), ("swe_chat_curated", ["private-s2"])]:
                path = root / "curated" / folder
                path.mkdir(parents=True)
                pd.DataFrame({"session_id": ids, "turn_type": ["single"]}).to_csv(path / "scenario_manifest.csv", index=False)
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            report = build_report(root, root / "curated")
            self.assertEqual(report["populations"]["CollabSkill"]["curated"]["mean"], 0)
            self.assertNotIn("swe_rubric_bands", report["populations"]["CollabSkill"]["source"])
            self.assertEqual(report["populations"]["SWE-chat"]["curated"]["missing"], 1)
            self.assertEqual(report["populations"]["SWE-chat"]["curated"]["scored"], 0)
            self.assertEqual(report["populations"]["SWE-chat"]["source"]["nonempty_outcome_summaries"], 1)
            self.assertNotIn("private-", json.dumps(report))
            self.assertNotIn("private grader", json.dumps(report))
            self.assertNotIn("private outcome", json.dumps(report))
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})


if __name__ == "__main__":
    unittest.main()
