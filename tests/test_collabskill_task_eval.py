import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from data.evaluation import collabskill


class CollabSkillEvalTests(unittest.TestCase):
    def rubric(self):
        return {
            "total_points": 100,
            "categories": [
                {"name": name, "max_points": points, "criteria": [{
                    "description": f"Check {name}", "points": points,
                    "expected_value": "report is a usable workbook",
                    "evaluation_method": "inspect the file",
                }]}
                for name, points in collabskill.WEIGHTS.items()
            ],
        }

    def test_paired_run_freezes_one_rubric_and_blinds_four_grades(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            split = root / "test.jsonl"
            split.write_text(json.dumps({
                "dataset": "CollabSkill", "example_id": "demo", "events": ["old interaction"],
            }))
            tasks = root / "tasks"
            case = tasks / "demo"
            (case / "references").mkdir(parents=True)
            (case / "prompt.txt").write_text("Prepare an Excel report.")
            (case / "references" / "source.txt").write_text("Source evidence.")
            runner = root / "runner.py"
            runner.write_text(
                "import json, sys, zipfile\n"
                "from pathlib import Path\n"
                "request = json.load(sys.stdin)\n"
                "assert Path(request['references'], 'source.txt').read_text() == 'Source evidence.'\n"
                "with zipfile.ZipFile(Path(request['output'], 'report.xlsx'), 'w') as out:\n"
                "    out.writestr('[Content_Types].xml', '<Types/>')\n"
                "    out.writestr('xl/workbook.xml', request['model'])\n"
            )
            run_dir = root / "run"
            common = ["--holdout", str(split), "--tasks", str(tasks), "--run-dir", str(run_dir)]
            commands = ["--gemini-model", "gemini-fixture", "--claude-model", "claude-fixture"]
            calls = []

            def model_json(command, prompt, payload, workspace, timeout):
                self.assertNotIn("old interaction", json.dumps(payload))
                self.assertNotIn("case_id", payload)
                if "rubric" not in payload:
                    calls.append("generate rubric")
                    self.assertFalse((workspace / "deliverables").exists())
                    return self.rubric()
                calls.append(command[0])
                self.assertEqual(payload["deliverables"]["files"], ["report.xlsx"])
                index = workspace / payload["deliverables"]["office_views"]["report.xlsx"]["index"]
                part = json.loads(index.read_text())["xl/workbook.xml"]
                improved = (index.parent / part).read_text() == "sft-fixture"
                criteria = payload["rubric"]["criteria"]
                return {
                    "summary": "Inspected the workbook package.",
                    "files_read": ["report.xlsx"],
                    "criteria": {
                        key: {"earned": item["points"] if improved else item["points"] / 2,
                              "reason": "Checked the workbook data."}
                        for key, item in criteria.items()
                    },
                }

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(collabskill.main(["generate", *common, "--base", "base-fixture",
                                                  "--sft", "sft-fixture", "--trial-config", "fixture",
                                                  "--runner", "python3", str(runner)]), 0)
                with patch.object(collabskill, "model_json", side_effect=model_json):
                    self.assertEqual(collabskill.main(["grade", *common, *commands]), 0)
                    self.assertEqual(collabskill.main(["grade", *common, *commands]), 0)
            self.assertEqual(calls, ["generate rubric", "gemini", "claude", "gemini", "claude"])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(collabskill.main(["summarize", *common]), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["complete_pairs"], 1)
            self.assertEqual(result["pairs"][0]["judge_scores"],
                             {"base": {"claude": 50, "gemini": 50},
                              "sft": {"claude": 100, "gemini": 100}})
            self.assertEqual(result["mean_sft_minus_base"], 50)

    def test_unverifiable_evidence_is_not_scored_as_zero(self):
        rubric = collabskill.normalize_rubric(self.rubric())
        checks = {key: {"earned": 0, "reason": "Missing required content."}
                  for key in rubric["criteria"]}
        grade = {"summary": "No files produced.", "criteria": checks, "files_read": []}
        self.assertEqual(collabskill.validate_grade(grade, rubric, {}), 0)
        self.assertIsNone(collabskill.validate_grade(grade, rubric, {"unreadable.pdf": "hash"}))
        checks["C001"]["earned"] = None
        self.assertIsNone(collabskill.validate_grade(grade, rubric, {}))
        checks["C001"]["earned"] = 41
        with self.assertRaisesRegex(ValueError, "outside the criterion range"):
            collabskill.validate_grade(grade, rubric, {})

    def test_provider_json_wrappers(self):
        payload = {"criteria": {}}
        for provider, key in (("gemini", "response"), ("claude", "result")):
            response = json.dumps({key: "```json\n" + json.dumps(payload) + "\n```"})
            with tempfile.TemporaryDirectory() as temp:
                with patch.object(collabskill.subprocess, "run") as run:
                    run.return_value = collabskill.subprocess.CompletedProcess([], 0, stdout=response)
                    self.assertEqual(collabskill.model_json([provider], "prompt", {}, Path(temp), 5),
                                     payload)
