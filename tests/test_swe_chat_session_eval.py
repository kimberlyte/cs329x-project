import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from data.evaluation import swe_chat


class SweChatSessionEvalTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.cases = self.root / "cases.jsonl"
        self.run_dir = self.root / "run"
        self.common = ["--cases", str(self.cases), "--run-dir", str(self.run_dir)]

    def stage(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = swe_chat.main(list(args))
        return status, stdout.getvalue(), stderr.getvalue()

    def test_end_to_end_resume_and_blind_grading(self):
        holdout = self.root / "test.jsonl"
        holdout.write_text(json.dumps({
            "example_id": "V2-TEST", "dataset": "SWE-chat", "session_success": 99,
            "events": [
                {"source_event": {"role": "user", "content": "Explain the parser.",
                                  "is_conversational": True}},
                {"source_event": {"role": "assistant", "content": "Historic response.",
                                  "is_conversational": True}},
                {"source_event": {"role": "user", "content": "Historic follow-up.",
                                  "is_conversational": True}},
            ],
        }) + "\n")
        status, _, errors = self.stage("prepare", "--holdout", str(holdout),
                                       "--cases", str(self.cases))
        self.assertEqual(status, 0, errors)
        self.assertEqual(json.loads(self.cases.read_text()), {
            "case_id": "V2-TEST",
            "initial_messages": [{"role": "user", "content": "Explain the parser."}],
        })
        adapter = self.root / "runner.py"
        adapter.write_text(
            "import json, sys\n"
            "req = json.load(sys.stdin)\n"
            "text = 'Detailed answer.' if req['model'] == 'mock-sft' else 'Partial answer.'\n"
            "print(json.dumps({'transcript': req['case']['initial_messages'] + "
            "[{'role': 'assistant', 'content': text}], 'tool_calls': "
            "[{'name': 'read_file', 'category': 'read'}], 'commit_evidence': []}))\n"
        )
        generate = ["generate", *self.common, "--base", "mock-base", "--sft", "mock-sft",
                    "--trial-config", "offline-test", "--runner", sys.executable, str(adapter)]
        status, _, errors = self.stage(*generate)
        self.assertEqual(status, 0, errors)
        with patch.object(swe_chat, "run_adapter", side_effect=AssertionError("ran again")):
            self.assertEqual(self.stage(*generate)[0], 0)

        requests = []

        def judge(request, *, timeout):
            body = json.loads(request.data)
            requests.append(body)
            evidence = json.loads(body["messages"][0]["content"])
            self.assertEqual(set(evidence), {"transcript", "tool_summary", "commit_evidence"})
            self.assertEqual(evidence["tool_summary"], {
                "category_counts": {"read": 1}, "top_tools": {"read_file": 1},
            })
            for hidden in ("V2-TEST", "mock-base", "mock-sft", "Historic", "session_success"):
                self.assertNotIn(hidden, body["messages"][0]["content"])
            score = 90 if evidence["transcript"][-1]["content"] == "Detailed answer." else 60
            return io.BytesIO(json.dumps({
                "id": "mock-response", "model": body["model"], "stop_reason": "end_turn",
                "content": [{"type": "text", "text": json.dumps({
                    "score": score, "reason": "Scored the newly generated explanation."
                })}],
            }).encode())

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "mock-key"}):
            with patch.object(swe_chat, "urlopen", side_effect=judge):
                status, _, errors = self.stage("grade", *self.common)
                self.assertEqual(status, 0, errors)
                self.assertEqual(self.stage("grade", *self.common)[0], 0)
        self.assertEqual(len(requests), 2)
        status, stdout, errors = self.stage("summarize", *self.common)
        self.assertEqual(status, 0, errors)
        summary = json.loads(stdout)
        self.assertEqual(summary["pairs"], [
            {"case_id": "V2-TEST", "base": 60, "sft": 90, "delta": 30}
        ])
        self.assertEqual(summary["sft_wins"], 1)

        (self.run_dir / "grades" / "V2-TEST" / "sft.json").unlink()
        status, stdout, _ = self.stage("summarize", *self.common)
        self.assertEqual(status, 1)
        self.assertEqual(json.loads(stdout)["incomplete_pairs"], [
            {"case_id": "V2-TEST", "available_scores": {"base": 60}}
        ])

        source = self.run_dir / "sessions" / "V2-TEST" / "base.json"
        session = json.loads(source.read_text())
        session["transcript"][-1]["content"] = "Different answer."
        source.write_text(json.dumps(session))
        status, _, errors = self.stage("summarize", *self.common)
        self.assertEqual(status, 1)
        self.assertIn("Stale grade", errors)

    def test_rejects_noninteger_and_out_of_range_scores(self):
        for score in (True, 82.5, "82", -1, 101):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    swe_chat.validate_grade({"score": score, "reason": "Example"})


if __name__ == "__main__":
    unittest.main()
