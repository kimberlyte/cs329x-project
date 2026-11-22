"""Run paired CollabSkill deliverable evaluation with two independent judges."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from .swe_chat import (
    CONDITIONS, fingerprint, freeze, json_text, positive_seconds, read, read_rows,
    save_once, text_field,
)

WEIGHTS = {
    "Correctness & Accuracy": 40,
    "Deliverable Completeness": 20,
    "Task Requirements": 25,
    "Technical Quality": 15,
}
JUDGES = ("gemini", "claude")
PROMPTS = Path(__file__).with_name("collabskill_prompts.txt")


def inventory(root):
    """Hash an explicit evidence tree, without following symlinks."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Expected an ordinary directory: {root}")
    files = {}
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root)
        if path.is_symlink() or any(part.startswith(".") for part in name.parts):
            raise ValueError(f"Symlinks and hidden entries are unsupported: {name}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Expected a regular file: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        files[name.as_posix()] = digest.hexdigest()
    return files


def cases_from_split(holdout, task_dir):
    cases = {}
    for row in read_rows(holdout):
        if row.get("dataset", row.get("source_dataset")) != "CollabSkill":
            continue
        if row.get("split", "test") != "test":
            raise ValueError("Expected a held-out CollabSkill split.")
        case_id = row["example_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", case_id) or case_id in cases:
            raise ValueError("Invalid or duplicate case ID.")
        source = task_dir / case_id
        if source.is_symlink():
            raise ValueError(f"Expected an ordinary case directory: {case_id}")
        files = inventory(source)
        if "prompt.txt" not in files or not text_field((source / "prompt.txt").read_text()):
            raise ValueError(f"{case_id}: add the original task as prompt.txt.")
        if any(name not in {"prompt.txt", "rubric.json"} and not name.startswith("references/")
               for name in files):
            raise ValueError(f"{case_id}: put task files under references/.")
        if "rubric.json" in files:
            normalize_rubric(read(source / "rubric.json"))
        cases[case_id] = files
    if not cases:
        raise ValueError("The split contains no CollabSkill cases.")
    return cases


def points(value):
    return type(value) in (int, float) and 0 <= value <= 100


def normalize_rubric(rubric, *, generated=False):
    if not isinstance(rubric, dict) or rubric.get("total_points") != 100:
        raise ValueError("The rubric must declare total_points of 100 and full criteria.")
    categories = rubric.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("The rubric must contain scoring categories.")
    weights, checks = {}, {}
    for category in categories:
        name, maximum, criteria = (category[key] for key in ("name", "max_points", "criteria"))
        if (not text_field(name) or name in weights or not points(maximum) or maximum == 0
                or not isinstance(criteria, list) or not criteria):
            raise ValueError("Invalid rubric category.")
        weights[name] = maximum
        total = 0
        for criterion in criteria:
            check = {key: criterion[key] for key in
                     ("description", "points", "expected_value", "evaluation_method")}
            if (not points(check["points"]) or check["points"] == 0
                    or not all(text_field(check[key]) for key in ("description", "evaluation_method"))
                    or check["expected_value"] in (None, [], {})
                    or (isinstance(check["expected_value"], str) and not check["expected_value"].strip())):
                raise ValueError("Each criterion needs points, an expected result, and a verification method.")
            total += check["points"]
            checks[f"C{len(checks) + 1:03}"] = {"category": name, **check}
        if not math.isclose(total, maximum, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Criterion points do not add up in {name}.")
    if not math.isclose(sum(weights.values()), 100, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Rubric weights must add up to 100.")
    if generated and weights != WEIGHTS:
        raise ValueError("Generated rubric weights differ from the paper's defaults.")
    critical = rubric.get("critical_requirements", [])
    if not isinstance(critical, list) or not all(text_field(item) for item in critical):
        raise ValueError("critical_requirements must be a list of descriptions.")
    return {"weights": weights, "criteria": checks, "critical_requirements": critical,
            "evaluation_notes": rubric.get("evaluation_notes", "")}


def evidence_views(root, destination, files):
    """Expose Office package XML in readable files alongside the original evidence."""
    views = {}
    for number, name in enumerate(files):
        file = root / name
        if file.suffix.lower() not in {".docx", ".xlsx", ".pptx"}:
            continue
        target = destination / f"office-{number:03}"
        target.mkdir(parents=True)
        if not zipfile.is_zipfile(file):
            views[name] = {"error": "The file is not a valid ZIP-based Office package."}
            continue
        with zipfile.ZipFile(file) as archive:
            members = [m for m in archive.infolist() if m.filename.endswith((".xml", ".rels"))]
            if len(members) > 1500 or sum(m.file_size for m in members) > 32 * 1024**2:
                raise ValueError(f"{name}: Office package exceeds inspection limits.")
            if "[Content_Types].xml" not in {member.filename for member in members}:
                views[name] = {"error": "The Office package has no [Content_Types].xml entry."}
                continue
            index = {}
            for index_number, member in enumerate(members):
                part = target / f"part-{index_number:04}.txt"
                part.write_bytes(archive.read(member))
                index[member.filename] = part.name
            (target / "index.json").write_text(json.dumps(index, indent=2))
        views[name] = {"index": (target / "index.json").relative_to(root.parent).as_posix()}
    return views


def stage_evidence(source, target, views_dir):
    files = inventory(source)
    shutil.copytree(source, target, dirs_exist_ok=True)
    return {"files": list(files), "office_views": evidence_views(target, views_dir, files)}


def model_json(command, prompt, payload, workspace, timeout):
    """Pass data on stdin and unwrap the provider's JSON response envelope."""
    if command[0] == "gemini":
        settings = workspace / ".gemini"
        settings.mkdir()
        (settings / "settings.json").write_text(json_text({
            "tools": {"core": ["read_file", "list_directory", "glob", "grep_search"]},
            "mcp": {"allowed": ["none"]},
        }))
    try:
        result = subprocess.run(
            command, input=prompt + "\n\nEvaluation data (JSON):\n" + json_text(payload),
            text=True, encoding="utf-8", stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            cwd=workspace, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"{command[0]} timed out after {timeout}s.") from exc
    if result.returncode:
        raise ValueError(f"{command[0]} exited with status {result.returncode}. Check CLI authentication and version.")
    try:
        envelope = json.loads(result.stdout)
        if envelope.get("is_error") or envelope.get("error"):
            raise ValueError("judge error")
        response = envelope.get("response", envelope.get("result")).strip()
        fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", response)
        parsed = json.loads(fenced[1] if fenced else response)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{command[0]} did not return valid model JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{command[0]} must return a JSON object.")
    return parsed


def judge_commands(args):
    return {
        "gemini": [
            "gemini", "--model", args.gemini_model, "--approval-mode", "plan",
            "--extensions", "none", "--allowed-mcp-server-names", "none", "--output-format", "json",
        ],
        "claude": [
            "claude", "--print", "--model", args.claude_model, "--output-format", "json",
            "--permission-mode", "plan", "--tools", "Read,Glob,Grep", "--strict-mcp-config",
            "--mcp-config", '{"mcpServers":{}}', "--setting-sources", "", "--no-session-persistence",
        ],
    }


def validate_grade(grade, rubric, files):
    criteria = rubric["criteria"]
    if (set(grade) != {"criteria", "summary", "files_read"}
            or not text_field(grade["summary"])
            or not isinstance(grade["files_read"], list)
            or any(not isinstance(name, str) or name not in files for name in grade["files_read"])
            or not isinstance(grade["criteria"], dict)
            or set(grade["criteria"]) != set(criteria)):
        raise ValueError("Grade does not match the frozen rubric or file inventory.")
    scores = []
    for key, result in grade["criteria"].items():
        if not isinstance(result, dict) or set(result) != {"earned", "reason"} or not text_field(result["reason"]):
            raise ValueError(f"{key}: expected earned points and an evidence reason.")
        earned = result["earned"]
        if earned is not None and (not points(earned) or earned > criteria[key]["points"]):
            raise ValueError(f"{key}: points outside the criterion range.")
        scores.append(earned)
    if None in scores or set(grade["files_read"]) != set(files):
        return None
    return round(sum(scores), 8)


def load_inputs(args):
    cases = cases_from_split(args.holdout, args.tasks)
    config = {"inputs": cases}
    freeze(args.run_dir / "inputs.json", config)
    return cases


def generate(args):
    cases = load_inputs(args)
    if not args.runner or args.base == args.sft:
        raise ValueError("Supply a trusted runner and two distinct checkpoints.")
    models = {"base": args.base, "sft": args.sft}
    freeze(args.run_dir / "generation.json", {
        "models": models, "runner": args.runner, "trial_config": args.trial_config,
        "timeout": args.timeout,
    })
    runner = args.runner.copy()
    # The runner works in a fresh directory, so resolve repository-relative paths first.
    for index, token in enumerate(runner):
        if not token.startswith("-") and Path(token).is_file():
            runner[index] = str(Path(token).resolve())
    failures = 0
    for case_id in cases:
        source = args.tasks / case_id
        for condition, model in models.items():
            destination = args.run_dir / "submissions" / case_id / condition
            if destination.exists():
                inventory(destination)
                continue
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
                    workspace = Path(temp)
                    output = workspace / "deliverables"
                    output.mkdir()
                    shutil.copyfile(source / "prompt.txt", workspace / "prompt.txt")
                    if (source / "references").is_dir():
                        shutil.copytree(source / "references", workspace / "references")
                    request = {"case_id": case_id, "model": model, "task": "prompt.txt",
                               "references": "references", "output": "deliverables"}
                    (workspace / "references").mkdir(exist_ok=True)
                    completed = subprocess.run(
                        runner, input=json_text(request), text=True, cwd=workspace,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=args.timeout, check=False,
                    )
                    if completed.returncode:
                        raise ValueError(f"Runner exited with code {completed.returncode}.")
                    inventory(output)
                    output.rename(destination)
                print(f"Saved {case_id} {condition} submission.")
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                failures += 1
                print(f"{case_id} {condition}: {exc}", file=sys.stderr)
    return bool(failures)


def get_rubric(args, case_id, generate_prompt, commands):
    source = args.tasks / case_id
    path = args.run_dir / "rubrics" / f"{case_id}.json"
    if path.exists():
        return read(path)
    if (source / "rubric.json").exists():
        rubric = normalize_rubric(read(source / "rubric.json"))
    else:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            references = {"files": [], "office_views": {}}
            if (source / "references").is_dir():
                references = stage_evidence(source / "references", workspace / "references",
                                            workspace / "inspection")
            payload = {"task": (source / "prompt.txt").read_text(encoding="utf-8"),
                       "references": references, "weights": WEIGHTS}
            rubric = normalize_rubric(model_json(commands["gemini"], generate_prompt, payload,
                                                workspace, args.timeout), generated=True)
    save_once(path, rubric)
    return rubric


def load_grade(path, rubric, file_hashes, config):
    record = read(path)
    if record["evidence_sha256"] != fingerprint([rubric, file_hashes, config]):
        raise ValueError(f"Saved grade no longer matches its evidence: {path}")
    return validate_grade(record["grade"], rubric, file_hashes)


def grade_case(args, case_id, rubric, config, prompt):
    source = args.tasks / case_id
    failed = False
    for condition in CONDITIONS:
        submission = args.run_dir / "submissions" / case_id / condition
        try:
            files = inventory(submission)
        except (OSError, ValueError) as exc:
            print(f"{case_id} {condition}: {exc}", file=sys.stderr)
            failed = True
            continue
        for judge, command in config["commands"].items():
            path = args.run_dir / "grades" / case_id / condition / f"{judge}.json"
            try:
                if path.exists():
                    load_grade(path, rubric, files, config)
                    continue
                with tempfile.TemporaryDirectory() as temp:
                    workspace = Path(temp)
                    evidence = stage_evidence(submission, workspace / "deliverables",
                                              workspace / "inspection" / "deliverables")
                    references = {"files": [], "office_views": {}}
                    if (source / "references").is_dir():
                        references = stage_evidence(source / "references", workspace / "references",
                                                    workspace / "inspection" / "references")
                    payload = {"task": (source / "prompt.txt").read_text(encoding="utf-8"),
                               "rubric": rubric, "deliverables": evidence, "references": references}
                    result = model_json(command, prompt, payload, workspace, args.timeout)
                score = validate_grade(result, rubric, files)
                save_once(path, {"evidence_sha256": fingerprint([rubric, files, config]), "grade": result})
                print(f"{case_id} {condition} {judge}: {score if score is not None else 'unassessable'}")
            except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as exc:
                print(f"{case_id} {condition} {judge}: {exc}", file=sys.stderr)
                failed = True
    return failed


def grade(args):
    cases = load_inputs(args)
    generate_prompt, grading_prompt = PROMPTS.read_text(encoding="utf-8").split("\n---\n")
    config = {"commands": judge_commands(args), "prompts_sha256": fingerprint([generate_prompt, grading_prompt]),
              "timeout": args.timeout}
    freeze(args.run_dir / "judges.json", config)
    failures = []
    for case_id in cases:
        try:
            rubric = get_rubric(args, case_id, generate_prompt, config["commands"])
            failures.append(grade_case(args, case_id, rubric, config, grading_prompt))
        except (ValueError, OSError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            print(f"{case_id}: {exc}", file=sys.stderr)
            failures.append(True)
    return any(failures)


def summarize(args):
    cases = load_inputs(args)
    config = read(args.run_dir / "judges.json")
    pairs, incomplete = [], []
    for case_id in cases:
        rubric_path = args.run_dir / "rubrics" / f"{case_id}.json"
        scores = {}
        for condition in CONDITIONS:
            submission = args.run_dir / "submissions" / case_id / condition
            grades = args.run_dir / "grades" / case_id / condition
            values = {}
            if rubric_path.exists() and submission.is_dir():
                rubric, files = read(rubric_path), inventory(submission)
                for judge in JUDGES:
                    path = grades / f"{judge}.json"
                    if path.exists():
                        values[judge] = load_grade(path, rubric, files, config)
            scores[condition] = values
        row = {"case_id": case_id, "judge_scores": scores}
        if all(len(values) == len(JUDGES) and None not in values.values() for values in scores.values()):
            means = {condition: statistics.mean(values.values()) for condition, values in scores.items()}
            pairs.append({**row, **means, "sft_minus_base": round(means["sft"] - means["base"], 8)})
        else:
            incomplete.append(row)
    deltas = [pair["sft_minus_base"] for pair in pairs]
    print(json.dumps({
        "expected_pairs": len(cases), "complete_pairs": len(pairs),
        "mean_sft_minus_base": statistics.mean(deltas) if deltas else None,
        "sft_wins": sum(delta > 0 for delta in deltas),
        "ties": deltas.count(0), "sft_losses": sum(delta < 0 for delta in deltas),
        "pairs": pairs, "incomplete_pairs": incomplete,
    }, indent=2))
    return bool(incomplete)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for action in (generate, grade, summarize):
        options = stages.add_parser(action.__name__)
        options.add_argument("--holdout", type=Path, required=True)
        options.add_argument("--tasks", type=Path, required=True)
        options.add_argument("--run-dir", type=Path, required=True)
        options.set_defaults(action=action)
        if action is generate:
            options.add_argument("--base", required=True)
            options.add_argument("--sft", required=True)
            options.add_argument("--trial-config", required=True)
            options.add_argument("--timeout", type=positive_seconds, default=1800)
            options.add_argument("--runner", nargs=argparse.REMAINDER, required=True)
        elif action is grade:
            options.add_argument("--gemini-model", required=True)
            options.add_argument("--claude-model", required=True)
            options.add_argument("--timeout", type=positive_seconds, default=600)
    args = parser.parse_args(argv)
    try:
        return int(bool(args.action(args)))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
