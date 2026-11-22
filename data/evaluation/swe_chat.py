"""Generate paired SWE-chat runs, grade them, and summarize within-case differences."""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CONDITIONS = ("base", "sft")
JUDGE_MODEL = "claude-sonnet-4-6"
RUBRIC = Path(__file__).with_name("swe_chat_rubric.txt")


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_once(path, value, *, lines=False):
    """Publish complete records without replacing an existing result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, encoding="utf-8", mode="w") as tmp:
        records = value if lines else [value]
        for record in records:
            tmp.write(json_text(record) + "\n")
        tmp.flush()
        os.link(tmp.name, path)


def freeze(path, value):
    try:
        save_once(path, value)
    except FileExistsError:
        if read(path) != value:
            raise ValueError(f"{path} has different settings. Use a new experiment directory.")


def read_rows(path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc


def prepare(args):
    cases = []
    for row in read_rows(args.holdout):
        if row.get("dataset", row.get("source_dataset")) != "SWE-chat":
            continue
        if row.get("split", "test") != "test":
            raise ValueError("Expected a held-out split, but found a non-test SWE-chat row.")
        opening = []
        for wrapper in row["events"]:
            event = wrapper.get("source_event", wrapper)
            if event.get("role") == "user" and event.get("is_conversational"):
                if event.get("content", "").strip():
                    opening.append({"role": "user", "content": event["content"]})
            elif opening:
                break
        if not opening:
            raise ValueError(f"{row['example_id']} has no usable opening request.")
        cases.append({"case_id": row["example_id"], "initial_messages": opening})
    if not cases:
        raise ValueError("The supplied split has no SWE-chat scenarios.")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate case IDs in the supplied split.")
    save_once(args.cases, cases, lines=True)
    print(f"Wrote {len(cases)} opening requests to {args.cases}.")


def text_field(value):
    return isinstance(value, str) and bool(value.strip())


def validate_messages(messages, roles):
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected a nonempty list of messages.")
    for message in messages:
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] not in roles or not text_field(message["content"])):
            raise ValueError("Messages must contain only role and nonempty content.")


def load_cases(path):
    cases = {}
    for case in read_rows(path):
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", case_id):
            raise ValueError("Invalid case ID.")
        if case_id in cases:
            raise ValueError(f"Duplicate case ID {case_id}.")
        validate_messages(case.get("initial_messages"), {"user"})
        if set(case) - {"case_id", "initial_messages", "runner_setup"}:
            raise ValueError(f"{case_id}: unexpected case fields.")
        if not isinstance(case.get("runner_setup", {}), dict):
            raise ValueError(f"{case_id}: runner_setup must be an object.")
        cases[case_id] = case
    if not cases:
        raise ValueError("No cases found.")
    return cases


def validate_session(session, case):
    if not isinstance(session, dict) or set(session) != {
        "transcript", "tool_calls", "commit_evidence"
    }:
        raise ValueError("A session requires transcript, tool_calls, and commit_evidence fields.")
    transcript = session["transcript"]
    validate_messages(transcript, {"user", "assistant"})
    opening = case["initial_messages"]
    if transcript[:len(opening)] != opening:
        raise ValueError("The transcript must start with the case's initial messages.")
    for name, keys in (("tool_calls", {"name", "category"}),
                       ("commit_evidence", {"message", "diff_summary"})):
        rows = session[name]
        if not isinstance(rows, list):
            raise ValueError(f"{name} must be a list.")
        for row in rows:
            if (not isinstance(row, dict) or set(row) != keys
                    or not all(text_field(value) for value in row.values())):
                raise ValueError(f"{name} entries require nonempty fields {', '.join(sorted(keys))}.")


def experiment(args):
    return args.run_dir / "experiment.json"


def load_experiment(args):
    manifest = read(experiment(args))
    cases = load_cases(args.cases)
    if fingerprint(cases) != manifest["cases_sha256"]:
        raise ValueError("Case inputs differ from the frozen experiment.")
    return manifest, cases


def record_path(args, stage, case_id, condition):
    return args.run_dir / stage / case_id / f"{condition}.json"


def run_adapter(command, case, model, timeout):
    request = {"case": case, "model": model}
    try:
        completed = subprocess.run(
            command, input=json_text(request), encoding="utf-8",
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"Runner timed out after {timeout}s.") from exc
    if completed.returncode:
        raise ValueError(f"Runner exited with code {completed.returncode}. Check runner logs.")
    try:
        session = json.loads(completed.stdout)
    except ValueError as exc:
        raise ValueError("Runner stdout was not a JSON session record.") from exc
    validate_session(session, case)
    return session


def generate(args):
    cases = load_cases(args.cases)
    if not args.runner or args.base == args.sft:
        raise ValueError("Supply a runner command and two different checkpoint IDs.")
    models = {"base": args.base, "sft": args.sft}
    manifest = {
        "cases_sha256": fingerprint(cases), "models": models,
        "runner": args.runner, "timeout": args.timeout, "trial_config": args.trial_config,
    }
    freeze(experiment(args), manifest)
    failures = 0
    for case_id, case in cases.items():
        for condition in CONDITIONS:
            path = record_path(args, "sessions", case_id, condition)
            try:
                if path.exists():
                    validate_session(read(path), case)
                    continue
                session = run_adapter(args.runner, case, models[condition], args.timeout)
                save_once(path, session)
                print(f"Saved {case_id} {condition} session.")
            except (ValueError, OSError) as exc:
                failures += 1
                print(f"{case_id} {condition}: {exc}", file=sys.stderr)
    return bool(failures)


def judge_input(session):
    calls = session["tool_calls"]
    names = Counter(call["name"] for call in calls)
    top_tools = sorted(names.items(), key=lambda entry: (-entry[1], entry[0]))[:10]
    return {
        "transcript": session["transcript"],
        "tool_summary": {
            "category_counts": dict(Counter(call["category"] for call in calls)),
            "top_tools": dict(top_tools),
        },
        "commit_evidence": session["commit_evidence"],
    }


def grade_api(model, prompt, evidence, key, timeout):
    request = Request(
        "https://api.anthropic.com/v1/messages",
        data=json_text({
            "model": model,
            "max_tokens": 1024,
            "temperature": 0,
            "system": prompt,
            "messages": [{"role": "user", "content": json_text(evidence)}],
        }).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except HTTPError as exc:
        raise ValueError(f"Judge API returned HTTP {exc.code}.") from exc
    except (URLError, TimeoutError) as exc:
        raise ValueError("Judge API connection failed or timed out.") from exc
    if payload.get("stop_reason") != "end_turn":
        raise ValueError(f"Judge did not complete its response ({payload.get('stop_reason')}).")
    try:
        parsed = json.loads("".join(
            block["text"] for block in payload["content"] if block["type"] == "text"
        ))
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError("Judge response was not a JSON score.") from exc
    validate_grade(parsed)
    return parsed, payload.get("model", model), payload.get("id")


def validate_grade(parsed):
    if (not isinstance(parsed, dict) or set(parsed) != {"score", "reason"}
            or type(parsed["score"]) is not int or not 0 <= parsed["score"] <= 100
            or not text_field(parsed["reason"])):
        raise ValueError("A judge grade requires an integer score from 0 to 100 and a reason.")


def existing_grade(path, session, config):
    result = read(path)
    if (result["session_sha256"] != fingerprint(session)
            or result["judge_sha256"] != fingerprint(config)):
        raise ValueError(f"Stale grade at {path}. Use a fresh experiment directory.")
    validate_grade(result["result"])
    return result["result"]["score"]


def grade(args):
    _, cases = load_experiment(args)
    prompt = args.rubric.read_text(encoding="utf-8")
    if not prompt.strip():
        raise ValueError("The judge rubric is empty.")
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Set ANTHROPIC_API_KEY to use the live judge.")
    config = {"model": args.judge_model, "rubric_sha256": fingerprint(prompt)}
    freeze(args.run_dir / "judge.json", config)
    freeze(args.run_dir / "rubric.json", {"prompt": prompt})
    failures = 0
    for case_id, case in cases.items():
        for condition in CONDITIONS:
            source = record_path(args, "sessions", case_id, condition)
            destination = record_path(args, "grades", case_id, condition)
            try:
                session = read(source)
                validate_session(session, case)
                if destination.exists():
                    existing_grade(destination, session, config)
                    continue
                result, model, request_id = grade_api(
                    args.judge_model, prompt, judge_input(session), key, args.timeout,
                )
                save_once(destination, {
                    "session_sha256": fingerprint(session), "judge_sha256": fingerprint(config),
                    "result": result, "response_model": model, "response_id": request_id,
                })
                print(f"Graded {case_id} {condition}.")
            except (ValueError, OSError) as exc:
                failures += 1
                print(f"{case_id} {condition}: {exc}", file=sys.stderr)
    return bool(failures)


def summarize(args):
    _, cases = load_experiment(args)
    config = read(args.run_dir / "judge.json")
    pairs = []
    incomplete = []
    for case_id, case in cases.items():
        scores = {}
        for condition in CONDITIONS:
            saved_grade = record_path(args, "grades", case_id, condition)
            saved_session = record_path(args, "sessions", case_id, condition)
            if not saved_grade.exists() or not saved_session.exists():
                continue
            session = read(saved_session)
            validate_session(session, case)
            scores[condition] = existing_grade(saved_grade, session, config)
        if len(scores) == len(CONDITIONS):
            pairs.append({"case_id": case_id, **scores, "delta": scores["sft"] - scores["base"]})
        else:
            incomplete.append({"case_id": case_id, "available_scores": scores})
    deltas = [pair["delta"] for pair in pairs]
    print(json.dumps({
        "expected_pairs": len(cases), "complete_pairs": len(pairs),
        "mean_base": statistics.mean(pair["base"] for pair in pairs) if pairs else None,
        "mean_sft": statistics.mean(pair["sft"] for pair in pairs) if pairs else None,
        "mean_sft_minus_base": statistics.mean(deltas) if deltas else None,
        "sft_wins": sum(delta > 0 for delta in deltas),
        "ties": deltas.count(0), "sft_losses": sum(delta < 0 for delta in deltas),
        "pairs": pairs, "incomplete_pairs": incomplete,
    }, indent=2))
    return bool(incomplete)


def positive_seconds(value):
    seconds = int(value)
    if seconds < 1:
        raise argparse.ArgumentTypeError("Timeout must be positive.")
    return seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    options = stages.add_parser("prepare", help="Export opening prompts from a curated test split.")
    options.add_argument("--holdout", type=Path, required=True)
    options.add_argument("--cases", type=Path, required=True)
    options.set_defaults(action=prepare)

    for name, action in (("generate", generate), ("grade", grade), ("summarize", summarize)):
        options = stages.add_parser(name)
        options.add_argument("--cases", type=Path, required=True)
        options.add_argument("--run-dir", type=Path, required=True)
        options.set_defaults(action=action)
        if name == "generate":
            options.add_argument("--base", required=True)
            options.add_argument("--sft", required=True)
            options.add_argument("--trial-config", required=True,
                                 help="Version or digest of the shared runner configuration.")
            options.add_argument("--timeout", type=positive_seconds, default=1800)
            options.add_argument("--runner", nargs=argparse.REMAINDER, required=True,
                                 help="Trusted runner executable and arguments. Place this flag last.")
        if name == "grade":
            options.add_argument("--judge-model", default=JUDGE_MODEL)
            options.add_argument("--rubric", type=Path, default=RUBRIC)
            options.add_argument("--timeout", type=positive_seconds, default=120)
    args = parser.parse_args(argv)
    try:
        return int(bool(args.action(args)))
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
