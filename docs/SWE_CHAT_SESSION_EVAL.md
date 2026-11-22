# SWE-chat session-success evaluation

Compare a base checkpoint and its SFT version on the same held-out tasks. The
grading stage follows the [SWE-chat v2 session-success procedure](https://arxiv.org/html/2604.20779v2#A6.SS2.SSS5).
The included prompt paraphrases its five dimensions and six scoring bands. The
default judge is Claude Sonnet 4.6. One API call grades each new session
independently. Ten cases with both runs available take 20 grading calls before
retries. This workflow adds paired session generation to the paper's grading
procedure for historical sessions.

Run from the repository root with Python 3.11 or later. The evaluator uses the
standard library.

## 1. Prepare cases

```sh
run_dir=outputs/model_results/swe-chat-session-v1
python -m data.evaluation.swe_chat prepare \
  --holdout outputs/curation-v2/swechat-v2/test.jsonl \
  --cases "$run_dir/cases.jsonl"
```

This uses the private v2 test split. It extracts the first block of conversational
user messages and assigns the existing example ID. Historical responses, tool
results, and outcome labels are excluded. Check the openings before generation.
If a case needs a file or repository state, add a `runner_setup` object with
the references your runner expects. Pin the pre-task revision where available.
Use the same setup for both runs. Record cases that lack essential context and
report them as unavailable if they cannot be run. Keep task inputs and run
records in the ignored results directory.

## 2. Generate and save paired sessions

```sh
python -m data.evaluation.swe_chat generate \
  --cases "$run_dir/cases.jsonl" --run-dir "$run_dir" \
  --base 'BASE_CHECKPOINT_ID' --sft 'SFT_CHECKPOINT_ID' \
  --trial-config 'SHARED_RUNNER_CONFIG_VERSION_OR_DIGEST' \
  --runner python /absolute/path/to/your_runner.py
```

Supply the command that runs your checkpoints. The evaluator calls it once per
case per checkpoint, using the same case object. The runner reads one JSON
object from stdin with `case` (containing `case_id`, `initial_messages`, and
optional `runner_setup`) and `model`. It writes one JSON object to stdout
after the interaction ends:

```json
{
  "transcript": [
    {"role": "user", "content": "Explain how this module loads preferences."},
    {"role": "assistant", "content": "The settings loader parses the project configuration first."}
  ],
  "tool_calls": [{"name": "read_file", "category": "read"}],
  "commit_evidence": []
}
```

- Echo the initial messages unchanged and append only the turns from this run.
  Use a shared human-user protocol or simulator that responds to each model's
  actual actions in multi-turn cases.
- Keep environment, harness, tools, user protocol, generation settings and
  stopping rules fixed across checkpoints. Start each run from its own clean
  workspace. The evaluator records the runner command and trial config ID.
  The runner applies and enforces the config.
- Provide one `tool_calls` entry per observed call. For commit evidence,
  use objects with `message` and `diff_summary` strings. Use empty lists
  when no such evidence exists. Keep terminal logs and raw artifacts in the
  runner's own output storage. Where the user and assistant discussed test
  results, include those turns in the transcript.
- Exit nonzero for infrastructure failures and keep diagnostic logs in a local
  file. A session that ends with a failed task can still produce a valid record.
  The evaluator stores only well-formed records and resumes from missing ones.

The evaluator passes arguments directly to the runner process. Use a trusted
runner and keep its credentials in the environment. The project currently
does not include a checkpoint-serving agent harness or a user simulator.

## 3. Grade and compare

Provide `ANTHROPIC_API_KEY` in the environment. The grade command sends each
fresh session's transcript, derived tool summary, and available commit evidence
to the Anthropic Messages API. It leaves out condition labels, checkpoint IDs,
case IDs, runner setup and original SWE-chat grades. Content inside the
conversation may itself reveal the agent's identity.

```sh
python -m data.evaluation.swe_chat grade \
  --cases "$run_dir/cases.jsonl" --run-dir "$run_dir"

python -m data.evaluation.swe_chat summarize \
  --cases "$run_dir/cases.jsonl" --run-dir "$run_dir"
```

The optional `--judge-model` and `--rubric` flags set the judge and prompt.
Choose them on development sessions. The first grading run freezes those
settings for the experiment. The evaluator checks each 0–100 integer grade
and reason before saving them, and ties each grade to a hash of the session
and judge settings. Grades and session records are stored per case and
condition. On failure rerun the same command to fill in missing records.
Use a fresh experiment directory if model, cases, runner settings, or judge
change. Run one instance per experiment directory to avoid duplicate calls.

The summary prints every complete pair, its SFT-minus-base difference,
mean differences, wins, ties, losses, and incomplete pairs. It exits
nonzero if pairs are missing. The ten-case holdout is appropriate for a
pilot. Treat live judge ratings as model judgments, and inspect the
evidence and reasons before interpreting differences.
