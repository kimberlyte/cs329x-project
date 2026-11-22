# CollabSkill task-outcome evaluation

Compare the base and SFT checkpoints on the same held-out tasks. Each produces a
fresh submission. Gemini CLI and Claude Code independently inspect each
submission against the same frozen rubric. The mean of their totals is the
submission's 0–100 grade. This follows the paper's artifact grading method and
adds a paired checkpoint comparison. It does not compute the paper's Bayesian
skill rankings. [CollabSkill §3.2 and Appendix D](https://arxiv.org/pdf/2606.09833#page=5)

## 1. Assemble the inputs

Run from the repository root with Python 3.11 or later, authenticated versions of
Gemini CLI and Claude Code, and the command that serves your two checkpoints.
Pin judge model IDs and record CLI versions before the pilot.

```sh
split=outputs/curation-v2/collabskill-v2/test.jsonl
tasks=outputs/model_results/collabskill-tasks
run_dir=outputs/model_results/collabskill-paired-v1
```

Use the private test split. Create one folder under `$tasks` for each
`example_id` in that split. Put the original assigned task text in
`prompt.txt` and the original source files under `references/`. The test split
selects the cases. Historical interactions, filenames, and scores are not used
as model inputs. Resolve the task mapping using the curator's source IDs and
retrieve the prompt and files from the originating task benchmark. The local
trajectory export does not contain the full set of original task inputs.
Keep inputs and run results under the ignored `outputs/model_results/` tree.

Optionally include a **full** `rubric.json` in a case folder. Use the Appendix D
format with `total_points`, `categories` and optional `critical_requirements` and
`evaluation_notes`. Each category needs `name`, `max_points` and `criteria`.
Each criterion needs `description`, `points`, `expected_value` and
`evaluation_method`. The local CollabSkill release has category summary
records, which do not contain the complete criterion-level rubric. Leave
`rubric.json` out for those cases. The grader will generate and freeze a
rubric from the task and references using Gemini CLI, with 40 points for
correctness, 20 for completeness, 25 for requirements, and 15 for technical
quality. Submission files are absent from the rubric-generation workspace.

## 2. Generate both submissions

```sh
python -m data.evaluation.collabskill generate \
  --holdout "$split" --tasks "$tasks" --run-dir "$run_dir" \
  --base 'BASE_CHECKPOINT_ID' --sft 'SFT_CHECKPOINT_ID' \
  --trial-config 'SHARED_RUNNER_CONFIG_VERSION_OR_DIGEST' \
  --runner python /absolute/path/to/your_runner.py
```

The evaluator starts the trusted runner in a fresh workspace for each checkpoint
and reads its exit status. The runner reads this JSON object from stdin:

```json
{"case_id":"DEMO","model":"CHECKPOINT_ID","task":"prompt.txt","references":"references","output":"deliverables"}
```

Read the task and references from those relative paths. Serve the specified
checkpoint using the same environment, tools, budget and user protocol across
conditions. Save all final files under `deliverables/`. Exit zero when the
interaction ends, even if it left required work incomplete. Exit nonzero on
infrastructure failure. Keep execution logs outside the deliverables folder.
The repository does not include a checkpoint-serving agent or user simulator.

To grade submissions from an existing experiment, populate
`$run_dir/submissions/<example_id>/base/` and
`$run_dir/submissions/<example_id>/sft/` with the final artifacts. Create empty
directories for runs that ended with no files. A missing directory represents
an unavailable run.

## 3. Grade and summarize

```sh
python -m data.evaluation.collabskill grade \
  --holdout "$split" --tasks "$tasks" --run-dir "$run_dir" \
  --gemini-model 'GEMINI_MODEL_ID' --claude-model 'CLAUDE_MODEL_ID'

python -m data.evaluation.collabskill summarize \
  --holdout "$split" --tasks "$tasks" --run-dir "$run_dir"
```

Each judge sees the task, shared rubric, references and a copy of one submission
in a fresh workspace. The invocation omits case IDs, checkpoint IDs and
historical scores. The files themselves can disclose the model's identity.
Use only task files and deliverables you can share with both providers.

The judges can read supported files directly. For DOCX, XLSX and PPTX, the
evaluator also supplies indices and raw Office package parts for inspecting
content, structure and formulas. This does not execute macros, recalculate
formulas, or visually render Office documents. A criterion requiring a check
the judge cannot perform is unassessable. Judges can award zero when a
requirement is demonstrably unmet. Review criteria that need rendering or a
specialized format before selecting a pilot task.

Rubrics and criterion-level JSON are stored under `$run_dir`. The evaluator
validates the full set of criterion IDs and point limits, computes the
judge totals, and requires both judges to provide assessable grades for both
submissions before including a pair. The summary reports each judge score,
both averages, the paired SFT-minus-base delta, win/tie/loss counts, and
incomplete pairs. It exits nonzero if any pairs are incomplete. Ten fully
graded tasks require 40 submission-grading calls and up to 10 additional rubric
generation calls.

Run one process per experiment directory. Rerun the same commands to resume
after a transient failure. Use a new experiment directory when changing
inputs, checkpoints, judges, or an already graded submission. Treat the small
holdout as a pilot and review the per-criterion reasons alongside the totals.
