# Midway annotation workflow

The current Midway Annotation Sample contains six full released threads: a bridal-sales pair, a UAT-plan pair and a SWE-chat code-review pair. Each pair includes one single-turn and one multi-turn example. The CollabSkill pairs share assigned task IDs. SWE-chat examples are thematically similar across projects and agents, not a controlled same-task comparison.

## Working independently

1. Download the midway package using the repository's pinned downloader.
2. Create your named file with `python annotation_tools.py create --annotator YOUR_ID`.
3. Read each complete thread and relevant tool events through `curated/midway-annotation-sample/practice_threads.html`.
4. Record the applicable boundary in your own words. Consider subsequent user instructions and when they take effect.
5. Assign the provisional adherence value only when the evidence supports assessment. Cite original zero-based event indices separated by semicolons, such as `3;39;88`. Explain uncertainty and missing context.
6. Validate your draft file. Mark finished cases `complete` or `not_assessable` before running `--require-complete`.
7. Keep your judgments separate from others until the group discussion. Submission and adjudication instructions will follow.

The schema is [annotation_schema.json](annotation_schema.json). Its CSV field remains `boundary_adherence`; the project's concept is delegation adherence. The draft values are `full`, `partial` and `none`. The team has not finalized operational definitions, extra dimensions, numeric scoring, confidence scales, assignments or an agreement metric. Do not treat this starter as a finalized rubric.

## What the checks require

Each working CSV contains the same six practice/scenario IDs and one annotator ID. Scores must use the draft vocabulary. A scored judgment needs a boundary description, rationale and valid event evidence. A completed judgment needs a rationale. A `not_assessable` case has a blank adherence score and an explanation in `missing_context` or `rationale`. A completed case with no applicable boundary can leave its boundary and score blank and explain that in its rationale.

Draft validation allows `not_started` and `in_progress`. `--require-complete` requires every row to be `complete` or `not_assessable`. Validation does not determine whether an adherence judgment is correct or whether the rubric is adequate.

## Evidence limits

The packet preserves all 233 released events. It does not independently establish completeness of the original interaction. Referenced instructions, attachments, generated deliverables or workspaces may be missing. The SWE-chat single-turn release contains tool calls without separate tool-result events. Record these limitations rather than inferring unavailable evidence.

Reading views escape source text and redact common contact and credential patterns. Exact source JSONL/metadata remain separate. Use the reading views during annotation and avoid source-generated outcome scores until independent annotations are complete. An assistant's claim that an artifact is correct is not independent verification.

Annotations belong in separate files under `annotations/`, not inside the downloaded package. The downloader preserves conflicting local files, and rerunning it can verify unchanged templates without touching your working CSVs. The package is for exposed midway practice, not a blind final test set.
