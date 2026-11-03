# Existing source outcomes

The consolidated [dataset review notebook](../data/analysis/dataset_review.ipynb) shows score coverage, source-versus-curated distributions, quality components and session-success bands. Its aggregate report is [outcome_summary.json](../data/analysis/reports/outcome_summary.json). No per-session scores, source comments, user IDs or human judgments are published in that report.

## CollabSkill task quality

`autograde_overall_score` is the source session’s 0–100 deliverable grade. Join `sessions.autograde_score_id` to `scores._id` for `overall_score` and `detailed_scores`, which contain criterion names, points and comments. These are session outcomes, distinct from the Bayesian human/agent skill ratings.

The usual top-level criteria are:

| Criterion | Usual maximum | Meaning |
| --- | ---: | --- |
| Correctness & Accuracy | 40 | Accuracy of facts, calculations, logic and source use |
| Deliverable Completeness | 20 | Presence of required files, sections and components |
| Task Requirements | 25 | Compliance with the assigned task’s instructions and constraints |
| Technical Quality | 15 | Formatting, organization, usability and technical execution |

The source uses generated task-specific rubrics and averages two independent agent judges’ totals. Of 383 detailed score records in the pinned release, 375 use the standard four names. Some records use custom criteria, and 25 of 165 task records have empty rubric lists. Do not assume every detailed record has the same components or normalize a custom rubric using these default weights.

Overall grades exist for all 386 source sessions and all 100 selected sessions. Detailed records exist for 383 source sessions and 99 selected sessions. Missing detailed records do not imply a zero overall grade. The grade evaluates submitted work against the assigned task and may miss preferences introduced later during interaction.

Source: [CollabSkill §3.2](https://arxiv.org/html/2606.09833v2#S3.SS2).

## SWE-chat session success

`session_success` is an existing LLM-assigned 0–100 score, serialized as text. The evaluation considers goal completion, how the session ended, efficiency, code/commit quality and user experience. Its released session table contains one overall score, without five separate numerical subscores.

The source rubric’s bands, paraphrased:

| Score | Interpretation |
| --- | --- |
| 90–100 | All requests resolved; clean ending |
| 70–89 | Core goal met; minor issues remain |
| 50–69 | Meaningful progress; important unfinished work or defects |
| 30–49 | Relevant output; primary goal unmet |
| 10–29 | Very little useful work |
| 0–9 | No useful progress |

The evaluator receives conversation, tool summaries and available commit evidence. This is a transcript-based judgment rather than an independently executed correctness test. Session scores exist for 17,563 of 17,968 source sessions and 99 of 100 selected sessions. Nonempty `summary_outcome` text exists for 732 source sessions and three selected sessions.

The source prompt requests integers, but fractional values occur in the downloaded release. Preserve them without rounding or rescaling. Derived bands use [0,10), [10,30), [30,50), [50,70), [70,90), [90,100]. Missing and invalid scores are counted separately.

Sources: [SWE-chat Appendix F.2.5](https://arxiv.org/html/2604.20779v2#A6.SS2.SSS5), [released schema](https://huggingface.co/datasets/SALT-NLP/SWE-chat#dataset-schema).

## Use with independent adherence labels

Keep per-session scores outside annotator-facing reading views. After independent adherence annotations are complete, join outcomes by source `session_id`. Do not infer adherence from task success or fill missing outcomes with zero.

Analyze the datasets separately. The two 0–100 scales reflect different evaluations. SWE-chat’s success score already considers correction effort and interaction friction. Associations with adherence can therefore partly reflect overlap in the criteria. CollabSkill’s Task Requirements component can also overlap with instruction compliance. These are observational associations, not causal effects.

The ten-point histogram bins shown for both datasets are plotting intervals. Only SWE-chat’s six semantic bands come from its source rubric.

## Rebuild the aggregate report

With the pinned raw source tables and curated manifests available under a data root:

```sh
python -m data.analysis.outcomes --source-root data --output outputs/outcome_summary_NEW.json
```

Use `--curated-root PATH` if the saved packages are elsewhere. The command reads source metadata and scores, not conversation text. It rejects an existing output path and leaves source files, sample membership, annotation packets and working judgments unchanged. Input hashes and source revisions are recorded. Review a new report before replacing the tracked snapshot.
