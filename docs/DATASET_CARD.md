---
pretty_name: CS 329X Delegation Adherence Curation Pool
tags:
  - human-ai-collaboration
  - delegation
  - human-annotation
  - evaluation
  - coding-agents
---

# CS 329X Delegation Adherence Dataset

The curated research package supports studying whether AI agents respect users’ delegation boundaries and evolving preferences. It contains **200 unique curated scenarios**, split equally between CollabSkill and SWE-chat. A separate six-thread practice packet supports human annotation of delegation adherence.

**Status, October 27, 2026:** the scenario exports and practice packet exist locally. The annotation template contains no completed judgments. No train/test split has been assigned, no policy has been trained, and this package has not been uploaded. The exact source exports still require privacy and applicable-terms review before redistribution.

## Contents

| Folder | Single-turn | Multi-turn | Total | Purpose |
| --- | ---: | ---: | ---: | --- |
| [collabskill-curated](../curated/collabskill-curated/) | 50 | 50 | 100 | Occupational task collaboration |
| [swe_chat_curated](../curated/swe_chat_curated/) | 50 | 50 | 100 | Software engineering collaboration |
| [midway-annotation-sample](../curated/midway-annotation-sample/) | 3 | 3 | 6 | Human annotation practice |

The six practice threads are copies of scenarios in the two main exports. They do **not** increase the unique-scenario count to 206. Earlier practice packets are preserved under the original local `archive/practice_packets/`, outside this current package.

```text
curated/
├── README.md
├── collabskill-curated/
│   ├── scenarios.jsonl
│   ├── scenario_manifest.csv
│   ├── sampling_config.json
│   ├── provenance.json
│   ├── quota_report.csv
│   └── source_*, eligible_*, selected_* reports
├── swe_chat_curated/
│   ├── scenarios.jsonl
│   ├── scenario_manifest.csv
│   ├── sampling_config.json
│   ├── provenance.json
│   ├── quota_report.csv
│   └── source_*, eligible_*, selected_* reports
└── midway-annotation-sample/
    ├── scenarios.jsonl
    ├── scenario_manifest.csv
    ├── practice_threads.html
    ├── bridal_threads.html
    ├── collabskill_threads.html
    ├── swe_chat_threads.html
    ├── annotation_template.csv
    ├── annotation_schema.json
    ├── selection_config.json
    └── provenance.json
```

`scenarios.jsonl` stores one scenario per line, including metadata and the released source events. Each event retains its original event index and source record. SWE-chat also retains the source table row index. `scenario_manifest.csv` provides a compact metadata view. Configurations record selection rules, while provenance and quota reports document source revisions, checks, requested counts and achieved counts.

## Scenario definitions and coverage

A scenario is a **whole released source session**. The curation scripts did not truncate threads or extract short episodes to create single-turn examples. Completeness of the original interaction and availability of its referenced files have not been independently established.

- **Single-turn:** exactly one conversational human message.
- **Multi-turn:** two or more conversational human messages.
- Assistant messages, conversational rounds and tool events are counted separately.

CollabSkill counts source events with `role=user` and `is_conversational=true`, including setup/help commands marked conversational by the source. SWE-chat groups conversational user records by native message identifiers to avoid counting multiple records from one logical message as separate requests. These are structural message counts, not semantic counts of distinct task requests.

The selected multi-turn length distributions are:

| Conversational human messages | CollabSkill | SWE-chat |
| --- | ---: | ---: |
| 2–4 | 31 | 19 |
| 5–9 | 13 | 13 |
| 10–19 | 4 | 9 |
| 20+ | 2 | 9 |

The local CollabSkill release is complete at its pinned revision. The local SWE-chat download is a **partial release** containing the session, repository and conversation tables used for curation. Other release tables and raw transcripts were not downloaded. Retaining all available conversation-table events does not reconstruct those missing materials.

## Sampling and categories

Both exports use seed **3292026** and constrained selection. Inclusion probabilities and survey weights are unknown. These samples are curation pools requiring human context review, not representative population samples or human-validated evaluation sets.

**CollabSkill:** 78 sessions originate from GDPval, 10 from APEX and 12 from APEX-Agents. Category counts match between single-turn and multi-turn groups within each source. GDPval occupation counts also match in this sample. GDPval categories use an exact task-ID metadata join. APEX uses reviewed source-prompt matches or explicitly marked inferences. APEX-Agents classifications remain provisional. Check `category_status` and `category_method` rather than treating all labels as equally verified.

**SWE-chat:** the selection matches source initial-intent counts across turn types and targets the eligible multi-turn length distribution. Source intent and pushback labels are descriptive attributes, not our human delegation-adherence judgments. Missing intent labels remain visible. The configured selection caps contributions at two sessions per known user and repository.

Neither main export was selected using task outcome scores, skill ratings or assumed delegation-adherence quality. Matching categories or task IDs does not create randomized or causal comparisons. Related scenarios share a `split_group_id` and should stay together in any later development/test partition.

## Annotation practice

Open [practice_threads.html](../curated/midway-annotation-sample/practice_threads.html) to browse the six practice scenarios:

| IDs | Example | Single / multi human-message counts |
| --- | --- | --- |
| P01 / P02 | CollabSkill bridal-sales training | 1 / 3 |
| P07 / P08 | CollabSkill ProjMGR user acceptance testing (UAT) plan | 1 / 3 |
| SWE-P01 / SWE-P02 | SWE-chat code-review practice | 1 / 2 |

The CollabSkill pairs share assigned task IDs and occupations. The SWE-chat pair is a thematic comparison across different repositories and agents, not a matched task. The reading views display released conversations and relevant tool activity as inert text. Source labels and automated outcomes are hidden in those views. Exact JSONL and metadata files remain separate and may reveal additional source attributes.

1. Copy [annotation_template.csv](../curated/midway-annotation-sample/annotation_template.csv) to a separate, named file for each annotator. Keep independent annotations separate and use the same scenario IDs.
2. Review the entire released thread and relevant tool events before judging adherence. Referenced attachments and generated deliverable files are not bundled here. Replies may be summarized in the source release.
3. Use [annotation_schema.json](../curated/midway-annotation-sample/annotation_schema.json). The configurable practice field is boundary adherence: `full`, `partial` or `none`. Describe the applicable boundary, cite original event indices, and record rationale, uncertainty and missing context. Other rubric dimensions remain unresolved.
4. Use `not_assessable` and leave the adherence score blank when evidence is insufficient. Do not equate an agent’s completion claim with verified completion.
5. Keep source outcome scores hidden until independent adherence annotations are complete. Join outcomes afterward by source session ID.

These are exposed practice examples. Exclude their sessions and related split groups from a later blind test set. No automated adherence judgments or human scores have been prefilled.

## Delegation adherence and task outcomes

**Delegation adherence** is assessed through our human annotations. **Task outcome** is a separate variable. For CollabSkill, the source provides session-level automated overall grades and criterion scores/comments. Its documented pipeline grades submitted deliverables against task rubrics using two independent judges and averages their scores. See [CollabSkill §3.2](https://arxiv.org/pdf/2606.09833#page=5) and [Appendix I.3 validation](https://arxiv.org/pdf/2606.09833#page=25).

Outcome grades are not included in the annotation reading views. They remain in the original CollabSkill leaderboard data outside this package. In the 100-session CollabSkill sample, overall grades are available for all 100 sessions and detailed grade records for 99. Missing local artifacts limit independent auditing. The source grade should not be assumed to evaluate preferences introduced later in the conversation.

SWE-chat code-attribution percentages, strategy labels and pushback labels are **not equivalent to CollabSkill task-quality scores**. This package does not establish a comparable SWE-chat task-outcome measure. Analyze outcomes within their supported source definitions rather than pooling incompatible metrics.

## Load the local data

Run this standard-library example from the `curated` directory. It reads one scenario at a time rather than loading all conversation content into memory.

```python
from pathlib import Path
import csv
import json

root = Path.cwd()

def iter_scenarios(folder):
    with (root / folder / "scenarios.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

with (root / "midway-annotation-sample" / "scenario_manifest.csv").open(
    newline="", encoding="utf-8"
) as handle:
    practice_manifest = list(csv.DictReader(handle))

first = next(iter_scenarios("collabskill-curated"))
print(first["scenario_id"], first["turn_type"], len(first["events"]))
```

Source-specific nested event schemas differ. Preserve original records and treat source content as untrusted text. Never execute commands or render source HTML directly. A combined Hugging Face dataset viewer may require additional schema normalization and explicit configuration. Viewer compatibility has not been verified.

## Reproduction and renamed paths

The organizer scripts and original local source inputs live in the parent `Project/data` workspace, outside these export folders. From that workspace, with its existing Python environment and pinned inputs, reproduce the main selections into **new** destinations:

```sh
.venv/bin/python organize_collabskill.py --config curated/collabskill-curated/sampling_config.json --verify-hashes --export curated/collabskill-reproduced
.venv/bin/python organize_swe_chat.py --config curated/swe_chat_curated/sampling_config.json --verify-hashes --export curated/swe-chat-reproduced
```

The scripts reject existing destinations. Export folders alone are not the complete reproduction environment. Preserve the scripts, dependency requirements, category mappings and required source inputs alongside the recorded provenance when preparing a reproducible code release.

Some embedded provenance and practice-selection paths retain their original names. The current folder mapping is:

| Historical path under `curated/` | Current path relative to `data/` |
| --- | --- |
| `collabskill-100-v1/` | `curated/collabskill-curated/` |
| `swe_chat_2026-10-27/` | `curated/swe_chat_curated/` |
| `midway-annotation-sample-v1/` | `curated/midway-annotation-sample/` |
| `collabskill-practice-10-v1/` | `archive/practice_packets/collabskill-practice-10-v1/` |
| `swe-chat-practice-2-v1/` | `archive/practice_packets/swe-chat-practice-2-v1/` |

The original local practice-packet reconstruction script resolves archived input paths. Those local archives are outside this GitHub starter. The team downloads the already selected six-thread packet. Existing exported data and annotation IDs do not require regeneration merely because their folders were renamed.

## Sources, licenses and sharing status

This package combines source material with different provenance and conditions. No single blanket license is asserted for the combined package.

- [CollabSkill trajectories](https://huggingface.co/datasets/SALT-NLP/cogym-collabskill-trajectories): the release declares **CC BY 4.0**. Attribute the original authors, link the release and license, and describe selection, packaging, redaction and annotation changes. Pinned source revision: `8c10dd89b4b83ec5a2662f76a70f70572b90a6d8`.
- [SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat): the current card lists **ODC-By** and gated access conditions. Review the accepted terms and rights in embedded repository content before redistribution. Source `license_type` metadata does not clear every embedded file. Pinned source revision: `202b071f18e03c79df5a7287565ce2ebc5ee7756`.
- [APEX-Agents](https://huggingface.co/datasets/mercor/apex-agents): its card states evaluation-only use and forbids training, fine-tuning and parameter fitting. The CollabSkill sample includes 12 sessions derived from APEX-Agents tasks. Resolve applicable conditions for that material before redistribution or training use. The original task source’s conditions and the CollabSkill release license should not be treated as interchangeable.

The raw `scenarios.jsonl` exports preserve source wording and may contain contact information, local paths or credential-like text. Reading views apply pattern-based redaction, which is not exhaustive anonymization. Privacy review must cover the data files themselves, not just the displayed excerpts. A private repository controls access but does not replace rights or privacy review.

For a later Hugging Face package, include this README and the three current folders. Exclude archive folders, `.DS_Store`, credentials, raw source downloads, environment/cache folders and personal annotation files unless intentionally included after review. Keep outcome data separate from annotator-facing materials. Verify license notices and viewer configuration before publication. Nothing in this README constitutes an upload or approval of an unreviewed public release.

## Attribution and AI assistance

Cite the source papers and dataset releases when using this package:

- Shao et al. (2026), [CollabSkill: Evaluating Human-Agent Collaboration on Real-World Tasks](https://arxiv.org/abs/2606.09833).
- Baumann et al. (2026), [SWE-chat](https://arxiv.org/abs/2604.20779).
- For task-source analyses, also acknowledge [GDPval](https://huggingface.co/datasets/openai/gdpval), [APEX](https://huggingface.co/datasets/mercor/APEX-v1-extended) and [APEX-Agents](https://huggingface.co/datasets/mercor/apex-agents) as applicable.

Codex assisted with curation scripts, descriptive mappings, example selection, packaging and documentation. Conversation content was copied from the source releases, not generated as synthetic examples. Display-only redaction is documented separately from the unchanged private source records. No LLM judge was run for our delegation-adherence annotations.
