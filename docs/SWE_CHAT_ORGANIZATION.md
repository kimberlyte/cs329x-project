# SWE-chat organization and sampling review

> **Notebook consolidation, October 27, 2026:** Use `dataset_statistics.ipynb` for source data and cross-dataset comparisons, and `sampling_review.ipynb` for both saved curated samples. Earlier notebook names in the historical execution notes below refer to archived versions in `archive/versions/2026-10-27-notebook-consolidation/`. Both 100-scenario candidate pools now exist. Consolidation did not change their contents, practice packets or annotations.

Prepared October 27, 2026. This work adds scripts and analysis. **After reviewing and testing the implementation, 100 SWE-chat candidate scenarios were selected and exported locally: 50 single-turn and 50 multi-turn.** No development/test split or human annotation has been performed. The separate analysis report contains population metadata only.

The local candidate pool is in `curated/swe_chat_curated/`. Open `scenario_manifest.csv` for the compact index, `scenarios.jsonl` for complete retained events, `quota_report.csv` for requested and achieved counts, and `provenance.json` for source checksums and runtime versions. No uploads or pushes occurred.

## Open the shared notebook

Open `conversation_length_comparison.ipynb`. Its original two-dataset length comparison and CollabSkill GDPval/APEX/APEX-Agents analysis are preserved. The added SWE-chat section contains observed and filtered populations, initial-intent matching support, proposed quotas, coding mode, pushback by length, missingness, concentration and tool/text-volume diagnostics. `explore_datasets.ipynb` includes a shorter SWE-chat review and retains its existing conversation browser.

Default cells load a validated metadata report or rebuild it using selected columns in bounded batches. They do not select scenarios. The optional preview cell is disabled.

## Inputs and coverage

- Source: [SALT-NLP/SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat).
- Pinned revision: `202b071f18e03c79df5a7287565ce2ebc5ee7756`.
- Local tables: sessions, repositories and conversations. These are a **three-table subset**, not the entire repository.
- Reverified all three SHA-256 checksums against the pinned inventory. Source size: 6,461,814,602 bytes.
- Other tables, raw/compacted transcripts, subagent transcripts, repository workspaces and external artifacts are not locally available through this subset.
- Existing downloads, downloader, exploration helpers and CollabSkill organizer are preserved. The SWE-chat script uses the same explicit review/preview/export pattern and the existing environment.

## Definitions

- **Source session:** a row in sessions.parquet, linked by session_id to released conversation events.
- **Scenario:** one whole released source session, including tools and metadata. No extracted episodes are constructed.
- **Human message:** a conversational role=user message, grouped first by source_entry_uuid, then response_id, with turn_id as fallback. Native-ID namespaces are kept distinct. Wording is not deduplicated merely because prompts repeat.
- **Assistant message:** the equivalent logical conversational assistant unit. Native block records are counted separately for auditing.
- **Round:** a consecutive run of conversational human messages, followed by any conversational assistant messages before the next human run. Tool/system events do not create rounds. Counts are structural, not a claim about semantic requests.
- **Single-turn:** exactly one logical human message in the whole released session. The agent may take many actions.
- **Multi-turn:** at least two logical human messages. Length bins are 2-4, 5-9, 10-19 and 20+.
- **Complete released session:** every event from the conversation table is retained at export. This does not certify completeness of the original interaction or access to external context.

This retains CollabSkill's one-human-message versus multiple-human-message comparison, while accounting for SWE-chat's split-message storage. CollabSkill counts released conversational user events because that schema does not expose equivalent native block identifiers.

## Eligibility and context review

Structural candidates require a human message, a first conversational message from the human, observable assistant/tool behavior after the last human message, contiguous source indices and source order. Known continuation messages and delta-only transcripts are excluded by default. Missing logs are distinct from logs containing zero human messages.

The initial-human and final-behavior checks are conservative. They can exclude valid greetings or sessions ending with acknowledgments. They improve assessability at the cost of shifting the population. Exclusion reasons and observed-versus-filtered distributions are shown rather than claiming representativeness.

Deduplication uses nonempty native content_hash values among structurally eligible sessions. Explicit export additionally checks exact retained conversational/tool text hashes and stops on a discovered duplicate rather than replacing it silently. Near duplicates are not automatically removed. User/repository connected groups preserve relationships for later splitting.

All retained candidates require human context review for missing attachments, compacted context, unrelated tasks, empty/command-only requests, and whether subsequent behavior supports assessing a boundary. Missing workspace artifacts do not by themselves prove transcript-level delegation cannot be assessed. Source continuation flags are not a complete detector of all missing history.

## Sampling design

The configuration requests 50 single-turn and 50 multi-turn scenarios. Targets are recalculated from the structurally eligible multi-turn population using largest-remainder allocation, with stable category order to break equal remainders.

1. Preserve the eligible multi-turn length distribution.
2. Match the counts of initial prompt-intent categories between single and multi groups.
3. Enforce configurable maxima of two sessions per known user and two per repository across the combined 100-scenario sample. These are concentration guardrails, not previously measured natural frequencies. Unknown users cannot be capped reliably and are reported separately.
4. Report coding mode, agent, strategy and pushback. They do not influence selection. Full joint distributions are not preserved.
5. Select using constrained integer optimization with fixed seed 3292026. Input rows are sorted before seeded random costs are assigned. A solver must prove its optimum. Runtime versions are recorded at export.

The first optimization stage maximizes matched category pairs within all quota ceilings and caps. The second selects among equally sized feasible solutions using seeded costs. These constraints make this a designed comparison sample with **unknown inclusion probabilities**. Probability and weight columns remain null. It is not an unbiased sample of SWE-chat, and group differences cannot be interpreted as a causal effect of conversation length.

No quota is silently relaxed. Infeasibility stops selection unless `--allow-shortfall` is supplied explicitly. That option returns a maximum feasible matched subset and records each requested, achieved and missing count. A count-level feasibility function checks the joint design without returning scenario IDs.

### Initial intent and missingness

Use the first observed logical conversational human message consistently. Preserve later intent changes in the source events. Conflicting labels across blocks of the first message become Unannotated with a conflict flag. Do not select the first *nonmissing* later intent, which would change the reference point across sessions.

Intent labels are source LLM outputs. Missing labels are an explicit category rather than exclusions. Matching Unannotated counts **does not establish task matching**. Before making a task-matched claim, review initial requests using a consistent guide. The configuration supports manually reviewed initial-intent overrides with required rationale, while preserving source labels and provenance. No overrides have been made.

### Pushback

Report source correction/rejection/failure_report labels on logical messages after the initial human message. A session can contain several types. Show the share among labeled later messages and the count of missing/conflicting labels separately. Single-turn rates are not applicable, rather than evidence of no corrective behavior. Consecutive human messages are not necessarily feedback on an intervening agent action. These fields are diagnostics, not human delegation judgments.

### Coding mode

Use 0% agent contribution for Human-only, >0% and <99% for Collaborative, and >=99% for Vibe coding, following Figure 5. The paper's prose uses >99%, so the exact endpoint convention is disclosed. Require a valid percentage and positive version-appropriate denominator. Version 2 uses total_lines_changed. Legacy/unspecified versions use total_committed. Missing/invalid attribution remains Unknown. Report attribution basis and unknown reason separately. Coding mode reflects realized contribution and is report-only in the sample design.

## Review findings on the pinned local release

| Population | Single | Multi | Zero human messages | Missing log |
|---|---:|---:|---:|---:|
| Observed source | 3,380 | 14,180 | 259 | 149 |
| Structural candidates | 2,745 | 11,239 | 0 | 0 |

There are 13,984 structural candidates. The aggregate intent/length design with known-user/repository caps is feasible. This does not complete human context review or prove every eventual draw is assessable.

| Human messages | Filtered multi population | Proposed count |
|---|---:|---:|
| 2-4 | 4,284 | 19 |
| 5-9 | 2,938 | 13 |
| 10-19 | 2,090 | 9 |
| 20+ | 1,927 | 9 |

Intent targets in each turn group are: create new code 8, understand 8, other 6, debug 5, refactor 4, git 3, test 1, connect 1, Unannotated 14. They follow filtered rather than earlier observed-population proportions. Missing initial intent affects 3,097 of the 11,239 multi-turn candidates (27.6%).

Missing model IDs affect 55.2% of candidates, and missing user IDs affect 33.3%. Source user identity is a resolved attribution identifier, not a verified participant identity. Coding mode is Unknown for 16.7% of multi-turn and 27.0% of single-turn candidates. These limitations are preserved in the tables.

## Commands

From Project/data, use the existing `.venv` and requirements.txt.

```bash
# Read-only rebuild and capacity review. No selection or writes.
.venv/bin/python organize_swe_chat.py --verify-hashes

# Optional NEW metadata report directory. Still no scenario selection.
.venv/bin/python organize_swe_chat.py --report analysis/swe_chat_review_NEW

# Synthetic-only checks, with disposable exports confined to a temporary folder.
.venv/bin/python -m unittest -v test_swe_chat_organization.py
```

After reviewing the report, these explicit commands can reproduce selection to a new destination:

```bash
# In-memory selection, no dataset files.
.venv/bin/python organize_swe_chat.py --preview

# Whole released sessions, exact source events and provenance in a NEW directory.
.venv/bin/python organize_swe_chat.py --export curated/swe_chat_NEW
```

Explicit export rehashes the source files, checks source counts/indices, retains original table row indices and event indices, records source revision and stable scenario IDs, writes quota/distribution reports, and reads the output back before revealing the destination. Existing destinations are rejected. Export cannot target raw/, metadata/ or archive/versions/.

Exact source events include their source labels and may contain direct contact information. A future blinded annotation packet must remove automated judgments and redact direct contact information with a documented mapping. This organizer's export is a private source-traceable candidate pool, not such a packet. It creates no human labels, LLM judgments, training data pairs or model runs.

## Sharing and disclosure

Keep raw data, report inventories and scenario exports out of a code repository. The lightweight code/configuration/notebook can be reviewed locally now. Any later sharing must respect source access conditions and content rights. No uploads or pushes are performed. Generated notebook outputs are aggregate metadata only.

Code, tests and notebook additions were produced with Codex assistance. Record this contribution in the project's AI-use disclosure. Source LLM classifications were supplied by SWE-chat and are separately identified.

References: Baumann et al. (2026), [SWE-chat, sections 2.3, 3.1, 3.2 and 4.4](https://arxiv.org/pdf/2604.20779), and the [released data dictionary](https://huggingface.co/datasets/SALT-NLP/SWE-chat#dataset-schema).

## Changelog

Added the SWE-chat organizer, configurable sampling design, synthetic regression tests and metadata report. Appended SWE-chat review sections to the shared conversation-length notebook and exploration notebook. Preserved their completed CollabSkill sections and archived prior files under archive/versions/2026-10-27-swe-chat-analysis/.

Review before selection: 30 synthetic tests passed. All 20 code cells of the shared comparison notebook and the changed exploration sections executed in process. A standard Jupyter kernel launch was blocked by the sandbox's local-socket restriction, so this does not claim a separate-kernel launch test. Chart layout was inspected. The count-level design was feasible. On the real in-memory selection, input-row shuffling preserved the selected IDs. Export then reverified checksums, full source boundaries and logical counts and read the JSONL back before making the new folder visible.

All intent and length quotas were achieved with zero shortfall. The result spans 75 repositories and six agent labels. Known users and repositories contribute at most two sessions each. There are 38 scenarios with missing source user IDs, so their user concentration is unverified. Fourteen scenarios per turn group have Unannotated initial intent. Coding-mode counts differ between groups and were not matched. These are candidate scenarios, not a human-validated or causally matched benchmark.
