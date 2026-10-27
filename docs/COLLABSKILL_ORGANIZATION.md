# CollabSkill organization and sampling review

> **Notebook consolidation, October 27, 2026:** Use `dataset_statistics.ipynb` for source data and cross-dataset comparisons, and `sampling_review.ipynb` for both saved curated samples. Earlier notebook names in the historical execution notes below refer to archived versions in `archive/versions/2026-10-27-notebook-consolidation/`. Both 100-scenario candidate pools now exist. Consolidation did not change their contents, practice packets or annotations.

Prepared October 27, 2026. This workflow concerns the CollabSkill release and its GDPval, APEX and APEX-Agents tasks. It does not curate SWE-chat or expand the pool to 200 examples by itself.

## What is ready

- `organize_collabskill.py`: read-only inventory/analysis, optional in-memory selection preview, and explicit export to a new directory.
- `collabskill_sampling.json`: seed, counting rule, pinned source revision and mapping hashes, category quotas, length quotas, optional concentration caps, mapping policy and manual exclusions.
- `collabskill_sampling_review.ipynb`: main review notebook. Default cells analyze source and filtered populations without selecting threads or writing datasets.
- Existing statistics, exploration and conversation-comparison notebooks now include source-specific category comparisons. Prior versions were copied unchanged into `archive/versions/2026-10-27-collabskill-analysis/` before editing.

No production candidate dataset has been selected or exported in this editing turn. The later execution commands below are intentionally separate from the code and notebook review.

## Definitions

**Source session:** one released CollabSkill session ID. **Scenario:** one entire released session when selected. No excerpts or extracted episodes are created. “Complete released session” describes export boundaries, not independently verified original-log completeness.

**Human message:** an event with `role=user` and `is_conversational=true`. **Single-turn:** exactly one such event. **Multi-turn:** two or more. Setup/help commands still count when marked conversational in the source. These counts do not identify substantive requests, revisions or feedback events semantically.

**Assistant message:** a source-marked conversational assistant event. **Conversational round:** a consecutive block of human conversational messages plus following assistant conversational messages, until the next human block. Consecutive human messages can therefore exceed round counts. Tool/system events do not open rounds. Source `turn_id` values are event identifiers, not this round count.

**Tool event:** a released tool-role event or an event identified as a tool event by its type/tool name. Heterogeneous interfaces log tools differently, so event volume is not a standardized measure of actual work.

Missing logs have unknown message/round counts (`null`/missing), not zero. Available logs with zero human messages are reported separately. Full source event indices, IDs, wording and tool actions are retained in the eventual local scenario JSONL.

## Categories and provenance

- GDPval: `sector` and `occupation` from an exact task-ID join to the pinned original benchmark metadata. Its sampling category is sector.
- APEX: separate native domain labels. Fifteen task mappings use reviewed source-prompt content matches, and five are category inferences from assignments/deliverable filenames. These are not all exact-ID joins.
- APEX-Agents: separate job-domain labels. Seventeen task classifications are provisional and two remain unresolved. Original task-level metadata access was denied. No occupation or sector labels are invented for these tasks.

`native_category` retains these source-specific category systems rather than asserting a common cross-benchmark taxonomy. `category_status` and `category_method` remain visible in tables and exports. Source categories are descriptive metadata, not delegation-quality labels.

The configuration permits the discussed inferred/provisional categories. Set `mapping_policy.allow_inferred` or `allow_provisional` to false for a stricter sensitivity analysis. This can make the requested source/category quotas infeasible. The code reports that condition rather than relabelling or replacing categories.

## Populations and eligibility

The notebooks distinguish **all source sessions**, **filtered structural candidates**, and **selected candidates** only when the optional preview is explicitly enabled. Source tables include missing logs. Both raw and filtered distributions are shown.

Filtering excludes missing logs, zero conversational human messages, no observable agent behavior after the first human message, exact duplicate transcripts, disallowed/unresolved mappings, and configured manual exclusions. Duplicate comparison uses exact event role/type/content/tool fields, independent of session ID. One deterministic canonical eligible session is retained per exact transcript hash. Shared tasks with different transcripts are not duplicates.

The known manually excluded session is `1a498080-a080-4c4b-98cd-be837e545ede`: its recorded election-article request differs from its assigned SEC-disclosure task.

Every remaining candidate has `requires_human_context_review=true`. Eligibility here is structural, not an assurance that context is sufficient to annotate adherence. First/last conversational role, command-like messages, tool volume and log status are diagnostics. Mixed tasks, missing initial context, inaccessible reference files and original extraction omissions still require human review. Exclude newly discovered unsuitable sessions through the config with a reason, then rerun. No keyword hits, task outcome scores, final pooled skill ratings, LLM judgments or assumed adherence labels drive selection.

## Sampling design

The current configuration requests 50 single-turn and 50 multi-turn sessions. Source totals per turn type are 39 GDPval, 5 APEX and 6 APEX-Agents. Multi-turn length quotas are 31 at 2–4 human messages, 13 at 5–9, 4 at 10–19 and 2 at 20+. The observed and filtered proportional targets are recomputed for inspection, but they never overwrite the saved configuration.

Selection is a seeded constrained optimization over whole sessions. It enforces equal single/multi counts within each source/category and the configured multi-turn length targets. Among feasible solutions, it prefers matching GDPval occupation counts and then uses seeded random costs. Sector-level or partial occupation matches remain flagged. No occupation labels are synthesized for APEX. This design balances category composition, but is not a causal experiment or a simple random sample. First-order inclusion probabilities and design weights are unknown and are exported as missing, not invented as `n/N`.

Concentration by user, task, agent and connected group is always reported. Caps default to `null` because numeric limits have not been agreed. Configure positive `max_per_user`, `max_per_task` and/or `max_per_agent` values before sampling to impose hard caps. Quotas are never silently relaxed. Solver failure/timeouts stop execution. If a full design is infeasible, the default stops without exporting. With the explicit `--allow-shortfall` flag, the solver returns the maximum feasible matched subset within every quota ceiling and reports each shortfall.

The same seed and unchanged inputs/runtime produce the same selection. Provenance records configuration, mapping/source checksums, code checksum, seed and runtime versions. SciPy 1.15.3 is pinned for the optimizer. A changed mapping hash or source revision requires deliberate config review.

## Development/test policy

No split or policy training is performed. `split_group_id` connects source sessions through any shared user or task, transitively, so later splits can keep these groups intact. Category matching is count-level matching, not a set of causal pairs. Future development/test assignment must consider category/length coverage and keep related scenarios on one side. Pilot/rubric-development examples must not enter a frozen final test set. APEX-Agents-derived examples are not cleared for training under the original source's stated evaluation-only conditions.

## Setup and commands

Use the existing `Project/data/.venv` and the **CS 329X Data Explorer** Jupyter kernel. Updated dependencies are in `requirements.txt`. SciPy is imported only when selection is requested.

```bash
# Run from Project/data. Read-only review, including all CollabSkill source hashes.
.venv/bin/python organize_collabskill.py --verify-hashes

# Later: select in memory and inspect achieved quotas; no dataset files written.
.venv/bin/python organize_collabskill.py --preview

# Later: generate a candidate dataset in a NEW directory.
.venv/bin/python organize_collabskill.py --verify-hashes --export curated/collabskill-curated

# Tests use disposable synthetic examples only.
.venv/bin/python -m unittest -v test_collabskill_organization.py
```

The new sampling-review notebook writes no figures, samples or annotation files by default. The older conversation-comparison notebook retains its pre-existing figure-export behavior when run in full. Raw sources and earlier helpers/notebooks are preserved.

## Later export contents

- `scenarios.jsonl`: whole released event sequences with original indices and source events.
- `scenario_manifest.csv`: selected IDs, categories/provenance, length counts, group IDs and matching status.
- `population_inventory.csv`: all source sessions, structural filters, review flags and exclusion reasons.
- Source/eligible/selected category, length and concentration tables.
- `quota_report.csv`: requested/achieved counts and shortfalls.
- Config and provenance snapshots, plus a local handling note.

An existing destination is rejected. Export destinations inside `raw/` or `metadata/` are rejected. No automatic download, upload, GitHub push, remote write, model inference, human annotation or training occurs. Raw source event wording is retained, so a later snapshot can contain contact information or sensitive text and is not a sharing-ready annotation packet. `curated/` is Git-ignored, which does not prevent Dropbox synchronization.

## Sources and AI-use disclosure

Use the pinned source revisions and citations in the existing data README and `metadata/apex_category_mapping/README.md`. CollabSkill is released by SALT-NLP. GDPval enrichment uses the original OpenAI dataset. The APEX source CSV and mapping decisions retain their own provenance. The code, notebook edits and provisional mappings were prepared with Codex assistance. No human-validated delegation annotations are claimed.

## Review completed before dataset generation

- Twenty-five synthetic tests passed: event/round counting, missingness, source-check integration, duplicate canonical selection, category-share denominators, quota enforcement, occupation balance, seed reproducibility under reordered inputs, explicit shortfalls, concentration caps, source-event round-trip export, existing-output protection and failed-export cleanup. Independent post-selection validation rechecks eligibility, source metadata, category/length quotas and concentration caps before export. Regression tests also cover a destination created by another process during staging and require an explicit flag for partial exports.
- All four notebooks passed schema validation and code parsing. Read-only analysis cells executed successfully in the new review notebook, the complete updated statistics notebook, the comparison notebook with its legacy figure outputs rerouted to a temporary directory, and the two appended exploration cells. Existing large SWE-chat scan/browser cells were not rerun.
- Charts were rendered and inspected. An import needed by a retained statistics cell was restored, and chart legends were moved away from data marks. Notebook outputs are cleared in the delivered versions.
- All 393 pinned CollabSkill source-file SHA-256 checksums and the GDPval enrichment checksum passed. Observed counts reconcile to 386 sessions, 325 available logs, 84 single-turn and 239 multi-turn sessions. Structural filters leave 321 candidates: 83 single-turn and 238 multi-turn. This is not a human-validated eligible pool.
- No production selection preview, dataset export, model inference, annotation, policy training or upload was performed. Export tests used disposable synthetic data only. The final sampling run and joint feasibility of its specific selected scenarios remain for the later execution step.

A second code review tightened export destination reservation and independent post-solver validation. Pre-review versions of the script, tests and this guide are retained in `archive/versions/2026-10-27-script-review/`.
