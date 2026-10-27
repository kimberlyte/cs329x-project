# Curation methods and raw-data reproduction

The team's annotation path downloads the saved curated packages. The original raw-to-curated scripts are included so reviewers can inspect how eligibility, message counting and selection work.

| Dataset | Script | Configuration |
| --- | --- | --- |
| CollabSkill | `organize_collabskill.py` | `collabskill_sampling.json` |
| SWE-chat | `organize_swe_chat.py` | `swe_chat_sampling.json` |

Both use seed 3292026. Saved targets, achieved quotas and exact source events are in each downloaded export. The selected sets contain 100 scenarios per source, each split 50/50 between single-turn and multi-turn. Neither organizer uses human adherence judgments to select the sample. Main script default behavior reviews metadata; selection/export requires explicit flags. Existing output destinations are rejected.

Full method notes are [COLLABSKILL_ORGANIZATION.md](COLLABSKILL_ORGANIZATION.md) and [SWE_CHAT_ORGANIZATION.md](SWE_CHAT_ORGANIZATION.md). Their historical execution references describe the original local environment. Archives and earlier notebooks referenced there are not part of this starter repository.

## Inputs for full reconstruction

Reproduction from raw is optional and is not required to annotate the midway sample. It requires the original pinned release files, verification inventories and enrichment/mapping inputs. Those source files are not bundled in GitHub or included by the curated downloader.

- `download_manifest.json` records the pinned upstream revisions. `download_datasets.py` can download raw sources after required access/authentication.
- CollabSkill also needs the pinned GDPval enrichment Parquet and its provenance under `metadata/external_sources/gdpval/`, reviewed APEX task mapping/provenance under `metadata/apex_category_mapping/`, and the upstream checksum inventory under `metadata/collabskill.checksums.json`.
- SWE-chat needs its three source tables, upstream checksum metadata, and source verification records generated/read by the organizer. Its optional population-review cache is separate from the curated download.
- The original source/mapping provenance is retained in the downloaded exports. A complete raw-reproduction input bundle or its download location is still to be arranged if the team needs to rerun source curation. The included scripts do not claim those missing inputs are present.

From a fully restored reproduction environment, write to new output folders:

```sh
python organize_collabskill.py --config curated/collabskill-curated/sampling_config.json --verify-hashes --export curated/collabskill-reproduced
python organize_swe_chat.py --config curated/swe_chat_curated/sampling_config.json --verify-hashes --export curated/swe-chat-reproduced
```

Use the saved sampling configs above when reproducing the saved design. The top-level editable configs are retained for method inspection and future deliberate changes. Dependencies and source revisions matter to exact reconstruction.

## Interpretation

A whole released session is retained rather than truncating a multi-turn conversation into a single-turn example. Original log completeness and artifact availability are not independently guaranteed. CollabSkill uses released conversational human-event units, while SWE-chat groups source fragments by native identifiers. Counts can include source-marked setup/control prompts.

Categories and initial intent are matched at count level. This does not randomize conditions, make examples identical, or establish causal effects of conversation length. Inclusion probabilities and design weights are unknown. Source-generated labels are descriptive retrieval aids, not human adherence ground truth.
