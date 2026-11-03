# Curation and raw-data reproduction

The annotation workflow downloads the three saved packages with `python -m data.curation.download_curated_data`. It does not require recreating the sample.

## Code and configuration

| Purpose | Module | Configuration under `data/curation/config/` |
| --- | --- | --- |
| Team-package download | `data.curation.download_curated_data` | `curated_download_manifest.json` |
| Optional upstream download | `data.curation.download_datasets` | `download_manifest.json` |
| CollabSkill curation | `data.curation.organize_collabskill` | `collabskill_sampling.json` |
| SWE-chat curation | `data.curation.organize_swe_chat` | `swe_chat_sampling.json` |

Run modules from the repository root. Runtime files live below `data/raw/`, `data/metadata/` and `data/curated/`. Configurations and analysis code are tracked. Raw data, conversation exports and personal annotations are ignored by Git.

Both organizers use seed 3292026. The saved sets contain 100 sessions per source, with 50 single-turn and 50 multi-turn each. Neither outcome scores nor adherence judgments drive selection. Default organizer behavior reviews metadata. Selection/export requires explicit flags, and existing output destinations are rejected.

The original execution notes are retained under [history/](history/): [CollabSkill](history/COLLABSKILL_ORGANIZATION.md) and [SWE-chat](history/SWE_CHAT_ORGANIZATION.md). Their old paths describe the original local workspace. Use the current module commands here for this repository.

## Inputs for reconstruction

Full reconstruction requires additional source inputs, which are not bundled in GitHub or the curated downloader:

- Pinned upstream files and verification inventories under `data/raw/` and `data/metadata/`.
- CollabSkill’s GDPval enrichment and provenance under `data/metadata/external_sources/gdpval/`.
- Reviewed APEX task mapping and provenance under `data/metadata/apex_category_mapping/`.
- SWE-chat’s session, repository and conversation tables and checksum inventories.

A complete reproduction input bundle remains to be arranged if the team needs it. Having scripts in GitHub does not imply those inputs are present.

From a restored reproduction environment:

```sh
python -m data.curation.organize_collabskill --config data/curated/collabskill-curated/sampling_config.json --verify-hashes --export data/curated/collabskill-reproduced
python -m data.curation.organize_swe_chat --config data/curated/swe_chat_curated/sampling_config.json --verify-hashes --export data/curated/swe-chat-reproduced
```

Use saved export configs to reproduce the saved design. The editable configs under `data/curation/config/` support inspection and deliberate future changes. Changed paths and module imports alter code hashes but do not change the saved dataset contents, selection rules or existing provenance.

## Interpretation

A scenario retains a whole released session. Original-log completeness and access to referenced artifacts are not guaranteed. CollabSkill counts conversational user events. SWE-chat groups fragments by native message IDs. Counts can include source-marked setup/control prompts.

Categories and initial intents are matched at count level. Matching does not randomize conditions or establish a causal effect of conversation length. Inclusion probabilities and design weights are unknown. Source labels are descriptive attributes, separate from human adherence judgments. Keep related `split_group_id` values together in future partitions.

The six practice sessions are copies from the 200-session pool. Their exposure during rubric development excludes them and related groups from a later blind holdout.
