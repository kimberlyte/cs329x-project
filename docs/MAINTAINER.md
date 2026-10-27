# Maintainer handoff

This is the team starter repository named `cs329x-project`. The user supplied the existing Hugging Face dataset repository [nyartist/cs329x-project](https://huggingface.co/datasets/nyartist/cs329x-project). Read-only inspection on October 27, 2026 found only `.gitattributes`, without the three curated packages. The GitHub repository distributes code and documentation. Curated packages are distributed separately through Hugging Face; the dataset upload and release pin remain pending.

## Before enabling the team download

1. Publish the reviewed curated package to `nyartist/cs329x-project` using the chosen team access settings. Preserve the current contents of these three folders: `collabskill-curated/`, `swe_chat_curated/`, `midway-annotation-sample/`. Put them at its root, or set `remote_prefix` in the download manifest to the chosen containing folder.
2. Record the dataset ID and full immutable 40-character commit hash in `curated_download_manifest.json`. Its `repo_id` is already `nyartist/cs329x-project`. The `revision` is null until the actual data commit exists; do not pin the initial empty-repository commit. Invite the teammates to the dataset if it is private.
3. Check that all 70 manifest files exist remotely with the recorded sizes and hashes. The manifest describes the currently saved three packages. If packaging changes, review the changes and deliberately regenerate the affected entries instead of bypassing verification.
4. Run the downloader from an empty checkout, authenticate locally, and execute both notebooks and annotation commands. Real authenticated Hub transfer remains to be verified after publishing. Current transfer tests use local fixtures and checksum checks.
5. Share the dataset ID/commit and access instructions. Teammates should not need the raw SWE-chat release for annotation.

The downloader supports one curated Hugging Face repository per manifest. Separate locations or a different package layout will require a deliberate manifest/downloader update when those details are known. Human annotation files remain separate from the original blank templates.

## Information still to supply

- Final project/repository name, team contacts and sharing permissions.
- Immutable data release revision and Hugging Face access instructions. The dataset ID is already configured.
- Final adherence rubric, any additional dimensions and agreed handling of ambiguous/no-boundary cases.
- Annotator IDs and assignments, review date, independent annotation protocol and submission destination.
- Disagreement-review procedure, any agreement statistic and later split policy.
- License/terms decisions for the team's code and dataset distribution. No new blanket license is asserted here.

## What to push

Push this prepared folder's code, notebooks, reports, tests, README and docs. Keep `.gitignore`. Do not copy the local course data folder wholesale into the repository. `raw/`, `curated/`, `annotations/*.csv`, local credentials, environments, caches and archives are ignored.

`preparation_manifest.json` records the preparation inputs, file checksums and notebook changes. It is packaging provenance, not evidence of a remote release or completed human annotation. `reports/preparation_checks.json` records local validation and its limits.

When team details change, update this repository's README/docs/configs. Preserve existing human annotation files and the pinned data identity unless a deliberate data revision is agreed.

Downloader API reference: [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/download). Downloads specify the dataset repository type and a full immutable revision.
