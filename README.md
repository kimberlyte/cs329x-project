# cs329x-project

This is the team's starter repository for reviewing curated human–AI conversations and practicing **delegation-adherence annotation**: whether an agent respects the user's stated boundaries and evolving instructions.

The first milestone is to annotate and discuss the **six-thread Midway Annotation Sample**. The detailed rubric, assignments, submission location and schedule will follow. Current fields and validation rules are provisional.

## What is included

- **100 CollabSkill scenarios:** 50 single-turn and 50 multi-turn.
- **100 SWE-chat scenarios:** 50 single-turn and 50 multi-turn.
- **Six midway practice threads:** bridal sales, UAT test planning and code review, each with one single-turn and one multi-turn example.

The six practice threads are copies from the 200 curated scenarios. They do not increase the unique count to 206. A turn count here means the number of conversational human messages, with native SWE-chat fragments grouped. Tool events are separate.

**GitHub contains the code, notebooks, aggregate source overview and documentation. Conversation datasets and the midway packet will be downloaded from the team's Hugging Face dataset repository.** They are not bundled in this repository. No human annotations have been completed in the starter template.

## 1. Set up

Use Python 3.11–3.13. Local checks used Python 3.13.5. From this repository's root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead. Keep authentication tokens outside repository files.

## 2. Download the curated data

**Team dataset:** [nyartist/cs329x-project](https://huggingface.co/datasets/nyartist/cs329x-project). The repository exists, but the curated packages have not been uploaded yet. The pinned data commit and any required access invitation will follow. Until a commit is configured, the download command intentionally stops with a configuration message. You can already read the source-overview notebook and these instructions.

Once you have access, authenticate locally and use the supplied values:

```sh
hf auth login
python download_curated_data.py --revision FULL_40_CHARACTER_COMMIT
```

`FULL_40_CHARACTER_COMMIT` is a placeholder for the data release commit, which will be shared after upload. The dataset ID is already configured. The maintainer can fill `revision` in `curated_download_manifest.json`, after which `python download_curated_data.py` is sufficient.

The downloader installs only the three curated packages, approximately **103 MB**, checks every file's size and SHA-256, and refuses to replace differing local files. It does not download the multi-gigabyte raw sources. The manifest assumes the three folders below are at the Hugging Face repository root. If the final layout changes, the maintainer will update the manifest.

```text
curated/
├── collabskill-curated/           # 100 saved scenarios
├── swe_chat_curated/              # 100 saved scenarios
└── midway-annotation-sample/      # 6 selected threads and blank template
```

To get only the midway packet, add `--dataset midway`. To verify already downloaded files without contacting Hugging Face:

```sh
python download_curated_data.py --verify-only
```

## 3. Review the notebooks

```sh
python -m jupyterlab
```

| Notebook | What it provides |
| --- | --- |
| [dataset_statistics.ipynb](dataset_statistics.ipynb) | Frozen aggregate source statistics from October 27, 2026, with separate CollabSkill/SWE-chat sections and comparison graphs. Rerunning renders the saved report without raw data. |
| [sampling_review.ipynb](sampling_review.ipynb) | Live checks and distributions for the downloaded 200 scenarios, including category/intent, coding mode, pushback and conversation length. Includes a check of the six-thread practice selection. |

The frozen source overview describes the available source populations used for curation, not a fresh download or the curated sample. The live sampling review requires all three curated packages and does not require the raw datasets. Neither notebook resamples data, assigns splits or writes annotations.

## 4. Start midway annotation

Open **`curated/midway-annotation-sample/practice_threads.html`** in a local browser. Use this reading page and its linked threads, which show the released conversations and relevant expandable tool events.

| Practice IDs | Pair | Single / multi human messages |
| --- | --- | --- |
| P01 / P02 | CollabSkill bridal sales training | 1 / 3 |
| P07 / P08 | CollabSkill UAT test plan | 1 / 3 |
| SWE-P01 / SWE-P02 | SWE-chat code review and cleanup | 1 / 2 |

Create an independent working file using your assigned annotator ID. `annotator01` is an example, not an assignment:

```sh
python annotation_tools.py create --annotator annotator01
```

This creates `annotations/annotator01.csv`. Open it in your preferred CSV editor. Do not edit the downloaded blank template or its IDs. Existing annotation files are never overwritten by the creation command.

For each thread, describe the applicable boundary, assess adherence where evidence permits, cite original event indices, and record your rationale, uncertainty and missing context. The provisional values are `full`, `partial` and `none`. Use `not_assessable` and a blank adherence value where evidence is insufficient. A completed case with no evidenced boundary can also leave the adherence value blank and explain why.

Check a draft, then check completed coverage before team review:

```sh
python annotation_tools.py validate annotations/annotator01.csv
python annotation_tools.py validate annotations/annotator01.csv --require-complete
```

These commands validate fields, IDs, statuses and event references. They do not judge your annotation quality. Keep independent judgments separate until the team compares them. **Submission details will follow.** Personal annotation CSVs are Git-ignored by default.

Read [docs/ANNOTATION.md](docs/ANNOTATION.md) for the current workflow and known context limits. The human rubric remains open for refinement after the midway exercise.

## Other repository contents

| Files | Purpose |
| --- | --- |
| `download_curated_data.py`, `curated_download_manifest.json` | Pinned download and integrity checks for the team packages. |
| `annotation_tools.py`, `annotations/README.md` | Independent working files and structural validation. |
| `organize_collabskill.py`, `collabskill_sampling.json` | Raw-to-curated CollabSkill eligibility, quotas and explicit export. |
| `organize_swe_chat.py`, `swe_chat_sampling.json` | Raw-to-curated SWE-chat counting, eligibility, quotas and explicit export. |
| `download_datasets.py`, `download_manifest.json` | Optional original raw-source downloads and pinned source references. |
| `notebook_review.py`, `exploration.py`, `notebook_review_snapshots.json` | Shared loaders, analysis and snapshot checks. |
| `source_report.py`, `reports/source_statistics.json` | Frozen aggregate source overview. |
| `tests/` | Download protection, annotation validation and organizer regression checks. |

The raw curation scripts are included so the method is inspectable. Recreating the pools from scratch requires additional raw sources and category-enrichment inputs, described in [docs/CURATION.md](docs/CURATION.md). Annotation does not require this step. The earlier practice-packet generators and local archives are outside this starter.

Run the local checks with `python -m unittest discover -s tests -q`.

## Interpretation and provenance

These are curated candidates, not a human-validated benchmark. Source automated intent/pushback labels, code-attribution measures, Bayesian ratings and task grades do not represent our human delegation-adherence labels. Task outcome and adherence are separate variables. Referenced attachments/workspaces may be unavailable. Keep source outcomes hidden while making independent adherence judgments.

The practice examples are exposed for rubric development and must stay out of a later blind holdout, including related split groups. No train/test split or training is assigned here.

Source releases: [CollabSkill](https://huggingface.co/datasets/SALT-NLP/cogym-collabskill-trajectories) and [SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat). Source attribution and applicable conditions are described in [docs/DATASET_CARD.md](docs/DATASET_CARD.md). Code and packaging involved Codex assistance. Conversation content was copied from the source releases.

## Maintainer handoff

The Hugging Face location is `nyartist/cs329x-project`. Its data-release commit, access details, final rubric, annotator assignments, review date and submission destination remain to be supplied. See [docs/MAINTAINER.md](docs/MAINTAINER.md). The GitHub repository contains code and documentation. Conversation datasets are distributed separately through the Hugging Face repository.
