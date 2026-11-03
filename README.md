# CS 329X: delegation adherence

Study whether agents respect users’ stated boundaries and evolving instructions, then compare independent adherence judgments with the datasets’ existing outcome scores.

The saved pool contains **200 unique sessions**: 100 CollabSkill and 100 SWE-chat, each with 50 single-turn and 50 multi-turn sessions. The six-thread midway practice packet is drawn from those 200 sessions. Human adherence judgments remain to be completed.

## Start here

Open **[the dataset review notebook](data/analysis/dataset_review.ipynb)**. It combines the previous source-statistics and sampling-review notebooks and includes:

- Source populations, conversation lengths, task categories, agents, coding modes and pushback.
- **CollabSkill task quality:** overall grades and four criterion components.
- **SWE-chat session success:** five evaluation dimensions, six original rubric bands, score distributions and missingness.
- Optional live integrity checks and analysis of the downloaded 200-session pool.

The source overview is a frozen October 27, 2026 snapshot. Outcome aggregates were computed on November 2, 2026 from the same pinned source releases and saved samples. Both render without raw data or dataset access. Live sample checks require the curated packages. Source scores are automated assessments and differ from human adherence judgments.

## Repository layout

```text
data/
├── curation/                # Downloaders, two source organizers and config/
├── analysis/                # One notebook, shared analysis helpers and reports/
├── annotation/              # Annotation tools and provisional schema
├── curated/                 # Downloaded packages, ignored by Git
├── raw/                     # Optional original sources, ignored by Git
└── metadata/                # Optional source checksums and mappings, ignored by Git
annotations/                 # Independent working CSVs, ignored by Git
docs/                        # Methods, outcome definitions and team instructions
└── history/                 # Original preparation records and execution notes
tests/                       # Offline regression checks
```

The unused source-exploration helper was removed from this GitHub package. Previous committed versions remain in Git history. The original local source workspace and dataset contents are unchanged.

## Setup and review

Use Python 3.11–3.13. Run commands from the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab data/analysis/dataset_review.ipynb
```

On Windows, activate with `.venv\Scripts\activate`.

## Download the curated packages

Team dataset: [nyartist/cs329x-project](https://huggingface.co/datasets/nyartist/cs329x-project). The tracked manifest still has no immutable release commit configured. The last recorded remote inspection was October 27, 2026, when packages had not been uploaded. Supply the released commit once the maintainer shares it:

```sh
hf auth login
python -m data.curation.download_curated_data --revision FULL_40_CHARACTER_COMMIT
```

Replace the placeholder with the actual 40-character commit. The downloader installs about 103 MB below `data/curated/`, verifies size/SHA-256 and preserves differing local files. It does not fetch raw sources. Use `--dataset midway` for only the practice packet, or `--verify-only` for offline checks. Configuration lives in [data/curation/config](data/curation/config/).

If you used the previous layout, move your existing repository-level `curated/` folder into `data/curated/` before verification. Preserve working CSVs in `annotations/`. Folder relocation does not change dataset hashes or scenario IDs.

## Independent adherence annotation

Open `data/curated/midway-annotation-sample/practice_threads.html` after downloading the packet. The three single-/multi-turn pairs cover bridal sales, UAT planning and SWE-chat code review.

```sh
python -m data.annotation.tools create --annotator YOUR_ID
python -m data.annotation.tools validate annotations/YOUR_ID.csv
python -m data.annotation.tools validate annotations/YOUR_ID.csv --require-complete
```

Use one file per annotator. Describe the boundary, cite original event indices, and record rationale, uncertainty and missing context. Values `full`, `partial` and `none` are provisional. Use `not_assessable` with a blank score when evidence is insufficient. Creation never replaces an existing file.

**Keep session-level outcome scores hidden until independent adherence judgments are complete.** The notebook publishes aggregate outcomes only. Annotation reading views hide source judgments, while exact private source exports can retain other labels. Join source outcomes afterward by `session_id`.

## Methods and maintenance

- [Outcome dimensions and scoring](docs/OUTCOMES.md)
- [Annotation workflow](docs/ANNOTATION.md)
- [Curation and optional raw reproduction](docs/CURATION.md)
- [Dataset card and provenance](docs/DATASET_CARD.md)
- [Maintainer handoff](docs/MAINTAINER.md)

Run `python -m unittest discover -s tests -q` for offline checks. The sample is a curation pool requiring context review. Category matching does not establish causal comparisons. No train/test partition or training is assigned. Keep exposed practice sessions and related groups out of a later blind holdout.

GitHub contains code, documentation and aggregate reports. Conversation data and personal annotations remain outside Git. Source datasets: [CollabSkill](https://huggingface.co/datasets/SALT-NLP/cogym-collabskill-trajectories) and [SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat). Codex assisted with code, analysis and documentation.
