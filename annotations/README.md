# Independent annotation files

After downloading the midway sample, run `python annotation_tools.py create --annotator YOUR_ID` from the repository root. This folder receives `YOUR_ID.csv`, containing all six selected practice IDs and blank judgments.

Keep one file per annotator and preserve scenario IDs. Validate drafts with `python annotation_tools.py validate annotations/YOUR_ID.csv`. Add `--require-complete` before review.

Working CSVs are Git-ignored. Submission details will be shared later. Do not replace the downloaded original template with completed annotations.
