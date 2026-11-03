# Generated outputs

| Folder | Contents |
| --- | --- |
| `annotations/` | One independent adherence CSV per annotator, with stable scenario IDs |
| `model_results/` | Future model predictions, evaluation scores, metrics and run logs |

Generated files in these folders are ignored by Git. The directory placeholders and this guide are tracked. No model runs or completed human judgments are bundled here. Keep secrets out of run logs and preserve useful results under distinct run names.

Annotation code and the provisional schema live in `data/annotation/`. After downloading the practice packet, run from the repository root:

```sh
python -m data.annotation.tools create --annotator YOUR_ID
python -m data.annotation.tools validate outputs/annotations/YOUR_ID.csv
python -m data.annotation.tools validate outputs/annotations/YOUR_ID.csv --require-complete
```

The creation command writes `outputs/annotations/YOUR_ID.csv` and rejects existing files. Keep one file per annotator and do not overwrite the original downloaded blank template.

The review notebook lives at [data/analysis/dataset_review.ipynb](../data/analysis/dataset_review.ipynb). It combines the former `dataset_statistics.ipynb` and `sampling_review.ipynb`. See [annotation instructions](../docs/ANNOTATION.md) for scoring guidance.
