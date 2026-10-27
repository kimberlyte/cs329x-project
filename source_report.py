"""Render the frozen aggregate source overview without requiring raw datasets."""
import json
from pathlib import Path
from IPython.display import display

ROOT = Path(__file__).resolve().parent


def display_source_snapshot(key):
    report = json.loads((ROOT / 'reports/source_statistics.json').read_text())
    for output in report['outputs'].get(key, []):
        data = {mime: ''.join(value) if isinstance(value, list) else value for mime, value in output['data'].items()}
        display(data, raw=True)
