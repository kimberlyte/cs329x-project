"""Create independent annotation files and check their structure against the midway packet."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
PACKET = ROOT / 'curated/midway-annotation-sample'
FIELDS = ['practice_id', 'scenario_id', 'annotator_id', 'annotation_status', 'applicable_boundary', 'boundary_adherence', 'evidence_event_indices', 'rationale', 'uncertainty', 'missing_context', 'notes']
STATUSES = {'not_started', 'in_progress', 'complete', 'not_assessable'}


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not set(FIELDS).issubset(reader.fieldnames or []):
            raise ValueError('Missing required annotation columns.')
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError('Malformed CSV row: missing cells or an unquoted comma.')
    return rows


def create(annotator, packet=PACKET, output=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', annotator):
        raise ValueError('Use an annotator ID containing only letters, digits, underscores or hyphens.')
    rows = read_csv(Path(packet) / 'annotation_template.csv')
    judgment_fields = FIELDS[4:]
    if any(row['annotator_id'] or row['annotation_status'] != 'not_started' or any(row[field] for field in judgment_fields) for row in rows):
        raise ValueError('The downloaded template is not blank. Obtain a fresh template before creating an independent file.')
    output = Path(output) if output else ROOT / 'annotations' / (annotator + '.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: annotator if key == 'annotator_id' else row[key] for key in FIELDS})
    return output


def validate(path, packet=PACKET, require_complete=False):
    rows = read_csv(path)
    records = {}
    for line in (Path(packet) / 'scenarios.jsonl').read_text().splitlines():
        record = json.loads(line)
        records[record['scenario_id']] = {int(event['source_event_index']) for event in record['events']}
    expected = {row['scenario_id']: row['practice_id'] for row in read_csv(Path(packet) / 'annotation_template.csv')}
    errors, seen, annotators = [], set(), set()
    for number, row in enumerate(rows, 2):
        prefix = f'Row {number}: '
        sid = row['scenario_id']; status = row['annotation_status']; score = row['boundary_adherence']
        if sid in seen:
            errors.append(prefix + 'duplicate scenario ID')
        seen.add(sid)
        if sid not in expected or row['practice_id'] != expected.get(sid):
            errors.append(prefix + 'unknown scenario or incorrect practice ID')
        if not row['annotator_id'].strip():
            errors.append(prefix + 'annotator_id is required')
        annotators.add(row['annotator_id'])
        if status not in STATUSES:
            errors.append(prefix + 'invalid annotation_status')
        if score not in {'', 'full', 'partial', 'none'}:
            errors.append(prefix + 'invalid boundary_adherence value')
        if status == 'not_assessable' and (score or not (row['missing_context'].strip() or row['rationale'].strip())):
            errors.append(prefix + 'not_assessable requires a blank score and an explanation')
        if score and not all(row[key].strip() for key in ['applicable_boundary', 'rationale', 'evidence_event_indices']):
            errors.append(prefix + 'a scored judgment requires a boundary, rationale and event evidence')
        if status == 'complete' and not row['rationale'].strip():
            errors.append(prefix + 'a completed judgment requires a rationale')
        if status == 'complete' and row['applicable_boundary'].strip() and not score:
            errors.append(prefix + 'a completed applicable boundary requires a score, or use not_assessable')
        if require_complete and status not in {'complete', 'not_assessable'}:
            errors.append(prefix + 'annotation is still a draft')
        text = row['evidence_event_indices'].strip()
        if text:
            try:
                indices = [int(value.strip()) for value in text.split(';')]
                if not set(indices).issubset(records.get(sid, set())):
                    raise ValueError()
            except ValueError:
                errors.append(prefix + 'evidence must contain existing original event indices separated by semicolons')
    if seen != set(expected):
        errors.append('The file must contain exactly the six selected scenarios.')
    if len(annotators) != 1:
        errors.append('Each file must contain one annotator ID.')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, default=PACKET)
    commands = parser.add_subparsers(dest='command', required=True)
    new = commands.add_parser('create'); new.add_argument('--annotator', required=True); new.add_argument('--output', type=Path)
    check = commands.add_parser('validate'); check.add_argument('path', type=Path); check.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    try:
        if args.command == 'create':
            print('Created ' + str(create(args.annotator, args.packet, args.output)))
        else:
            errors = validate(args.path, args.packet, args.require_complete)
            if errors:
                parser.exit(1, '\n'.join(errors) + '\n')
            print('Annotation structure and event references are valid. This does not evaluate judgment quality.')
    except (ValueError, OSError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
