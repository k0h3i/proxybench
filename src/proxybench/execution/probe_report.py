"""Score completed local probes without changing outputs or development references."""

import argparse
import json
from pathlib import Path

from proxybench.execution.runner import ScheduledInput, preflight, write_json
from proxybench.evaluation.scoring import parsed_response, score_fragment
from proxybench.normalization.values import ACTIVE_GUIDE


def score_probe(root, schedule, captures):
    results = []
    for entry in schedule:
        item = ScheduledInput(**entry)
        _, reference, context = preflight(root, item, ACTIVE_GUIDE)
        directory = Path(captures) / item.input_id
        raw_path, capture_path = directory / 'raw.txt', directory / 'capture.json'
        raw = raw_path.read_bytes() if raw_path.exists() else b''
        capture = json.loads(capture_path.read_text()) if capture_path.exists() else None
        score = score_fragment(raw, reference, context=context)
        parsed, count, normalized, errors, envelope = parsed_response(raw, guide_version=ACTIVE_GUIDE)
        complete = bool(envelope and parsed['status'] == 'COMPLETE' and count == 1 and not errors)
        evidence_valid = all(r['citations']['valid'] == r['citations']['submitted']
                             and r['citations']['missing_required'] == 0 for r in score['evidence']['records'])
        usable = bool(capture and capture['terminated'] and complete and not score['metadata_errors']
                      and not score['anchor_errors'] and evidence_valid)
        categories = []
        if capture is None:
            categories.append('NOT_COMPLETED')
        elif not capture['terminated']:
            categories.append('TRUNCATION')
        if raw and parsed is None:
            categories.append('SERIALIZATION')
        if errors and parsed is not None:
            categories.append('MISSING_FIELDS' if any('required object keys differ' in error for error in errors) else 'CONTRACT_ERROR')
        if not evidence_valid or score['anchor_errors']:
            categories.append('EVIDENCE_ERROR')
        if 'FIELD_MISMATCH' in score['reasons']:
            categories.append('FIELD_DISAGREEMENT')
        results.append({'input_id': item.input_id, 'capture': capture, 'contract_usable': usable,
                        'whole_record_match': bool(capture and capture['terminated'] and score['passed']),
                        'failure_categories': categories, 'score': score})
    return {'purpose': 'ASSISTED_DEVELOPMENT_FEASIBILITY', 'scheduled': len(schedule),
            'completed': sum(r['capture'] is not None for r in results),
            'contract_usable': sum(r['contract_usable'] for r in results),
            'whole_record_matches': sum(r['whole_record_match'] for r in results),
            'field_matches': sum(sum(r['score']['field_matches'].values()) for r in results if r['capture'] is not None),
            'field_total': sum(r['score']['included_fields'] for r in results),
            'unsupported_facts': 'NOT_ASSESSED; field disagreement alone does not establish unsupported source assertions',
            'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('schedule', type=Path)
    parser.add_argument('captures', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    write_json(args.output, score_probe(args.root, json.loads(args.schedule.read_text()), args.captures))


if __name__ == '__main__':
    main()
