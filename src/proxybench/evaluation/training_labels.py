"""Frozen source-value comparisons for the fourteen-field training contract."""

import json
import re

from proxybench.training.labels import read_json, validate

SCORER_VERSION = 'historical-pilot-v1'


def text(value, path):
    value = re.sub(r'\s+', ' ', value).strip()
    if path == 'issuer_name' or path in {'reporting_scope.name', 'reporting_scope.members'}:
        value = value.casefold()
    return value


def normalized(node, path, *, origins=False, quotes=False, omit=()):
    if path in omit:
        return None
    if isinstance(node, str):
        return text(node, path)
    if isinstance(node, list):
        # Sorting retains multiplicity. A set would hide duplicate components.
        return sorted((normalized(n, path, origins=origins, quotes=quotes, omit=omit) for n in node),
                      key=lambda n: json.dumps(n, sort_keys=True))
    if not isinstance(node, dict):
        return node
    if 'availability' in node:
        result = dict(availability=node['availability'],
                      value=normalized(node['value'], path, origins=origins, quotes=quotes, omit=omit))
        if origins:
            result['origin'] = node['origin']
        if quotes or node['value'] == 'OTHER':
            result['raw_text'] = text(node['raw_text'], path) if node['raw_text'] is not None else None
        return result
    return {key: normalized(value, path+'.'+key, origins=origins, quotes=quotes, omit=omit)
            for key, value in node.items() if path+'.'+key not in omit}


def primary_omissions(reference):
    omit = ['reporting_scope.scope_type']
    if reference['fields']['participation']['origin'] == 'DERIVED':
        omit.append('participation')
    return omit


def parse_answer(answer):
    if answer.get('status') != 'COMPLETE':
        return None
    try:
        return validate(read_json(answer['text']))
    except (ValueError, TypeError, KeyError):
        return None


def quote_paths(node, path=''):
    if isinstance(node, dict):
        if 'raw_text' in node and node['raw_text']:
            yield path, node['raw_text']
        for key, value in node.items():
            if key != 'raw_text':
                yield from quote_paths(value, path+'.'+key if path else key)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from quote_paths(value, f'{path}[{index}]')


def score(reference, answer, source, *, subject_equivalent=False, quotation_errors=None):
    validate(reference)
    prediction = parse_answer(answer)
    if prediction is None:
        return dict(format_valid=False, exact=False, source_value_correct=False,
                    field_correct={key: False for key in reference['fields']},
                    primary_errors=['invalid_or_incomplete_output'], origin_errors=[],
                    derivation_errors=[], unsupported_quotes=[], quotation_errors=quotation_errors)
    omit = primary_omissions(reference)
    field_correct, origins, derivations = {}, [], []
    for key, expected in reference['fields'].items():
        actual = prediction['fields'][key]
        same = normalized(expected, key, omit=omit) == normalized(actual, key, omit=omit)
        if key == 'separate_subject' and subject_equivalent:
            same = expected['availability'] == actual['availability'] == 'PRESENT'
        field_correct[key] = same
        # Report origin and derivation errors independently from primary values.
        if origin_tree(expected, key) != origin_tree(actual, key):
            origins.append(key)
        if key == 'reporting_scope':
            a, b = expected['value'], actual['value']
            if isinstance(a, dict) and (not isinstance(b, dict) or a['scope_type'] != b['scope_type']):
                derivations.append('reporting_scope.scope_type')
        if key == 'participation' and key in omit and normalized(expected, key) != normalized(actual, key):
            derivations.append(key)
    source = re.sub(r'\s+', ' ', source).strip()
    unsupported = [path for path, quote in quote_paths(prediction['fields'])
                   if re.sub(r'\s+', ' ', quote).strip() not in source]
    errors = [key for key, correct in field_correct.items() if not correct]
    return dict(format_valid=True, exact=reference == prediction, source_value_correct=not errors,
                field_correct=field_correct, primary_errors=errors, origin_errors=origins,
                derivation_errors=derivations, unsupported_quotes=unsupported,
                quotation_errors=quotation_errors, quotation_semantics='REVIEW_REQUIRED' if quotation_errors is None else 'REVIEWED')


def origin_tree(node, path):
    if isinstance(node, list):
        # Pair each component with its source value before comparing origins.
        return [origin_tree(n, path) for n in sorted(node, key=lambda n: json.dumps(normalized(n, path), sort_keys=True))]
    if isinstance(node, dict):
        if 'availability' in node:
            return dict(origin=node['origin'], children=origin_tree(node['value'], path))
        return {key: origin_tree(value, path+'.'+key) for key, value in node.items()}
    return None


def terminal(answer):
    if answer.get('status') == 'LENGTH_STOP':
        return 'length'
    return answer.get('status')


def export_comparison(reference, candidate):
    """Admit exact behavior, or request source review for narrowly allowed differences."""
    if reference.get('prompt_token_ids') != candidate.get('prompt_token_ids'):
        return dict(status='FAILED', reason='Prompt token IDs differ')
    if terminal(reference) == 'length' or terminal(candidate) == 'length':
        passed = terminal(reference) == terminal(candidate) and reference.get('token_ids') == candidate.get('token_ids')
        return dict(status='PASS' if passed else 'FAILED', reason='Output limit comparison')
    if reference.get('status') != 'COMPLETE' or candidate.get('status') != 'COMPLETE':
        return dict(status='FAILED', reason='Incomplete request')
    left, right = parse_answer(reference), parse_answer(candidate)
    if left is None or right is None:
        passed = left is None and right is None and reference['text'] == candidate['text']
        return dict(status='PASS' if passed else 'FAILED', reason='Malformed output comparison')
    # No origin/derivation changes are permitted for engine admission.
    changed = []
    for key in left['fields']:
        a, b = left['fields'][key], right['fields'][key]
        if normalized(a, key, origins=True, quotes=True) == normalized(b, key, origins=True, quotes=True):
            continue
        changed.append(key)
        if origin_tree(a, key) != origin_tree(b, key):
            return dict(status='FAILED', reason='Origin or structure differs')
        if key != 'separate_subject' and normalized(a, key, origins=True) != normalized(b, key, origins=True):
            return dict(status='FAILED', reason='Source value or multiplicity differs')
        if key == 'separate_subject' and (a['availability'] != 'PRESENT' or b['availability'] != 'PRESENT'):
            return dict(status='FAILED', reason='Subject availability differs')
    return dict(status='REVIEW_REQUIRED' if changed else 'PASS', changed_fields=changed)


def paired_report(cases):
    """Keep all targets in the denominator, including malformed answers."""
    def summary(rows):
        result = dict(targets=len(rows), wins=0, losses=0, ties=0)
        for model in ('original', 'trained'):
            result[model] = {key: sum(bool(row[model][key]) for row in rows)
                             for key in ('format_valid', 'exact', 'source_value_correct')}
            result[model]['fields'] = {key: sum(row[model]['field_correct'][key] for row in rows)
                                      for key in rows[0][model]['field_correct']} if rows else {}
            for key in ('origin_errors', 'derivation_errors', 'unsupported_quotes', 'quotation_errors'):
                result[model][key] = sum(len(row[model][key] or []) for row in rows)
        for row in rows:
            a, b = row['original']['source_value_correct'], row['trained']['source_value_correct']
            result['ties' if a == b else 'wins' if b else 'losses'] += 1
        return result
    return dict(scorer=SCORER_VERSION, aggregate=summary(cases), cases=cases,
                families={family: summary([r for r in cases if r['family'] == family])
                          for family in sorted({r['family'] for r in cases})},
                interpretation='Related development targets, not an independent test set')
