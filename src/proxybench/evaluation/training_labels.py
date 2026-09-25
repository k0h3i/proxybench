"""Frozen source-value comparisons for the fourteen-field training contract."""

import json
import re

from proxybench.training.labels import read_json, validate

SCORER_VERSION = 'source-values-v1'


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
