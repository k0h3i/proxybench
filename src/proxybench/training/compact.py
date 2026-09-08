"""Reversible compact development records; original source bytes stay unchanged."""

from copy import deepcopy
import json

from proxybench.extraction.bundles import read_bundle
from proxybench.normalization.values import ACTIVE_GUIDE
from proxybench.schemas.records import (
    D2_RULE, EVIDENCE_KEYS, FIELD_TYPES, SPAN_KEYS, WRAPPER_KEYS,
    normalize_record, object_keys, require, strict_json, validate_envelope, validate_span,
)


VERSION = 'compact-fragment-v1'
DEFAULT = dict(value=None, raw_text=None, availability='ABSENT_IN_CONTEXT', origin=None,
               evidence=[], reason=None, rule_id=None)
KEYS = {'value': 'v', 'raw_text': 'raw', 'availability': 'state', 'origin': 'origin',
        'evidence': 'e', 'reason': 'reason', 'rule_id': 'rule'}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def compact_source(raw):
    """Select no source content using answers; retain all blocks and both views."""
    bundle = read_bundle(raw)
    blocks = []
    target = None
    for i, block in enumerate(bundle['blocks']):
        gap = i > 0 and bundle['blocks'][i - 1]['span']['end_byte'] != block['span']['start_byte']
        blocks.append({'id': i, 'gap_before': gap, 'source': block['original_text'], 'text': block['prepared_text']})
        start, end = block['span']['start_byte'], block['span']['end_byte']
        if start <= bundle['target']['start_byte'] < bundle['target']['end_byte'] <= end:
            target = [i, bundle['target']['start_byte'] - start, bundle['target']['end_byte'] - start]
    return {'target': target, 'blocks': blocks}


def encode_span(span, bundle):
    object_keys(span, SPAN_KEYS, 'span')
    for i, block in enumerate(bundle['blocks']):
        basis = block['span']
        if (span['document_id'] == basis['document_id'] and span['source_sha256'] == basis['source_sha256']
                and basis['start_byte'] <= span['start_byte'] < span['end_byte'] <= basis['end_byte']):
            return [i, span['start_byte'] - basis['start_byte'], span['end_byte'] - basis['start_byte']]
    raise ValueError('Citation leaves the supplied source boundary')


def decode_span(value, bundle):
    require(isinstance(value, list) and len(value) == 3 and all(type(x) is int for x in value),
            'Span requires three integer coordinates')
    i, start, end = value
    require(0 <= i < len(bundle['blocks']), 'Unknown source block')
    span = bundle['blocks'][i]['span']
    require(0 <= start < end <= span['end_byte'] - span['start_byte'], 'Span leaves source block')
    return dict(document_id=span['document_id'], source_sha256=span['source_sha256'],
                start_byte=span['start_byte'] + start, end_byte=span['start_byte'] + end)


def encode_evidence(evidence, bundle):
    object_keys(evidence, EVIDENCE_KEYS, 'evidence')
    span = encode_span({k: evidence[k] for k in SPAN_KEYS}, bundle)
    block = bundle['blocks'][span[0]]
    require(evidence['view_id'] in (None, bundle['view_id'])
            and evidence['block_id'] in (None, block['block_id']), 'Unknown evidence view')
    return {'span': span, 'basis': evidence['quote_basis'], 'quote': evidence['quote'],
            'view': evidence['view_id'] is not None, 'block': evidence['block_id'] is not None}


def decode_evidence(value, bundle):
    object_keys(value, {'span', 'basis', 'quote', 'view', 'block'}, 'compact evidence')
    require(type(value['view']) is bool and type(value['block']) is bool, 'Evidence flags must be boolean')
    span = decode_span(value['span'], bundle)
    block = bundle['blocks'][value['span'][0]]
    result = dict(**span, quote_basis=value['basis'], quote=value['quote'],
                view_id=bundle['view_id'] if value['view'] else None,
                block_id=block['block_id'] if value['block'] else None)
    validate_span(result, evidence=True)
    if value['basis'] == 'PREPARED_VIEW':
        require(span == block['span'] and value['quote'] in block['prepared_text'], 'Invalid prepared quotation')
    elif value['basis'] == 'ORIGINAL_DECODED':
        _, start, end = value['span']
        text = block['original_text'].encode('utf-8')[start:end].decode('utf-8')
        require(value['quote'] in text, 'Quotation does not occur in cited bytes')
    return result


def encode_record(record, raw_bundle):
    """Retain every semantic and evidence field; administrative record ID is external."""
    bundle = read_bundle(raw_bundle)
    normalize_record(record, guide_version=ACTIVE_GUIDE)
    require(record['enrichments'] == [], 'Compact v1 does not yet admit enrichment records')
    citations = []

    def field(value):
        if not isinstance(value, dict):
            return [field(x) for x in value] if isinstance(value, list) else value
        if set(value) != WRAPPER_KEYS:
            return {key: field(child) for key, child in value.items()}
        if value == DEFAULT:
            return None
        require(value['rule_id'] != 'D2', 'Literal D2 rule ID conflicts with the compact reserved alias')
        defaults = DEFAULT | ({'availability': 'PRESENT', 'origin': 'EXTRACTED'}
                              if value['availability'] == 'PRESENT' else {})
        result = {}
        for key, child in value.items():
            if child == defaults[key] and not (key == 'availability' and child != 'PRESENT'):
                continue
            if key == 'value':
                child = field(child)
            elif key == 'evidence':
                ids = []
                for evidence in child:
                    item = encode_evidence(evidence, bundle)
                    if item not in citations:
                        citations.append(item)
                    ids.append(citations.index(item))
                child = ids
            elif key == 'rule_id' and child == D2_RULE:
                child = 'D2'
            result[KEYS[key]] = child
        return result

    anchor = record['source_anchor']
    result = {'fields': {key: field(value) for key, value in record['fields'].items()},
              'citations': citations,
              'anchor': None if anchor is None else {k: [encode_span(s, bundle) for s in v] for k, v in anchor.items()},
              'enrichments': [{'field_path': x['field_path'], 'field': field(x['field'])} for x in record['enrichments']]}
    return result


def encode_response(response, raw_bundle):
    validate_envelope(response)
    bundle = read_bundle(raw_bundle)
    require(response['input_id'] == bundle['input_id'], 'Response input identity differs')
    require(response['status'] != 'COMPLETE' or len(response['records']) == 1, 'Expected one logical record')
    return {'status': response['status'], 'records': [encode_record(r, raw_bundle) for r in response['records']],
            'failure': deepcopy(response['failure'])}


def decode_response(raw, raw_bundle):
    compact, bundle = strict_json(raw), read_bundle(raw_bundle)
    object_keys(compact, {'status', 'records', 'failure'}, 'compact response')
    require(isinstance(compact['records'], list), 'Records must be a list')
    require(compact['status'] != 'COMPLETE' or len(compact['records']) == 1, 'Expected one logical record')
    response = dict(schema_version='benchmark-v1', input_id=bundle['input_id'], status=compact['status'],
                    records=[decode_record(dumps(r), raw_bundle, record_id=f"{bundle['input_id']}-target-{i}")
                             for i, r in enumerate(compact['records'])], failure=compact['failure'])
    validate_envelope(response)
    return response


def decode_record(raw, raw_bundle, *, record_id):
    """Expand only declared syntax defaults and source metadata, never reference answers."""
    bundle, compact = read_bundle(raw_bundle), strict_json(raw)
    object_keys(compact, {'fields', 'citations', 'anchor', 'enrichments'}, 'compact record')
    object_keys(compact['fields'], FIELD_TYPES, 'compact fields')
    require(isinstance(compact['citations'], list), 'Citations must be a list')
    citations = [decode_evidence(x, bundle) for x in compact['citations']]

    def field(value, kind):
        if value is None:
            return deepcopy(DEFAULT)
        require(isinstance(value, dict) and set(value) <= set(KEYS.values()), 'Unknown compact field keys')
        state = value.get('state', 'PRESENT')
        result = deepcopy(DEFAULT)
        result.update(availability=state, origin='EXTRACTED' if state == 'PRESENT' else None)
        for key, short in KEYS.items():
            if short not in value:
                continue
            child = value[short]
            if key == 'value' and state == 'PRESENT':
                if isinstance(kind, dict):
                    object_keys(child, kind, 'compact container')
                    child = {k: field(v, kind[k]) for k, v in child.items()}
                elif isinstance(kind, list):
                    require(isinstance(child, list), 'Compact list field must contain a list')
                    item_kind = kind[0]
                    expanded = []
                    for item in child:
                        if isinstance(item_kind, dict):
                            object_keys(item, item_kind, 'compact list member')
                            expanded.append({k: field(v, item_kind[k]) for k, v in item.items()})
                        else:
                            expanded.append(field(item, item_kind))
                    child = expanded
            elif key == 'evidence':
                require(isinstance(child, list) and all(type(i) is int and 0 <= i < len(citations) for i in child),
                        'Unknown citation index')
                child = [deepcopy(citations[i]) for i in child]
            elif key == 'rule_id' and child == 'D2':
                child = D2_RULE
            result[key] = child
        return result

    anchor = compact['anchor']
    if anchor is not None:
        object_keys(anchor, {'subject_spans', 'scope_spans'}, 'compact anchor')
        require(all(isinstance(v, list) for v in anchor.values()), 'Anchor spans must be lists')
        anchor = {k: [decode_span(s, bundle) for s in v] for k, v in anchor.items()}
    # Enrichment kinds are obtained from the core structure, never external labels.
    require(compact['enrichments'] == [], 'Compact v1 does not yet admit enrichment records')
    result = dict(record_id=record_id, source_anchor=anchor,
                  fields={k: field(v, FIELD_TYPES[k]) for k, v in compact['fields'].items()}, enrichments=[])
    normalize_record(result, guide_version=ACTIVE_GUIDE)
    return result
