"""Read source-only fragments without opening source or reference paths."""

from proxybench.annotation.bindings import sha256
from proxybench.schemas.records import object_keys, require, strict_json, validate_span


def read_bundle(raw):
    bundle = strict_json(raw)
    require(isinstance(bundle, dict) and bundle.get('input_version') == 'fragment-input-v1'
            and bundle.get('track') == 'fragment', 'Unsupported fragment bundle')
    object_keys(bundle, {'input_version', 'input_id', 'track', 'view_id', 'document_id', 'encoding',
                         'source_sha256', 'target', 'blocks'}, 'source-only bundle')
    require(bundle.get('encoding') == 'utf-8', 'Only UTF-8 source slices are supported')
    require(isinstance(bundle.get('input_id'), str) and bool(bundle['input_id']), 'Missing input identity')
    blocks = bundle.get('blocks')
    require(isinstance(blocks, list) and bool(blocks), 'Missing source blocks')
    ids, ranges = set(), []
    target = bundle['target']
    object_keys(target, {'start_byte', 'end_byte', 'sha256'}, 'target')
    require(type(target['start_byte']) is int and type(target['end_byte']) is int
            and 0 <= target['start_byte'] < target['end_byte'], 'Invalid target range')
    target_bytes = None
    for block in blocks:
        object_keys(block, {'block_id', 'span', 'original_text', 'prepared_text'}, 'source block')
        require(isinstance(block['original_text'], str) and isinstance(block['prepared_text'], str), 'Source text must be strings')
        span = block['span']
        validate_span(span)
        require(span['document_id'] == bundle['document_id']
                and span['source_sha256'] == bundle['source_sha256'], 'Source block identity differs')
        require(isinstance(block['block_id'], str) and block['block_id'] not in ids, 'Duplicate source block')
        ids.add(block['block_id'])
        raw_slice = block['original_text'].encode('utf-8')
        a, b = span['start_byte'], span['end_byte']
        require(len(raw_slice) == b - a, 'Source slice byte length differs')
        require(all(b <= x or a >= y for x, y in ranges), 'Overlapping source blocks')
        ranges.append((a, b))
        if a <= target['start_byte'] < target['end_byte'] <= b:
            target_bytes = raw_slice[target['start_byte'] - a:target['end_byte'] - a]
    require(target_bytes is not None and sha256(target_bytes) == target['sha256'], 'Target leaves source or hash differs')
    return bundle
