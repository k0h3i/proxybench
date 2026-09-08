"""Source-selected XML packets with exact attachment-to-packet byte projection."""

import hashlib
import json

from proxybench.extraction.bundles import read_bundle
from proxybench.sources.npx_xml import FORM_NS, VOTE_NS, XMLRejected, parse_xml

VERSION = 'xml-slices-v1'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def build(primary, votes, *, primary_id, votes_id, index, input_id):
    """Keep the complete primary attachment and the selected vote with root context.

    Selection depends only on source structure, never on mapped field values.
    Prepared text is literal XML. Entity decoding is deliberately not applied.
    """
    parse_xml(primary, namespace=FORM_NS, root_name='edgarSubmission')
    table = parse_xml(votes, namespace=VOTE_NS, root_name='proxyVoteTable')
    if (table.text.strip() or any(n.name != 'proxyTable' for n in table.children)
            or type(index) is not int or not 0 <= index < len(table.children)):
        raise XMLRejected('Unsupported table context or target index')
    if primary_id == votes_id:
        raise ValueError('Attachments require distinct identities')
    row = table.children[index]
    # Preserve declarations and root namespaces, including prefixed namespaces.
    root_end = votes.index(b'>', table.start) + 1
    close_start = votes.rfind(b'</', row.end, table.end)
    if close_start < row.end:
        raise XMLRejected('Missing vote-table closing context')
    selections = [(primary_id, primary, 0, len(primary), 'complete primary attachment'),
                  (votes_id, votes, 0, root_end, 'vote attachment opening context'),
                  (votes_id, votes, row.start, row.end, 'selected vote entry'),
                  (votes_id, votes, close_start, len(votes), 'vote attachment closing context')]
    pieces, locations, blocks = [], [], []
    position = 0
    for i, (document, raw, start, end, role) in enumerate(selections):
        separator = f'\n[Generated boundary: block {i}; {role}]\n'.encode()
        pieces.append(separator)
        position += len(separator)
        copied = raw[start:end]
        pieces.append(copied)
        locations.append(dict(block_id=f'xml-{i}', attachment=dict(document_id=document,
            source_sha256=digest(raw), start_byte=start, end_byte=end),
            packet_start=position, packet_end=position + len(copied), role=role))
        position += len(copied)
    packet = b''.join(pieces)
    document_id = input_id + '-derived-packet'
    sha = digest(packet)
    for location in locations:
        start, end = location['packet_start'], location['packet_end']
        text = packet[start:end].decode('utf-8')
        blocks.append(dict(block_id=location['block_id'],
            span=dict(document_id=document_id, source_sha256=sha, start_byte=start, end_byte=end),
            original_text=text, prepared_text=text))
    target = locations[2]
    bundle = dict(input_version='fragment-input-v1', input_id=input_id, track='fragment',
        view_id=input_id + '-literal-view', document_id=document_id, encoding='utf-8',
        source_sha256=sha, target=dict(start_byte=target['packet_start'], end_byte=target['packet_end'],
        sha256=digest(votes[row.start:row.end])), blocks=blocks)
    read_bundle(json.dumps(bundle))
    lineage = dict(version=VERSION, packet_document_id=document_id, packet_sha256=sha,
        selected_physical_index=index, logical_boundary_review='PENDING', training_admitted=False,
        representation='Derived exact slices, not a complete original XML document',
        prepared_text='Literal UTF-8 XML, entities remain escaped', locations=locations)
    originals = {primary_id: primary, votes_id: votes}
    for block in blocks:
        project(block['span'], packet, lineage, originals)
    return dict(source_bytes=packet, bundle=bundle, lineage=lineage)


def project(span, packet, lineage, originals):
    """Reject changed bytes, unknown identities, invalid bounds, and separators."""
    if (lineage['version'] != VERSION or digest(packet) != lineage['packet_sha256']
            or span['document_id'] != lineage['packet_document_id']
            or span['source_sha256'] != lineage['packet_sha256']):
        raise ValueError('Packet identity differs')
    start, end = span['start_byte'], span['end_byte']
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(packet):
        raise ValueError('Invalid packet bounds')
    for location in lineage['locations']:
        a, b = location['packet_start'], location['packet_end']
        if a <= start < end <= b:
            source = location['attachment']
            raw = originals[source['document_id']]
            x, y = source['start_byte'], source['end_byte']
            if (type(x) is not int or type(y) is not int or not 0 <= x < y <= len(raw)
                    or y - x != b - a or digest(raw) != source['source_sha256']
                    or raw[x:y] != packet[a:b]):
                raise ValueError('Copied bytes differ from preserved attachment')
            # Byte positions inside a multibyte character cannot support text evidence.
            packet[start:end].decode('utf-8')
            return dict(document_id=source['document_id'], source_sha256=source['source_sha256'],
                        start_byte=x + start - a, end_byte=x + end - a)
    raise ValueError('Citation crosses a generated separator or leaves copied source')
