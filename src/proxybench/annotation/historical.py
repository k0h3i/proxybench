"""Prepare reviewed historical ranges without extracting voting facts."""

import html
from html.parser import HTMLParser
from pathlib import Path
import re

from proxybench.annotation.bindings import canonical_bytes, sha256

VIEW_VERSION = 'historical-cells-v1'
PARTITIONS = {'training', 'development', 'test'}


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style', 'iframe', 'object', 'table', 'td', 'th', 'tr'}:
            raise ValueError('Unsupported nested or active HTML in a source cell')
        if tag in {'br', 'p', 'div', 'li'}:
            self.parts.append(' ')

    def handle_endtag(self, tag):
        if tag in {'p', 'div', 'li'}:
            self.parts.append(' ')

    def handle_data(self, data):
        self.parts.append(data)


def visible_text(markup):
    parser = _Text()
    parser.feed(markup)
    parser.close()
    return ' '.join(''.join(parser.parts).split())


def source_block(raw, start, end, kind, encoding):
    """A row preserves separate cells, including empty and spanning cells."""
    text = raw[start:end].decode(encoding, errors='strict')
    cells = []
    if kind == 'row':
        if not re.fullmatch(r'\s*<tr\b[^>]*>.*</tr\s*>\s*', text, re.I | re.S):
            raise ValueError('A row range must contain a complete HTML row')
        if len(re.findall(r'<tr\b', text, re.I)) != 1:
            raise ValueError('Nested rows require a separate representation review')
        matches = list(re.finditer(r'<(td|th)\b([^>]*)>(.*?)</\1\s*>', text, re.I | re.S))
        residue = re.sub(r'^\s*<tr\b[^>]*>|</tr\s*>\s*$', '', text, flags=re.I | re.S)
        residue = re.sub(r'<(td|th)\b[^>]*>.*?</\1\s*>', '', residue, flags=re.I | re.S)
        if not matches or residue.strip():
            raise ValueError('Unmapped content or incomplete cells in an HTML row')
        for match in matches:
            attrs = dict((k.lower(), v.strip('\"\'')) for k, v in re.findall(
                r'(colspan|rowspan)\s*=\s*(\"[^\"]*\"|\'[^\']*\'|[^\s>]+)', match[2], re.I))
            if any(not v.isdigit() or int(v) < 1 for v in attrs.values()):
                raise ValueError('Invalid cell span')
            cells.append(dict(text=visible_text(match[3]), **attrs,
                              start_byte=start + len(text[:match.start()].encode(encoding)),
                              end_byte=start + len(text[:match.end()].encode(encoding))))
    elif kind == 'cell':
        # Some filings omit the opening row tag around a context heading.
        match = re.fullmatch(r'\s*<(td|th)\b([^>]*)>(.*?)</\1\s*>\s*', text, re.I | re.S)
        if not match:
            raise ValueError('A context cell requires one complete HTML cell')
        attrs = dict((k.lower(), v.strip('\"\'')) for k, v in re.findall(
            r'(colspan|rowspan)\s*=\s*(\"[^\"]*\"|\'[^\']*\'|[^\s>]+)', match[2], re.I))
        if any(not v.isdigit() or int(v) < 1 for v in attrs.values()):
            raise ValueError('Invalid cell span')
        cells.append(dict(text=visible_text(match[3]), **attrs, start_byte=start, end_byte=end))
    elif kind == 'block':
        # Callers supply one paragraph or heading, never a table or joined passages.
        if not re.fullmatch(r'\s*<(p|div)\b[^>]*>.*</\1\s*>\s*', text, re.I | re.S):
            raise ValueError('A block requires one complete paragraph or heading')
        if len(re.findall(r'<(?:p|div)\b', text, re.I)) != 1:
            raise ValueError('Separate paragraphs must remain separate source blocks')
        cells.append(dict(text=visible_text(text), start_byte=start, end_byte=end))
    elif kind == 'text':
        if '\n\n' in text.strip():
            raise ValueError('Separate text paragraphs must remain separate source blocks')
        cells.append(dict(text=text, start_byte=start, end_byte=end))
    else:
        raise ValueError('Unknown historical block kind')
    return dict(start_byte=start, end_byte=end, kind=kind, cells=cells,
                source_slice_sha256=sha256(raw[start:end]),
                line_start=raw[:start].count(b'\n') + 1,
                line_end=raw[:end - 1].count(b'\n') + 1)


def context_blocks(root, selection):
    """Preserve bounded context from other documents in the same filing."""
    blocks, lines, views = [], [], []
    for document_index, context in enumerate(selection.get('context_sources', []), 1):
        if context.get('accession') != selection['accession'] or not context.get('association_review'):
            raise ValueError('Context document needs the same accession and reviewed association')
        path = Path(context['source_path'])
        if path.is_absolute() or '..' in path.parts or not (root / path).resolve().is_relative_to(root):
            raise ValueError('Context source path leaves the workspace')
        source = root / path
        if source.stat().st_size > 100 * 1024 ** 2:
            raise ValueError('Context source exceeds the 100 MiB input limit')
        raw = source.read_bytes()
        if sha256(raw) != context['source_sha256']:
            raise ValueError('Context source hash changed')
        spans = context['spans']
        if (not spans or any(not 0 <= a < b <= len(raw) for a, b, _ in spans)
                or any(left[1] > right[0] for left, right in zip(spans, spans[1:]))):
            raise ValueError('Context ranges are empty, overlapping, or out of order')
        previous = 0
        for index, (a, b, kind) in enumerate(spans, 1):
            block = source_block(raw, a, b, kind, context.get('encoding', 'utf-8'))
            block.update(block_id=f'D{document_index}:B{index}', target=False, source_path=path.as_posix())
            blocks.append(block)
            if a > previous:
                gap = f'[OMITTED D{document_index} SOURCE BYTES {previous}:{a}]'
                lines.append(gap)
                views.append('<p>' + html.escape(gap) + '</p>')
            import json
            title = f"{block['block_id']} source bytes {a}:{b}"
            rendered = [json.dumps(c['text'], ensure_ascii=False) + ''.join(
                f' {key}={c[key]}' for key in ('colspan', 'rowspan') if key in c) for c in block['cells']]
            lines.append(title + '\n' + ' | '.join(rendered))
            body = ''.join('<td' + ''.join(f' {key}="{int(c[key])}"' for key in ('colspan', 'rowspan') if key in c)
                           + '>' + html.escape(c['text']) + '</td>' for c in block['cells'])
            views.append('<section><p>' + html.escape(title) + '</p><table><tr>' + body + '</tr></table></section>')
            previous = b
        if previous < len(raw):
            gap = f'[OMITTED D{document_index} SOURCE BYTES {previous}:{len(raw)}]'
            lines.append(gap)
            views.append('<p>' + html.escape(gap) + '</p>')
    return blocks, lines, views


def prepare_historical(root, selection):
    """Build a packet from source-reviewed blocks and one target range.

    The selection supplies boundary evidence. This function does not infer it.
    Shared-row subtargets remain unsupported and must be deferred.
    """
    root = Path(root).resolve()
    source_path = Path(selection['source_path'])
    if source_path.is_absolute() or '..' in source_path.parts:
        raise ValueError('Source path must be project-relative without parent traversal')
    path = (root / source_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Source path leaves the workspace')
    if path.stat().st_size > 100 * 1024 ** 2:
        raise ValueError('Source exceeds the 100 MiB input limit')
    raw = path.read_bytes()
    if sha256(raw) != selection['source_sha256']:
        raise ValueError('Source hash changed')
    if selection['split'] not in PARTITIONS:
        raise ValueError('Unknown historical source partition')
    if not selection.get('boundary_review') or not selection.get('reviewer'):
        raise ValueError('A reviewed logical target boundary is required')
    ranges = selection['spans']
    if not ranges or any(not 0 <= a < b <= len(raw) for a, b, _ in ranges):
        raise ValueError('Source ranges are empty or out of bounds')
    if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
        raise ValueError('Source ranges overlap or are out of order')
    target = selection['target']
    marked = [i for i, (a, b, _) in enumerate(ranges) if target[0] <= a < b <= target[1]]
    if any(ranges[i][2] == 'cell' for i in marked):
        raise ValueError('Standalone cells provide context, not shared-row targets')
    if (not marked or ranges[marked[0]][0] != target[0] or ranges[marked[-1]][1] != target[1]
            or any(ranges[i][1] != ranges[i + 1][0] for i in marked[:-1])):
        raise ValueError('Target must cover complete adjacent blocks. Defer shared-row subtargets.')
    blocks, lines, views = context_blocks(root, selection)
    document_prefix = f'D{len(selection["context_sources"]) + 1}:' if selection.get('context_sources') else ''
    previous = 0
    for index, (a, b, kind) in enumerate(ranges):
        block = source_block(raw, a, b, kind, selection.get('encoding', 'utf-8'))
        block.update(block_id=f'{document_prefix}B{index + 1}', target=index in marked)
        if document_prefix:
            block['source_path'] = source_path.as_posix()
        blocks.append(block)
        if a > previous:
            gap = f'[OMITTED SOURCE BYTES {previous}:{a}]'
            lines.append(gap)
            views.append('<p>' + html.escape(gap) + '</p>')
        title = f"{block['block_id']} source bytes {a}:{b}"
        if index == marked[0]:
            lines.append('BEGIN MARKED TARGET')
        cells = [c['text'] for c in block['cells']]
        # JSON string quoting distinguishes source characters from our cell separators.
        import json
        rendered = []
        for c in block['cells']:
            span = ''.join(f' {k}={c[k]}' for k in ('colspan', 'rowspan') if k in c)
            rendered.append(json.dumps(c['text'], ensure_ascii=False) + span)
        lines.append(title + '\n' + ' | '.join(rendered))
        if index == marked[-1]:
            lines.append('END MARKED TARGET')
        style = ' style="background:#fff0ad"' if index in marked else ''
        body = ''.join('<td' + ''.join(f' {k}="{int(c[k])}"' for k in ('colspan', 'rowspan') if k in c)
                       + '>' + html.escape(c['text']) + '</td>' for c in block['cells'])
        views.append('<section' + style + '><p>' + html.escape(title) +
                     (' [MARKED TARGET]' if index in marked else '') + '</p><table><tr>' + body + '</tr></table></section>')
        previous = b
    if previous < len(raw):
        gap = f'[OMITTED SOURCE BYTES {previous}:{len(raw)}]'
        lines.append(gap)
        views.append('<p>' + html.escape(gap) + '</p>')
    source = '\n'.join(lines)
    model_input = source
    view = ('<!doctype html><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" '
            'content="default-src \'none\'; style-src \'unsafe-inline\'">'
            '<style>body{font:16px sans-serif}td{border:1px solid #aaa;padding:6px;white-space:pre-wrap}'
            'table{border-collapse:collapse}section{margin:15px 0}</style>' + ''.join(views))
    manifest = dict(selection, packet_version=VIEW_VERSION, blocks=blocks,
                    view_version=VIEW_VERSION)
    manifest['target_id'] = sha256(canonical_bytes([selection['accession'], selection['source_sha256'], target]))
    return dict(manifest=manifest, source_view=view, model_input=model_input)


def check_packet(root, packet):
    rebuilt = prepare_historical(root, packet['manifest'])
    if any(rebuilt[k] != packet[k] for k in ('manifest', 'source_view', 'model_input')):
        raise ValueError('Packet differs from its original source or source rendering')


def literal_support(label, packet):
    """Check quotations only. This does not establish semantic support."""
    passages = [c['text'] for b in packet['manifest']['blocks'] for c in b['cells']]
    def visit(value, path):
        if isinstance(value, dict):
            if 'raw_text' in value:
                raw = value['raw_text']
                if raw is not None and (not raw or not any(raw in p for p in passages)):
                    raise ValueError(f'{path}: quotation is not contiguous within one source cell')
            for key, child in value.items():
                visit(child, f'{path}.{key}')
        elif isinstance(value, list):
            for i, child in enumerate(value):
                visit(child, f'{path}[{i}]')
    visit(label, 'label')
