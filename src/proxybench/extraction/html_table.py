"""A finite, source-only HTML row parser. No file or reference access."""

from dataclasses import dataclass, field as dataclass_field
from datetime import date
from html.parser import HTMLParser
import json
import re
import sys

from proxybench.extraction.bundles import read_bundle
from proxybench.normalization.values import ACTIVE_GUIDE, calendar_date, quantity, scalar, text
from proxybench.schemas.records import D2_RULE, FIELD_TYPES, ValidationError, require, normalize_record

PARSER_VERSION = 'html-row-v1'
ALIASES = {
    'issue no.': 'proposal_number', 'ballot issue': 'proposal_number',
    'proposal no': 'proposal_number', 'proposal number': 'proposal_number',
    'description': 'raw_description', 'proposal': 'raw_description',
    'proponent': 'proposal_source', 'proposed by': 'proposal_source',
    'vote cast': 'vote_components', 'mgmt rec': 'management_recommendation',
    'management recommendation': 'management_recommendation',
    'for/agnst mgmt': 'alignment', 'issuer': 'issuer_name', 'ticker': 'ticker',
    'cusip': 'security_identifiers', 'cusip9': 'security_identifiers',
    'security id': 'security_identifiers', 'meeting date': 'meeting_date',
    'meeting date (dd/mm/yyyy)': 'meeting_date', 'meeting type': 'meeting_type',
    'meeting status': 'participation',
}


@dataclass
class Node:
    tag: str
    start: int
    end: int = 0
    attrs: dict = dataclass_field(default_factory=dict)
    parts: list = dataclass_field(default_factory=list)
    children: list = dataclass_field(default_factory=list)

    @property
    def value(self):
        return text(' '.join(self.parts))


class Structure(HTMLParser):
    def __init__(self, block):
        super().__init__(convert_charrefs=True)
        self.block = block
        self.source = block['original_text']
        self.lines = [0]
        for match in re.finditer('\n', self.source):
            self.lines.append(match.end())
        self.stack, self.nodes = [], []
        self.feed(self.source)
        self.close()
        require(not any(n.tag in ('td', 'th', 'tr') for n in self.stack), 'Incomplete table structure')

    def position(self):
        line, column = self.getpos()
        return self.lines[line - 1] + column

    def handle_starttag(self, tag, attrs):
        if tag in ('br', 'hr', 'img', 'meta', 'link', 'input', 'wbr'):
            for node in self.stack:
                node.parts.append('\n' if tag == 'br' else ' ')
            return
        node = Node(tag, self.position(), attrs=dict(attrs))
        if self.stack:
            self.stack[-1].children.append(node)
        self.nodes.append(node)
        self.stack.append(node)

    def handle_endtag(self, tag):
        if not any(n.tag == tag for n in self.stack):
            return
        while self.stack:
            node = self.stack.pop()
            node.end = self.source.find('>', self.position()) + 1
            if node.tag == tag:
                break
            require(node.tag not in ('td', 'th', 'tr'), 'Malformed table nesting')

    def handle_data(self, value):
        # Footnote letters are not column headings or cell values.
        if any(n.tag in ('sup', 'script', 'style') for n in self.stack):
            return
        for node in self.stack:
            node.parts.append(value)

    def span(self, node):
        a = self.block['span']['start_byte']
        return {**self.block['span'], 'start_byte': a + len(self.source[:node.start].encode()),
                'end_byte': a + len(self.source[:node.end].encode())}

    def evidence(self, node):
        return {**self.span(node), 'view_id': None, 'block_id': None,
                'quote': self.source[node.start:node.end], 'quote_basis': 'ORIGINAL_DECODED'}


def wrapped(value=None, evidence=(), *, raw=None, availability=None, reason=None, origin=None, rule_id=None):
    return {'value': value, 'raw_text': raw, 'availability': availability or ('PRESENT' if value is not None else 'ABSENT_IN_CONTEXT'),
            'origin': origin or ('EXTRACTED' if value is not None else None), 'evidence': list(evidence),
            'reason': reason, 'rule_id': rule_id}


def unresolved(raw, evidence, reason, availability='AMBIGUOUS'):
    return wrapped(raw=raw, evidence=evidence, availability=availability, reason=reason)


def key(node):
    return node.value.rstrip(':').casefold()


def cells(row):
    result = [n for n in row.children if n.tag in ('td', 'th')]
    require(not any(n.tag == 'table' for c in result for n in c.children), 'Nested tables are unsupported')
    require(all(c.attrs.get('rowspan', '1') == '1' for c in result), 'Row spans are unsupported')
    return result


def columns(row):
    output = []
    for cell in cells(row):
        count = cell.attrs.get('colspan', '1')
        require(str(count).isdigit() and 1 <= int(count) <= 32, 'Unsupported column span')
        output.append(cell)
        output.extend([None] * (int(count) - 1))
    return output


def date_from_heading(value, heading):
    if key(heading) == 'meeting date (dd/mm/yyyy)':
        match = re.fullmatch(r'([0-9]{1,2})/([0-9]{1,2})/([0-9]{4})', value.strip())
        if not match:
            raise ValueError('Unsupported explicit day/month syntax')
        day, month, year = map(int, match.groups())
        return date(year, month, day).isoformat()
    return calendar_date(value, guide_version=ACTIVE_GUIDE)


def extract(raw):
    bundle = read_bundle(raw)
    response = {'schema_version': 'benchmark-v1', 'input_id': bundle['input_id'], 'status': 'ABSTAINED',
                'records': [], 'failure': None}
    try:
        response['records'] = [_extract(bundle)]
        response['status'] = 'COMPLETE'
    except (ValidationError, ValueError) as error:
        response['failure'] = {'code': 'UNSUPPORTED_LAYOUT', 'message': str(error), 'stage': 'extraction', 'truncated': False}
    return response


def _extract(bundle):
    structures = [Structure(b) for b in sorted(bundle['blocks'], key=lambda b: b['span']['start_byte'])]
    rows = [(s, n) for s in structures for n in s.nodes if n.tag == 'tr']
    require(not any(n.tag == 'table' and any(c.tag == 'table' for c in s.nodes if n.start < c.start < n.end)
                    for s in structures for n in s.nodes), 'Nested tables are unsupported')
    target = bundle['target']
    matches = [(s, n) for s, n in rows if s.span(n)['start_byte'] == target['start_byte']
               and s.span(n)['end_byte'] == target['end_byte']]
    require(len(matches) == 1, 'Target must mark exactly one complete HTML row')
    target_structure, target_row = matches[0]
    fields = {name: wrapped() for name in FIELD_TYPES}
    header = None
    context_pairs = []
    issuer_candidate = None
    for s, row in rows:
        if s.span(row)['start_byte'] >= target['start_byte']:
            break
        cs = columns(row)
        names = {ALIASES.get(key(c)) for c in cs if c}
        if {'raw_description', 'vote_components'} <= names:
            header = (s, cs)
            continue
        # Adjacent label/value rows, with empty spacer cells after the value.
        populated = [c for c in cs if c and c.value]
        if len(populated) == 2 and cs[0] is populated[0] and key(populated[0]) in ALIASES and key(populated[1]) not in ALIASES:
            context_pairs.append((s, populated[0], populated[1]))
        elif names & {'meeting_date', 'ticker', 'meeting_type'}:
            index = rows.index((s, row))
            if index + 1 < len(rows):
                ns, next_row = rows[index + 1]
                if ns is s and ns.span(next_row)['end_byte'] <= target['start_byte']:
                    values = columns(next_row)
                    if len(values) == len(cs):
                        context_pairs.extend((s, c, values[i]) for i, c in enumerate(cs)
                                             if c and key(c) in ALIASES and values[i])
        elif len(populated) == 1 and cs[0] is populated[0] and len(cs) > 1 and not context_pairs:
            issuer_candidate = (s, populated[0])
    require(header is not None, 'No supported column headings before target')
    hs, headings = header
    target_cells = columns(target_row)
    require(len(target_cells) == len(headings), 'Target columns differ from headings')
    require(all(c is not None for c in target_cells), 'Target column spans are unsupported')
    seen = set()
    for heading, cell in zip(headings, target_cells):
        if heading is None or not heading.value:
            require(not cell.value, 'Unlabeled populated target column')
            continue
        name = ALIASES.get(key(heading))
        require(name is not None and name not in seen, 'Unknown or duplicate target heading')
        seen.add(name)
        context_pairs.append((target_structure, heading, cell))
    for s, heading, cell in context_pairs:
        name = ALIASES[key(heading)]
        if not cell.value:
            continue
        # A target header can belong to a separate supplied block.
        heading_source = hs if heading in headings else s
        evidence = [heading_source.evidence(heading), s.evidence(cell)]
        raw = cell.value
        value = raw
        if name in ('vote_components', 'alignment'):
            continue
        try:
            if name == 'meeting_date':
                value = date_from_heading(raw, heading)
            elif name in ('management_recommendation', 'participation'):
                value = scalar(raw, name, guide_version=ACTIVE_GUIDE)
            elif name == 'security_identifiers':
                label = heading.value.rstrip(':')
                identifier = raw
                match = re.fullmatch(r'(CUSIP9|CUSIP)\s+(.+)', raw, re.I)
                if match:
                    label, identifier = match.groups()
                value = [{'source_label': wrapped(label, evidence, raw=label), 'value': wrapped(identifier, evidence, raw=identifier)}]
            fields[name] = wrapped(value, evidence, raw=raw)
        except ValueError:
            fields[name] = unresolved(raw, evidence, 'The declared value grammar does not resolve this cell.')
    if fields['issuer_name']['value'] is None and issuer_candidate:
        s, node = issuer_candidate
        fields['issuer_name'] = wrapped(node.value, [s.evidence(node)], raw=node.value)
    # Standalone fund headings precede the target. Later incomplete headings interrupt attribution.
    scopes = []
    for s, row in rows:
        if s.span(row)['end_byte'] > target['start_byte']:
            continue
        cs = cells(row)
        if len(cs) >= 2 and key(cs[0]) in ('fund name', 'fund', 'fund group'):
            scopes.append((s, row, cs[1].value, key(cs[0]) == 'fund group'))
    for s in structures:
        for n in s.nodes:
            if n.tag not in ('p', 'div', 'h1', 'h2', 'h3') or not n.end or s.span(n)['end_byte'] > target['start_byte']:
                continue
            if any(r.start <= n.start < r.end for r in s.nodes if r.tag == 'tr'):
                continue
            match = re.fullmatch(r'(Fund Group|Fund|Registrant Name|Registrant)\s*:\s*(.*)', n.value, re.I)
            if match:
                scopes.append((s, n, match[2], match[1].casefold() == 'fund group'))
            elif re.search(r'\bFund\b', n.value) and len(n.value) < 180:
                scopes.append((s, n, n.value, False))
    scope_spans = []
    if scopes:
        s, n, name, group = max(scopes, key=lambda entry: entry[0].span(entry[1])['start_byte'])
        ev = [s.evidence(n)]
        if not name or name.endswith(('...', '…')) or re.search(r'_{2,}', name):
            fields['reporting_scope'] = unresolved(n.value, ev, 'The latest supplied fund heading is incomplete.')
        else:
            fields['reporting_scope'] = wrapped({'name': wrapped(name, ev, raw=name),
                'scope_type': wrapped('FUND_GROUP' if group else 'INDIVIDUAL_FUND', ev, raw=n.value),
                'members': wrapped()}, ev, raw=n.value)
            scope_spans = [s.span(n)]
    votes = [(heading, cell) for heading, cell in zip(headings, target_cells)
             if heading and ALIASES.get(key(heading)) == 'vote_components']
    require(len(votes) == 1 and bool(votes[0][1].value), 'No cast-vote value')
    heading, cell = votes[0]
    ev = [hs.evidence(heading), target_structure.evidence(cell)]
    components = []
    for part in cell.value.split(';'):
        match = re.fullmatch(r'(For|Against|Abstain|Withhold)(?:\s+([0-9,.]+)\s+(shares|votes))?', part.strip(), re.I)
        if not match:
            fields['vote_components'] = unresolved(cell.value, ev, 'The vote direction or component syntax is unresolved.')
            break
        direction, amount, unit = match.groups()
        q = wrapped()
        if amount is not None:
            amount = quantity(amount)
            if amount == '0':
                continue
            q = wrapped({'amount': wrapped(amount, ev, raw=part.strip()), 'unit': wrapped(unit, ev, raw=unit)}, ev, raw=part.strip())
        components.append({'direction': wrapped(direction.upper(), ev, raw=part.strip()), 'quantity': q,
                           'disclosed_management_alignment': wrapped()})
    else:
        if components:
            for heading, cell in zip(headings, target_cells):
                if heading and ALIASES.get(key(heading)) == 'alignment' and cell.value:
                    ae = [hs.evidence(heading), target_structure.evidence(cell)]
                    try:
                        alignment = wrapped(scalar(cell.value, 'disclosed_management_alignment'), ae, raw=cell.value)
                    except ValueError:
                        alignment = unresolved(cell.value, ae, 'The alignment value is unresolved.')
                    if len(components) > 1:
                        alignment = unresolved(cell.value, ae, 'The row alignment does not associate each split vote component.')
                    for component in components:
                        component['disclosed_management_alignment'] = alignment
            fields['vote_components'] = wrapped(components, ev, raw=votes[0][1].value)
            if fields['participation']['availability'] == 'ABSENT_IN_CONTEXT':
                fields['participation'] = wrapped('VOTED', ev, raw=votes[0][1].value, origin='DERIVED', rule_id=D2_RULE)
        else:
            fields['vote_components'] = unresolved(votes[0][1].value, ev, 'Only zero quantities appear. No cast vote is established.')
    description = fields['raw_description']
    proponent = fields['proposal_source']
    if (isinstance(description['value'], str) and re.search(r'\bshareholder proposal\b', description['value'], re.I)
            and proponent['value'] in ('Board of Directors', 'Management', 'Mgmt')):
        fields['proposal_source'] = unresolved(proponent['raw_text'], proponent['evidence'] + description['evidence'],
            'The description names a shareholder proposal but the proponent cell names management.', 'CONFLICTING')
    record = {'record_id': bundle['input_id'] + '-target', 'source_anchor': {
        'subject_spans': [target_structure.span(target_row)], 'scope_spans': scope_spans}, 'fields': fields, 'enrichments': []}
    normalize_record(record, guide_version=ACTIVE_GUIDE)
    return record


def main():
    raw = sys.stdin.buffer.read()
    try:
        response = extract(raw)
    except (ValueError, KeyError, TypeError, UnicodeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    sys.stdout.buffer.write(json.dumps(response, ensure_ascii=False, allow_nan=False).encode())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
