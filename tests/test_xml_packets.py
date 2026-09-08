"""Exact original-byte projection and source-context failures."""
from copy import deepcopy
import json
import unittest

from proxybench.annotation.xml_packets import build, project, digest
from proxybench.sources.npx_xml import XMLRejected
from proxybench.training.compact import compact_source, decode_evidence
from test_local_inputs import PRIMARY, ROW, votes


class PacketTests(unittest.TestCase):
    def packet(self, raw=None):
        return build(PRIMARY, raw or votes(ROW + ROW), primary_id='p', votes_id='v', index=1, input_id='test')

    def test_repeated_entities_unicode_and_attachment_context(self):
        raw = votes(ROW + ROW)
        packet = self.packet(raw)
        bundle = packet['bundle']
        self.assertEqual(bundle['blocks'][0]['original_text'].encode(), PRIMARY)
        self.assertIn('xmlns=', bundle['blocks'][1]['original_text'])
        self.assertIn('&amp;', bundle['blocks'][2]['prepared_text'])
        projected = project(bundle['blocks'][2]['span'], packet['source_bytes'], packet['lineage'], {'p': PRIMARY, 'v': raw})
        self.assertEqual(projected['start_byte'], raw.rindex(b'<proxyTable>'))
        self.assertEqual(raw[projected['start_byte']:projected['end_byte']], ROW.encode())
        compact = compact_source(json.dumps(bundle))
        self.assertEqual(compact['target'], [2, 0, len(ROW.encode())])
        self.assertEqual(len(compact['blocks']), 4)
        evidence = {'span': [2, 0, len(ROW.encode())], 'basis': 'ORIGINAL_DECODED',
                    'quote': 'Élan &amp; Co', 'view': False, 'block': False}
        decode_evidence(evidence, bundle)
        evidence['quote'] = 'Élan & Co'
        with self.assertRaises(ValueError):
            decode_evidence(evidence, bundle)

    def test_separators_changed_source_bounds_and_packet(self):
        p = self.packet()
        span = p['bundle']['blocks'][2]['span']
        originals = {'p': PRIMARY, 'v': votes(ROW + ROW)}
        for changed in (dict(span, start_byte=span['start_byte'] - 1), dict(span, end_byte=-1),
                        dict(span, start_byte=True), dict(span, document_id='wrong')):
            with self.assertRaises(ValueError):
                project(changed, p['source_bytes'], p['lineage'], originals)
        with self.assertRaises(ValueError):
            project(span, p['source_bytes'] + b'x', p['lineage'], originals)
        with self.assertRaises(ValueError):
            project(span, p['source_bytes'], p['lineage'], originals | {'v': originals['v'].replace(b'ABSTAIN', b'AGAINST')})
        lineage = deepcopy(p['lineage'])
        lineage['locations'][2]['attachment']['start_byte'] = 0
        with self.assertRaises(ValueError):
            project(span, p['source_bytes'], lineage, originals)

    def test_unknown_context_rejected_but_supplement_retained(self):
        with self.assertRaises(XMLRejected):
            self.packet(votes(ROW + ROW + '<unmapped>governing note</unmapped>'))
        with self.assertRaises(XMLRejected):
            self.packet(votes(ROW + ROW).replace(b'informationtable', b'unknown'))
        raw = votes(ROW + ROW.replace('<voteSeries>', '<voteOtherInfo>Do not discard this</voteOtherInfo><voteSeries>'))
        self.assertIn('Do not discard this', self.packet(raw)['bundle']['blocks'][2]['original_text'])
        raw = votes(ROW + ROW.replace('<issuerName>', '<issuerName><nested>value</nested>'))
        self.assertIn('<nested>', self.packet(raw)['bundle']['blocks'][2]['original_text'])

    def test_prefixed_root_namespace_is_copied(self):
        raw = votes(ROW + ROW).replace(b'<proxyVoteTable xmlns=', b'<v:proxyVoteTable xmlns:v=').replace(b'</proxyVoteTable>', b'</v:proxyVoteTable>')
        # Rows use a separately declared default namespace inherited from this root.
        raw = raw.replace(b'xmlns:v=', b'xmlns="http://www.sec.gov/edgar/document/npxproxy/informationtable" xmlns:v=')
        p = self.packet(raw)
        self.assertIn('xmlns:v=', p['bundle']['blocks'][1]['original_text'])


if __name__ == '__main__':
    unittest.main()
