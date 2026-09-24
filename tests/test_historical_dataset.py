"""Historical evidence and exact acceptance survive preparation and release."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proxybench.annotation.bindings import review_binding
from proxybench.annotation.historical import prepare_historical, literal_support, check_packet, source_block
from proxybench.annotation.review import write_review
from proxybench.training.dataset import publish, read_release, check_assignments, draft_gate, length_report
from proxybench.training.labels import TYPES, to_review, read_json, sha


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        # Distinct prompt and answer tokens expose truncation and missing endings.
        prefix = [1] * (len(messages[0]['content']) // 10 + 1)
        return prefix if len(messages) == 1 else prefix + [2] * 20 + [99]


class HistoricalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = '<p>Fund é</p><p>Other heading</p><tr><td>01</td><td>A &amp; <b>B</b></td><td></td><td>For</td></tr>'.encode()
        (self.root / 'source.htm').write_bytes(self.raw)
        a, b = self.raw.index(b'<tr>'), len(self.raw)
        self.selection = dict(packet_id='one', accession='filing', source_path='source.htm',
                              source_sha256=sha(self.raw), source_url='https://example.invalid',
                              split='legacy_development', group_id='legacy', encoding='utf-8',
                              spans=[[0, self.raw.index(b'</p>') + 4, 'block'], [a, b, 'row']],
                              target=[a, b], boundary_review='One subject has one disclosed vote.',
                              reviewer='assistant')
        self.policy = 'Extract only the marked subject.'
        self.packet = prepare_historical(self.root, self.selection, self.policy)
        self.assignments = {'filing': dict(split='legacy_development', group_id='legacy', development_exposed=True)}

    def test_cells_entities_inline_markup_gaps_and_multibyte_offsets(self):
        m = self.packet['manifest']
        cells = m['blocks'][1]['cells']
        self.assertEqual([c['text'] for c in cells], ['01', 'A & B', '', 'For'])
        self.assertEqual(self.raw[cells[1]['start_byte']:cells[1]['end_byte']], b'<td>A &amp; <b>B</b></td>')
        self.assertIn('OMITTED SOURCE BYTES', self.packet['model_input'])
        self.assertIn('A &amp; B', self.packet['source_view'])
        self.assertNotIn('Other heading', self.packet['model_input'])
        check_packet(self.root, self.packet, self.policy)
        literal_support({'raw_text': 'A & B'}, self.packet)
        with self.assertRaises(ValueError):
            literal_support({'raw_text': 'A & B For'}, self.packet)
        with self.assertRaises(ValueError):
            literal_support({'raw_text': 'A &amp; B'}, self.packet)

    def test_source_policy_and_input_mutation_invalidate_binding(self):
        for key in ('model_input', 'source_view'):
            changed = deepcopy(self.packet)
            changed[key] += 'added'
            with self.assertRaises(ValueError):
                check_packet(self.root, changed, self.policy)
        with self.assertRaises(ValueError):
            check_packet(self.root, self.packet, self.policy + ' changed')
        (self.root / 'source.htm').write_bytes(self.raw + b'changed')
        with self.assertRaises(ValueError):
            check_packet(self.root, self.packet, self.policy)

    def test_boundary_and_cell_join_rejection(self):
        bad = deepcopy(self.selection)
        bad['target'][0] += 4
        with self.assertRaisesRegex(ValueError, 'complete adjacent'):
            prepare_historical(self.root, bad, self.policy)
        bad = deepcopy(self.selection)
        bad['spans'][0][1] = self.raw.index(b'<tr>')
        with self.assertRaisesRegex(ValueError, 'Separate paragraphs'):
            prepare_historical(self.root, bad, self.policy)
        bad = deepcopy(self.selection)
        bad['boundary_review'] = ''
        with self.assertRaises(ValueError):
            prepare_historical(self.root, bad, self.policy)

    def test_orphan_heading_cell_is_context_only(self):
        raw = b'<td colspan="5">A &amp; B<br>Meeting Date: MAY 09, 2014</td></tr>'
        end = raw.index(b'</td>') + 5
        block = source_block(raw, 0, end, 'cell', 'utf-8')
        self.assertEqual(block['cells'][0]['text'], 'A & B Meeting Date: MAY 09, 2014')
        self.assertEqual(block['cells'][0]['colspan'], '5')
        (self.root / 'source.htm').write_bytes(raw)
        selection = dict(self.selection, source_sha256=sha(raw), spans=[[0, end, 'cell']], target=[0, end])
        with self.assertRaisesRegex(ValueError, 'context'):
            prepare_historical(self.root, selection, self.policy)

    def test_browser_supports_declared_partitions_and_excludes_test(self):
        (self.root / 'packet-set.json').write_text(json.dumps([self.packet]))
        page = write_review(self.root, packet_directory='.', training_contract=self.policy).read_text()
        self.assertIn("' · Partition: '+p.manifest.split", page)
        self.assertIn('legacy_development', page)
        test_packet = deepcopy(self.packet)
        test_packet['manifest']['split'] = 'test'
        (self.root / 'packet-set.json').write_text(json.dumps([test_packet]))
        with self.assertRaises(ValueError):
            write_review(self.root, packet_directory='.', training_contract=self.policy)

    def test_partition_leakage_exposure_duplicates_and_overlap(self):
        check_assignments([self.packet], self.assignments)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            check_assignments([self.packet, self.packet], self.assignments)
        second = deepcopy(self.packet)
        second['manifest']['target'][0] += 1
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            check_assignments([self.packet, second], self.assignments)
        bad = deepcopy(self.assignments)
        bad['other'] = dict(split='development', group_id='legacy', development_exposed=False)
        with self.assertRaises(ValueError):
            check_assignments([self.packet], bad)

    def test_filing_selection_cap(self):
        packets = []
        for index in range(12):
            packet = deepcopy(self.packet)
            packet['manifest']['target'] = [index * 2, index * 2 + 1]
            packets.append(packet)
        check_assignments(packets, self.assignments)
        extra = deepcopy(self.packet)
        extra['manifest']['target'] = [24, 25]
        with self.assertRaisesRegex(ValueError, '12-target selection cap'):
            check_assignments(packets + [extra], self.assignments)

    def export_inputs(self):
        label = {'fields': {k: dict(value=None, availability='ABSENT_IN_CONTEXT', origin=None, raw_text=None) for k in TYPES}}
        label['fields']['issuer_name'] = dict(value='A & B', availability='PRESENT', origin='EXTRACTED', raw_text='A & B')
        review = dict(schema='training-review-v1', packets={'one': dict(reviewed=True, input_binding=review_binding(self.packet), fields=to_review(label))})
        r, a = self.root / 'review.json', self.root / 'approval.json'
        r.write_text(json.dumps(review))
        approval = dict(decision='ACCEPTED', export_sha256=sha(r.read_bytes()), accepted_packet_ids=['one'],
                        reviewer='assistant', delegation='User delegated review.', accepted_at='2026-09-17')
        a.write_text(json.dumps(approval))
        evidence = {}
        for name, value in [('draft', label), ('review_history', {'fields_reviewed': list(TYPES)}), ('quality_gate', {'passed': True})]:
            p = self.root / (name + '.json')
            p.write_text(json.dumps(value))
            evidence[name] = dict(path=p.name, sha256=sha(p.read_bytes()))
        return r, a, {'one': evidence}

    def test_atomic_release_roundtrip_exact_review_and_evidence(self):
        r, a, evidence = self.export_inputs()
        out = self.root / 'release'
        kwargs = dict(assignments=self.assignments, evidence=evidence, tokenizer=Tokenizer())
        # A byte-level change invalidates acceptance even if JSON has the same meaning.
        original = r.read_bytes()
        r.write_bytes(original + b' ')
        with self.assertRaises(ValueError):
            publish(self.root, [self.packet], self.policy, r, a, out, **kwargs)
        self.assertFalse(out.exists())
        r.write_bytes(original)
        publish(self.root, [self.packet], self.policy, r, a, out, **kwargs)
        rows, manifest = read_release(out)
        self.assertEqual(len(rows['legacy_development']), 1)
        self.assertEqual(rows['legacy_development'][0]['messages'][0]['content'], self.packet['model_input'])
        answer = read_json(rows['legacy_development'][0]['messages'][1]['content'])
        self.assertEqual(answer['fields']['meeting_date']['availability'], 'ABSENT_IN_CONTEXT')
        self.assertEqual(manifest['examples'][0]['lengths']['response_tokens'], 21)
        with self.assertRaises(FileExistsError):
            publish(self.root, [self.packet], self.policy, r, a, out, **kwargs)
        (out / 'evidence' / evidence['one']['draft']['sha256']).write_text('{}')
        with self.assertRaises(ValueError):
            read_release(out)

    def test_failed_gate_and_interrupted_publication_leave_no_release(self):
        r, a, evidence = self.export_inputs()
        out = self.root / 'release'
        gate = self.root / 'quality_gate.json'
        gate.write_text('{"passed":false}')
        evidence['one']['quality_gate']['sha256'] = sha(gate.read_bytes())
        kwargs = dict(assignments=self.assignments, evidence=evidence, tokenizer=Tokenizer())
        with self.assertRaises(ValueError):
            publish(self.root, [self.packet], self.policy, r, a, out, **kwargs)
        r, a, evidence = self.export_inputs()
        kwargs['evidence'] = evidence
        with patch('proxybench.training.dataset.os.rename', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                publish(self.root, [self.packet], self.policy, r, a, out, **kwargs)
        self.assertFalse(out.exists())
        self.assertTrue(list(self.root.glob('release.incomplete-*')))
        self.assertTrue((self.root / 'release.publishing').exists())

    def test_strict_json_and_error_denominators(self):
        with self.assertRaises(ValueError):
            read_json('{"fields":{},"fields":{}}')
        good = dict(usable=True, accepted=True)
        outcomes = [dict(good) for _ in range(8)]
        self.assertTrue(draft_gate(outcomes, calibration=True)['passed'])
        outcomes[0]['usable'] = False
        self.assertFalse(draft_gate(outcomes, calibration=True)['passed'])
        outcomes = [dict(good) for _ in range(12)]
        outcomes[0]['source_value_error'] = True
        self.assertTrue(draft_gate(outcomes)['passed'])
        outcomes[1]['source_value_error'] = True
        self.assertFalse(draft_gate(outcomes)['passed'])
        self.assertFalse(draft_gate([dict(good, source_value_error=True)])['passed'])
        with self.assertRaises(ValueError):
            length_report(Tokenizer(), 'x' * 40000, '{}')


if __name__ == '__main__':
    unittest.main()
