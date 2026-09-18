"""Quotation review never repairs predictions or replays GPU work."""

from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from proxybench.evaluation.training_labels import export_comparison, score
from proxybench.execution.resources import durable_json
from proxybench.training.export_amendment import amend_export, amended_code
from proxybench.training.historical import PHASES, binding, inventory, validate_identity
from proxybench.training.historical_run import gate_exports, main
from proxybench.training.smoke import digest
from test_training_labels import field, label


def malformed(value=None):
    value = label()['fields'] if value is None else value
    return dict(status='COMPLETE', prompt_token_ids=[1, 2], token_ids=[3, 99],
                text='```json\n'+json.dumps(value)+'\n```')


class MalformedQuotationTests(unittest.TestCase):
    def test_fenced_or_plain_invalid_structure_requires_review_and_stays_wrong(self):
        a = label()['fields']
        b = deepcopy(a)
        b['issuer_name']['raw_text'] = 'Issuer: A & B'
        for fenced in (True, False):
            left, right = malformed(a), malformed(b)
            if not fenced:
                left['text'], right['text'] = json.dumps(a), json.dumps(b)
            result = export_comparison(left, right)
            self.assertEqual(result['status'], 'REVIEW_REQUIRED')
            self.assertEqual(result['changed_paths'], ['issuer_name.raw_text'])
            self.assertFalse(result['format_valid'])
            for answer in (left, right):
                scored = score(label(), answer, 'Issuer: A & B')
                self.assertFalse(scored['format_valid'])
                self.assertFalse(scored['source_value_correct'])

    def test_changed_facts_origins_absence_structure_and_components_fail(self):
        original = label()['fields']
        variants = []
        for key, value in [('value', 'Different issuer'), ('origin', 'DERIVED'), ('availability', 'AMBIGUOUS')]:
            row = deepcopy(original)
            row['issuer_name'][key] = value
            variants.append(row)
        row = deepcopy(original)
        row['vote_components']['value'] *= 2
        variants.append(row)
        row = deepcopy(original)
        row['issuer_name']['extra'] = 'extra'
        variants.append(row)
        variants.extend([{'fields': original}, {**original, 'extra_record': original}])
        for row in variants:
            if 'issuer_name' in row:
                row['issuer_name']['raw_text'] = 'Issuer: A & B'
            self.assertEqual(export_comparison(malformed(original), malformed(row))['status'], 'FAILED')

    def test_unsafe_quotation_nulls_other_and_unparseable_bodies_fail(self):
        for before, after in [(None, 'A & B'), ('A & B', None), ('A & B', '')]:
            a, b = label()['fields'], label()['fields']
            a['issuer_name']['raw_text'], b['issuer_name']['raw_text'] = before, after
            self.assertEqual(export_comparison(malformed(a), malformed(b))['status'], 'FAILED')
        a, b = label()['fields'], label()['fields']
        a['participation'] = field('OTHER')
        b['participation'] = field('OTHER')
        a['participation']['raw_text'], b['participation']['raw_text'] = 'Declined', 'Unavailable'
        self.assertEqual(export_comparison(malformed(a), malformed(b))['status'], 'FAILED')
        for text in ('not JSON', '[{}, {}]', '{"fields":{},"fields":{}}', '{"fields":NaN}',
                     malformed()['text']+' extra commentary', malformed()['text'][:-8]):
            self.assertEqual(export_comparison(malformed(), dict(malformed(), text=text))['status'], 'FAILED')

    def test_validity_prompt_and_termination_must_not_change(self):
        left = malformed(label())  # JSON is valid only after removing fences.
        valid = dict(left, text=json.dumps(label()))
        self.assertEqual(export_comparison(left, valid)['status'], 'FAILED')
        for changes in (dict(prompt_token_ids=[8]), dict(status='TIMEOUT'), dict(status='LENGTH_STOP')):
            self.assertEqual(export_comparison(left, dict(left, **changes))['status'], 'FAILED')


class AmendmentRunTests(unittest.TestCase):
    def fixture(self, root):
        prepared = dict(panel=[0, 1, 2], order=list(range(192)), examples=[dict(split='training',
            packet=dict(manifest=dict(blocks=[dict(target=True, cells=[dict(text='Issuer: A & B')])])))]*3)
        durable_json(root/'prepared.json', prepared)
        completed = {}
        for name, _, _ in PHASES[:4]:
            output = root/name
            output.mkdir()
            durable_json(output/'complete.json', dict(status='COMPLETE'))
            if name in ('original-export', 'original-panel'):
                for index in prepared['panel']:
                    value = label()['fields']
                    if name == 'original-panel' and index == 1:
                        value['issuer_name']['raw_text'] = 'Issuer: A & B'
                    durable_json(output/f'answer-{index}.json', malformed(value))
            completed[name] = dict(output=str(output), files=inventory(output.iterdir()))
        identity = dict(configuration={}, code={}, inputs={}, model={}, tokenizer={}, runtime={},
                        git_commit='old', prepared_sha256=digest(root/'prepared.json'),
                        executable=str(Path(sys.executable).absolute()), python=sys.version)
        state = dict(status='EXPORT_REJECTED', active_phase='original-panel', identity=binding(identity),
                     attempts={n: 1 for n in completed}, completed_phases=completed,
                     reservations={n: s for n, _, s in PHASES}, error='Old gate')
        durable_json(root/'identity.json', identity)
        durable_json(root/'state.json', state)
        durable_json(root/'original-export-gate.json', {'1': {'status': 'FAILED'}})
        for resource in ('cpu', 'gpu'):
            (root/f'{resource}-ledger.jsonl').write_text(json.dumps(dict(status='EXITED',
                elapsed_seconds=100, run=str(root/'old-worker'), phase='probe'))+'\n')
        return identity, state, prepared

    def test_amend_review_and_continue_reuse_completed_work_and_time(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            _, previous, _ = self.fixture(root)
            ledgers = {p.name: p.read_bytes() for p in root.glob('*-ledger.jsonl')}
            with patch('proxybench.training.historical_run.read_configuration', return_value={}), \
                 patch('proxybench.training.export_amendment.runtime_files', return_value={}), \
                 patch('proxybench.training.export_amendment.amended_code', return_value=({}, 'new')), \
                 patch('proxybench.training.historical_run.supervise') as gpu:
                self.assertEqual(main(['amend-export', '--run', str(root), '--reason', 'Authorized quotation fix']), 3)
                gpu.assert_not_called()
            state = json.loads((root/'state.json').read_text())
            self.assertEqual(state['status'], 'REVIEW_REQUIRED')
            self.assertEqual(state['completed_phases'], previous['completed_phases'])
            self.assertEqual(state['attempts'], previous['attempts'])
            self.assertEqual(state['reservations'], previous['reservations'])
            self.assertNotEqual(state['identity'], previous['identity'])
            self.assertEqual(json.loads((root/'export-amendment/state.json').read_text()), previous)
            self.assertEqual({p.name: p.read_bytes() for p in root.glob('*-ledger.jsonl')}, ledgers)
            decisions = root/'review-original/decision-template.json'
            value = json.loads(decisions.read_text())
            value.update(reviewer='test', rationale='Same source quotation and unchanged malformed answer')
            value['cases']['1'] = dict(decision='PASS', reason='The same issuer occurs in both quotations')
            decisions.write_text(json.dumps(value))
            with patch('proxybench.training.historical.runtime_files', return_value={}), \
                 patch('proxybench.training.historical.inventory', return_value={}):
                validate_identity(root, {})
                # Exercise the real gate, review binding and continuation checks.
                with patch('proxybench.training.historical_run.read_configuration', return_value={}):
                    self.assertEqual(main(['review', '--run', str(root), '--decisions', str(decisions)]), 0)
                calls = []
                def stop_at_baseline(command, output, limits, **kwargs):
                    calls.append(kwargs['phase'])
                    return 'STOP_REQUESTED'
                with patch('proxybench.training.historical_run.read_configuration', return_value=dict(
                        minimum_disk_bytes=0, limits={}, cpu_limits={})), \
                     patch('proxybench.training.historical_run.validate_identity', side_effect=lambda r, c:
                           ({}, json.loads((r/'state.json').read_text()))), \
                     patch('proxybench.training.historical_run.update_estimates'), \
                     patch('proxybench.training.historical_run.environment', return_value={}), \
                     patch('proxybench.training.historical_run.subprocess.check_output', return_value='RTX 3090'), \
                     patch('proxybench.training.historical_run.supervise', side_effect=stop_at_baseline):
                    self.assertEqual(main(['continue', '--run', str(root)]), 130)
                self.assertEqual(calls, ['baseline'])
                (root/'export-amendment/state.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'File identity changed'):
                    validate_identity(root, {})

    def test_migration_refuses_tampering_training_active_workers_and_repeated_use(self):
        for problem in ('output', 'input', 'training', 'active', 'reason'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root)
                if problem == 'output':
                    (root/'original-panel/answer-1.json').write_text('{}')
                elif problem == 'input':
                    (root/'prepared.json').write_text('{}')
                elif problem == 'training':
                    (root/'training-journal.json').write_text('{}')
                elif problem == 'active':
                    (root/'gpu-ledger.active.json').write_text('{}')
                before = (root/'state.json').read_bytes()
                with patch('proxybench.training.export_amendment.runtime_files', return_value={}), \
                     patch('proxybench.training.export_amendment.amended_code', return_value=({}, 'new')):
                    with self.assertRaises(ValueError):
                        amend_export(root, {}, '' if problem == 'reason' else 'Authorized')
                self.assertEqual((root/'state.json').read_bytes(), before)
                self.assertFalse((root/'export-amendment').exists())

    def test_gate_saves_all_cases_before_reporting_a_failed_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, state, prepared = self.fixture(root)
            value = label()['fields']
            value['issuer_name']['value'] = 'Another issuer'
            durable_json(root/'original-panel/answer-1.json', malformed(value))
            with self.assertRaisesRegex(ValueError, '1: Malformed output comparison'):
                gate_exports(root, state, prepared, 'original-panel')
            result = json.loads((root/'original-export-gate.json').read_text())
            self.assertEqual(set(result), {'0', '1', '2'})
            self.assertEqual(result['2']['status'], 'PASS')
            self.assertFalse((root/'review-original').exists())

    def test_code_amendment_refuses_unrelated_and_uncommitted_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_cwd = Path.cwd()
            try:
                os.chdir(tmp)
                base = Path('src/proxybench')
                base.mkdir(parents=True)
                gate, worker, added = [base/n for n in ('gate.py', 'worker.py', 'amend.py')]
                gate.write_text('old gate')
                worker.write_text('unchanged worker')
                prior = inventory(base.iterdir())
                gate.write_text('new gate')
                added.write_text('migration')
                committed = {str(p): p.read_bytes() for p in base.iterdir()}
                def git(args, **kwargs):
                    return 'revision\n' if args[1] == 'rev-parse' else committed[args[2].split(':', 1)[1]]
                with patch('proxybench.training.export_amendment.PREVIOUS_CODE', {str(gate): prior[str(gate)]}), \
                     patch('proxybench.training.export_amendment.ADDED_CODE', {str(added)}), \
                     patch('proxybench.training.export_amendment.subprocess.check_output', side_effect=git):
                    self.assertEqual(amended_code(dict(code=prior))[1], 'revision')
                    worker.write_text('changed worker')
                    with self.assertRaisesRegex(ValueError, 'File identity changed'):
                        amended_code(dict(code=prior))
                    worker.write_text('unchanged worker')
                    gate.write_text('uncommitted gate')
                    with self.assertRaisesRegex(ValueError, 'Commit and test'):
                        amended_code(dict(code=prior))
                    gate.write_text('new gate')
                    (base/'unrelated.py').write_text('extra code')
                    with self.assertRaisesRegex(ValueError, 'Unrelated source files'):
                        amended_code(dict(code=prior))
            finally:
                os.chdir(old_cwd)
