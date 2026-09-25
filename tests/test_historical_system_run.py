"""CPU checks for the versioned historical system-message training input."""

import json
from pathlib import Path
import tempfile
import unittest

from proxybench.training.dataset import read_release
from proxybench.training.historical import admit_schedule, prepare_sequences, read_configuration
from proxybench.training.historical_run import main
from proxybench.training.system_release import SOURCE_PREFIX, source_only


class RoleTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        if [m['role'] for m in messages] == ['system', 'user']:
            return [1, 2, 3]
        if [m['role'] for m in messages] == ['system', 'user', 'assistant']:
            return [1, 2, 3, 4, 99, 5]
        raise ValueError('Unexpected role order')

    def decode(self, tokens, **kwargs):
        return 'answer' if tokens == [4] else 'wrong'


class HistoricalSystemRunTests(unittest.TestCase):
    def test_system_and_user_tokens_are_masked_but_answer_and_end_are_kept(self):
        messages = [dict(role='system', content='contract'), dict(role='user', content='source'),
                    dict(role='assistant', content='answer')]
        rows = {split: [dict(messages=messages)] for split in ('training', 'development')}
        config = dict(context_tokens=8, input_tokens=4, response_tokens=4)
        items = prepare_sequences(RoleTokenizer(), rows, config)
        self.assertEqual(items['training'][0]['input_ids'], [1, 2, 3, 4, 99])
        self.assertEqual(items['training'][0]['labels'], [-100, -100, -100, 4, 99])
        self.assertEqual(items['development'][0]['input_tokens'], 3)
        with self.assertRaises(ValueError):
            prepare_sequences(RoleTokenizer(), rows, dict(config, response_tokens=1))

    def test_source_only_message_preserves_marked_cells(self):
        source = '[OMITTED SOURCE BYTES 0:1]\nBEGIN MARKED TARGET\nVote\nEND MARKED TARGET'
        packet = dict(model_input='policy'+SOURCE_PREFIX+source)
        self.assertEqual(source_only(packet, 'policy'), source)
        with self.assertRaises(ValueError):
            source_only(packet, 'changed policy')

    def test_frozen_profile_and_resource_reservations(self):
        config = read_configuration('configs/qwen35-4b-historical-b35-system-v1.json')
        self.assertEqual((config['training_examples'], config['development_examples'], config['updates']),
                         (330, 90, 660))
        state = dict(completed_phases={}, reservations={'training': 6000},
                     resource_limits=dict(gpu_total=14400, cpu_total=3600, gpu_phase=1800,
                                          training_phase=6000, cpu_phase=1800,
                                          stop_reserve=1200, test_reserve=3300))
        admitted = admit_schedule(state, dict(gpu=[], cpu=[]), phase='training', operation=['training'])
        self.assertEqual(admitted['stop_reserve_seconds'], 4500)
        with self.assertRaises(ValueError):
            admit_schedule(state, dict(gpu=[dict(phase='training', elapsed_seconds=1)], cpu=[]),
                           phase='training', operation=['training'])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.json'
            for key, value in (('updates', 659), ('checkpoint_interval', 48)):
                path.write_text(json.dumps(dict(config, **{key: value})))
                with self.assertRaises(ValueError):
                    read_configuration(path)
        with self.assertRaisesRegex(ValueError, 'separately tested scorer'):
            main(['evaluate', '--configuration', 'configs/qwen35-4b-historical-b35-system-v1.json',
                  '--run', 'artifacts/runs/qwen35-4b-historical-b35-system-v1-01'])

    @unittest.skipUnless(Path('data/annotations/sol-historical-20260925-b35-system-v1/manifest.json').exists(),
                         'Local accepted release is not installed')
    def test_local_release_preserves_parent_answers_and_source(self):
        rows, manifest = read_release('data/annotations/sol-historical-20260925-b35-system-v1')
        self.assertEqual((len(rows['training']), len(rows['development'])), (330, 90))
        old_rows, _ = read_release(manifest['parent_release'])
        for split in ('training', 'development'):
            for new, old in zip(rows[split], old_rows[split], strict=True):
                self.assertEqual([m['role'] for m in new['messages']], ['system', 'user', 'assistant'])
                self.assertEqual(new['messages'][2]['content'], old['messages'][1]['content'])
                self.assertNotIn('Write review notes separately', new['messages'][1]['content'])
                self.assertIn('BEGIN MARKED TARGET', new['messages'][1]['content'])
                json.loads(new['messages'][2]['content'])
        with tempfile.TemporaryDirectory() as tmp:
            copied = Path(tmp)
            for name in ('manifest.json', 'policy.md', 'system-prompt.txt', 'training.jsonl', 'development.jsonl'):
                (copied/name).write_bytes((Path('data/annotations/sol-historical-20260925-b35-system-v1')/name).read_bytes())
            (copied/'training.jsonl').write_bytes((copied/'training.jsonl').read_bytes()+b'{}\n')
            with self.assertRaises(ValueError):
                read_release(copied)
