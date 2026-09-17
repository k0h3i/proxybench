"""CPU stand-ins exercise the user command schedule without loading a GPU model."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proxybench.execution.resources import durable_json
from proxybench.training.historical import PHASES
from proxybench.training.historical_run import launch, main


class PilotLaunchTests(unittest.TestCase):
    def test_resume_command_starts_at_training_and_keeps_prior_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prior = {name: {'output': str(root/name), 'files': {}} for name, _, _ in PHASES[:5]}
            state = dict(status='CLEAN_STOP', identity='one', completed_phases=prior, attempts={},
                         reservations={n: s for n, _, s in PHASES})
            durable_json(root/'state.json', state)
            durable_json(root/'prepared.json', dict(order=list(range(192))))
            durable_json(root/'training-journal.json', dict(status='CLEAN_STOP', checkpoint=str(root/'checkpoint')))
            config = dict(minimum_disk_bytes=0, limits={}, cpu_limits={})
            calls = []
            def supervise(command, output, limits, **kwargs):
                calls.append(kwargs['phase'])
                return 'EXITED'  # A new clean stop, without a phase completion file.
            with patch('proxybench.training.historical_run.read_configuration', return_value=config), \
                 patch('proxybench.training.historical_run.validate_identity', return_value=({}, state)), \
                 patch('proxybench.training.historical_run.require_clean_stop', return_value=7) as boundary, \
                 patch('proxybench.training.historical_run.update_estimates'), \
                 patch('proxybench.training.historical_run.environment', return_value={}), \
                 patch('proxybench.training.historical_run.subprocess.check_output', return_value='RTX 3090'), \
                 patch('proxybench.training.historical_run.supervise', supervise), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['resume', '--run', str(root)]), 130)
                self.assertEqual(calls, ['training'])
                self.assertEqual(boundary.call_count, 2)
            self.assertEqual(json.loads((root/'state.json').read_text())['completed_phases'], prior)

    def test_user_run_and_continue_reuse_every_completed_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = dict(status='PREPARED', identity='identity', completed_phases={}, attempts={},
                         reservations={n: s for n, _, s in PHASES})
            prepared = dict(order=list(range(192)), panel=[0, 1, 2], examples=[], rows={})
            durable_json(root/'prepared.json', prepared)
            durable_json(root/'state.json', state)
            calls = []
            def fake_supervise(command, output, limits, **kwargs):
                phase = command[command.index('--phase')+1]
                calls.append(phase)
                durable_json(output.parent/'complete.json', dict(status='COMPLETE'))
                return 'EXITED'
            def gate(root, state, prepared, name):
                if name == 'original-panel':
                    review = root/'review-original'
                    review.mkdir()
                    (review/'accepted.json').write_text('{}')
                    from proxybench.training.smoke import digest
                    state.update(status='REVIEW_REQUIRED', review=str(review), next_phase=name,
                                 accepted_review_sha256=digest(review/'accepted.json'))
                    durable_json(root/'state.json', state)
                    return False
                return True
            def validate(root, config):
                return {}, json.loads((root/'state.json').read_text())
            config = dict(minimum_disk_bytes=0, limits={}, cpu_limits={})
            with patch('proxybench.training.historical_run.validate_identity', validate), \
                 patch('proxybench.training.historical_run.update_estimates'), \
                 patch('proxybench.training.historical_run.environment', return_value={}), \
                 patch('proxybench.training.historical_run.subprocess.check_output', return_value='RTX 3090'), \
                 patch('proxybench.training.historical_run.supervise', fake_supervise), \
                 patch('proxybench.training.historical_run.gate_exports', gate), \
                 patch('proxybench.training.historical_run.require_decisions'), \
                 patch('proxybench.training.historical_run.read_configuration', return_value=config), \
                 patch('proxybench.training.historical_run.create_final_review') as final, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['run', '--run', str(root)]), 3)
                self.assertEqual(calls, [p[0] for p in PHASES[:4]])
                self.assertEqual(main(['continue', '--run', str(root)]), 3)
                self.assertEqual(calls, [p[0] for p in PHASES])
                self.assertEqual(calls.count('training'), 1)
                final.assert_called_once()
                with self.assertRaises(ValueError):
                    launch(root, config, Path('config.json'), 'run')

    def test_engine_evaluates_all_24_malformed_answers(self):
        try:
            import transformers
        except ImportError:
            self.skipTest('Requires the separate tokenizer environment, with CUDA disabled')
        from proxybench.training.historical_engine import engine
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return 'prompt'
        class Process:
            pid = 987654
            returncode = 0
            def poll(self):
                return None
            def terminate(self):
                pass
            def wait(self, **kwargs):
                return 0
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root/'output'
            output.mkdir()
            conversion = root/'conversion'
            conversion.mkdir()
            (root/'model.gguf').write_bytes(b'fixture, not model weights')
            from proxybench.training.smoke import digest
            durable_json(conversion/'complete.json', dict(status='COMPLETE', path=str(root/'model.gguf'), sha256=digest(root/'model.gguf')))
            durable_json(root/'state.json', dict(completed_phases={'original-conversion': dict(output=str(conversion))}))
            durable_json(root/'prepared.json', dict(items={'development': [dict(input_ids=[1], input_tokens=1)]*24},
                         rows={'development': [dict(messages=[dict(role='user', content='source')])]*24}))
            durable_json(root/'pin.json', dict(files={}, library_path=''))
            durable_json(root/'engine.json', dict(server_arguments=[], startup_seconds=1))
            calls = []
            def generate(port, tokenizer, prompt, path, **kwargs):
                calls.append(kwargs['index'])
                value = dict(status='COMPLETE', text='malformed', format_valid=False, token_ids=[3, 99],
                             timing=dict(request_seconds=1))
                durable_json(path, value)
                return value
            def popen(*args, **kwargs):
                kwargs['stdout'].write(b'offloaded 33/33 layers to GPU\n')
                kwargs['stdout'].flush()
                return Process()
            # Only the tokenizer import occurs here. No model or GPU import runs.
            with patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()), \
                 patch('proxybench.training.historical_engine.subprocess.Popen', popen), \
                 patch('proxybench.training.historical_engine.socket.socket') as sock, \
                 patch('proxybench.training.historical_engine.json_request', return_value={'status': 'ok'}), \
                 patch('proxybench.training.historical_engine.require_prompt_tokens'), \
                 patch('proxybench.training.historical_engine.generate', generate), \
                 contextlib.redirect_stdout(io.StringIO()):
                sock.return_value.__enter__.return_value.getsockname.return_value = ('127.0.0.1', 12345)
                engine(root, 'baseline', output, dict(engine_root=str(root), runtime_pin=str(root/'pin.json'),
                       engine_configuration=str(root/'engine.json')))
            self.assertEqual(calls, list(range(24)))
            self.assertEqual(len(list(output.glob('answer-*.json'))), 24)
            self.assertTrue((output/'complete.json').exists())
