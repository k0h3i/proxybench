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
            prior = {name: {'output': str(root/name), 'files': {}} for name in ('probe', 'original-export', 'original-conversion', 'original-panel')}
            state = dict(status='CLEAN_STOP', operation='separate', identity='one', completed_phases=prior, attempts={},
                         reservations={n: s for n, _, s in PHASES})
            durable_json(root/'state.json', state)
            durable_json(root/'prepared.json', dict(order=list(range(192))))
            durable_json(root/'training-journal.json', dict(status='CLEAN_STOP', checkpoint=str(root/'checkpoint')))
            config = dict(minimum_disk_bytes=0, limits={}, cpu_limits={},
                          training_examples=96, development_examples=24, updates=192)
            calls = []
            def supervise(command, output, limits, **kwargs):
                calls.append(kwargs['phase'])
                return 'EXITED'  # A new clean stop, without a phase completion file.
            with patch('proxybench.training.historical_run.read_configuration', return_value=config), \
                 patch('proxybench.training.historical_run.validate_identity', return_value=({}, state)), \
                 patch('proxybench.training.historical_run.require_clean_stop', return_value=7) as boundary, \
                 patch('proxybench.training.historical_run.environment', return_value={}), \
                 patch('proxybench.training.historical_run.supervise', supervise), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['resume', '--run', str(root)]), 130)
                self.assertEqual(calls, ['training'])
                self.assertEqual(boundary.call_count, 2)
            self.assertEqual(json.loads((root/'state.json').read_text())['completed_phases'], prior)

    def test_training_saves_and_exits_then_evaluation_requires_another_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = dict(status='TRAINING_READY', operation='separate', identity='identity',
                         completed_phases={}, attempts={}, reservations={n: s for n, _, s in PHASES})
            durable_json(root/'prepared.json', dict(order=list(range(192))))
            durable_json(root/'state.json', state)
            config = dict(minimum_disk_bytes=0, limits={}, cpu_limits={},
                          training_examples=96, development_examples=24, updates=192)
            calls = []
            def fake_supervise(command, output, limits, **kwargs):
                phase = kwargs['phase']
                calls.append(phase)
                durable_json(output.parent/'complete.json', dict(status='COMPLETE', updates=192))
                return 'EXITED'
            def validate(root, config):
                return {}, json.loads((root/'state.json').read_text())
            with patch('proxybench.training.historical_run.validate_identity', validate), \
                 patch('proxybench.training.historical_run.environment', return_value={}), \
                 patch('proxybench.training.historical_run.supervise', fake_supervise), \
                 patch('proxybench.training.historical_run.read_configuration', return_value=config), \
                 patch('proxybench.training.historical_run.create_final_review') as final, \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    main(['evaluate', '--run', str(root)])
                self.assertEqual(main(['train', '--run', str(root)]), 0)
                self.assertEqual(calls, ['training'])
                self.assertEqual(json.loads((root/'state.json').read_text())['status'], 'TRAINED')
                final.assert_not_called()
                with self.assertRaises(ValueError):
                    main(['train', '--run', str(root)])
                self.assertEqual(main(['evaluate', '--run', str(root)]), 3)
                self.assertEqual(calls, [n for n, _, _ in PHASES])
                self.assertEqual(calls.count('training'), 1)
                final.assert_called_once()

    def test_engine_evaluates_all_configured_malformed_answers(self):
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
            development_examples = 3
            durable_json(root/'prepared.json', dict(
                items={'development': [dict(input_ids=[1], input_tokens=1)]*development_examples},
                rows={'development': [dict(messages=[dict(role='user', content='source')])]*development_examples}))
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
                       engine_configuration=str(root/'engine.json'), development_examples=development_examples))
            self.assertEqual(calls, list(range(development_examples)))
            self.assertEqual(len(list(output.glob('answer-*.json'))), development_examples)
            self.assertTrue((output/'complete.json').exists())
