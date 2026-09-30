"""Run the public evaluation command with a synthetic subprocess model server."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

from proxybench.evaluation.workflow import create_run, runtime_binding
from proxybench.extraction.runtime import runtime_identity, source_messages
from proxybench.runstate import binding, file_hash
from proxybench.training.labels import FIELDS


TOKENIZER = '''
import sys
print('Synthetic tokenizer library banner', flush=True)
print('Synthetic tokenizer library warning', file=sys.stderr, flush=True)

class AutoTokenizer:
    eos_token_id = 9999

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        assert kwargs == dict(local_files_only=True, trust_remote_code=False)
        return cls()

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        text = messages[0]['content'] + '\\n' + messages[1]['content']
        return list(text.encode()) if tokenize else text

    def decode(self, tokens, **kwargs):
        return ''.join(chr(token) if token < 256 else '<eos>' for token in tokens)
'''


SERVER = '''
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import sys

assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
root = Path(__file__).parent

def event(kind, **values):
    with (root / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(kind=kind, **values)) + '\\n')

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def respond(self, payload):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        assert self.path == '/health'
        self.respond(b'{"status":"ok"}')

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.path == '/tokenize':
            event('tokenize', **request)
            self.respond(json.dumps(dict(tokens=list(request['content'].encode()))).encode())
            return
        assert self.path == '/completion'
        event('completion', **request)
        response = b'not-json' if b'case 1' in bytes(request['prompt']) else (root / 'response.json').read_bytes()
        tokens = list(response) + [9999]
        terminal = dict(tokens=tokens, stop=True, stop_type='eos', truncated=False,
                        tokens_evaluated=len(request['prompt']), tokens_predicted=len(tokens),
                        timings=dict(cache_n=0, prompt_n=len(request['prompt'])))
        self.respond(b'data: ' + json.dumps(terminal).encode() + b'\\n\\n')

event('start')
port = int(sys.argv[sys.argv.index('--port') + 1])
HTTPServer(('127.0.0.1', port), Handler).serve_forever()
'''


class EvaluationCommandTests(unittest.TestCase):
    def test_real_command_shares_process_and_recovers_without_reloading(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            server = root/'llama-server'
            server.write_text('#!' + sys.executable + '\n' + textwrap.dedent(SERVER))
            server.chmod(0o700)
            (root/'transformers.py').write_text(textwrap.dedent(TOKENIZER))
            (root/'model.gguf').write_bytes(b'synthetic model; no weights')
            (root/'prompt.txt').write_text('Return the fields found in the marked source.')
            (root/'tokenizer').mkdir()
            library = root/'libcudart.so.12'
            library.write_bytes(b'synthetic pinned library')
            (root/'manifest.json').write_text(json.dumps(dict(source_commit='synthetic',
                files={'llama-server': file_hash(server)})))
            reference = dict(fields={name: dict(value=None, raw_text=None,
                availability='ABSENT_IN_CONTEXT', origin=None) for name in FIELDS})
            reference['fields']['issuer_name'] = dict(value='A & B', raw_text='A & B',
                availability='PRESENT', origin='EXTRACTED')
            (root/'response.json').write_text(json.dumps(reference))
            config = dict(server=str(server), model=str(root/'model.gguf'),
                tokenizer=str(root/'tokenizer'), runtime_manifest=str(root/'manifest.json'),
                source_commit='synthetic', library_path='', server_arguments=[],
                external_libraries={str(library): file_hash(library)},
                system_prompt=str(root/'prompt.txt'), input_tokens=4096,
                response_tokens=4096, context_tokens=8192, startup_seconds=5,
                natural_request_seconds=5, limits=dict(cpu_only=True, phase_seconds=10,
                    total_seconds=30, start_host_bytes=0, stop_host_bytes=0,
                    device_margin_bytes=0, heartbeat_seconds=15))
            source_cell = dict(block_index=0, cell_index=0, text='Issuer: A & B\n<script>bad()</script>')
            cases = [dict(id=str(index), reference=reference, source_cells=[source_cell],
                messages=source_messages(source_cell['text'] +
                    f'\nBEGIN MARKED TARGET case {index} END MARKED TARGET', config))
                for index in range(3)]
            inputs = dict(identity=dict(dataset='synthetic', model=file_hash(config['model']),
                runtime=runtime_binding(config), prompt=binding([case['messages'][0] for case in cases])),
                cases=cases)
            run = root/'run'
            create_run(run, inputs, config)
            source = Path(__file__).resolve().parents[1]/'src'
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='',
                               PYTHONPATH=os.pathsep.join((str(root), str(source))))
            command = [sys.executable, '-m', 'proxybench']

            def execute(*extra, expected_returncode=2, action='evaluate'):
                result = subprocess.run([*command, action, '--run-dir', str(run), *extra], cwd=root, env=environment,
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, expected_returncode, result.stdout + result.stderr)
                return result

            def evaluate(*extra):
                execute(*extra)
                return json.loads((run/'evaluation/report.json').read_text())

            identity = runtime_identity(config)
            initial_state = json.loads((run/'run.json').read_text())
            library.unlink()
            failed = execute(expected_returncode=1)
            self.assertIn(str(library), failed.stderr)
            self.assertIn('docs/preparation.md', failed.stderr)
            self.assertNotIn('FileNotFoundError', failed.stderr)
            self.assertFalse((root/'events.jsonl').exists())
            self.assertFalse(list(run.rglob('resources.jsonl')))
            self.assertFalse(list((run/'evaluation/capture').glob('session-*')))
            current = json.loads((run/'run.json').read_text())
            for key in ('evaluation_sessions', 'pending_charge', 'consumed_seconds'):
                self.assertEqual(current.get(key), initial_state.get(key), key)

            library.write_bytes(b'synthetic pinned library')
            self.assertEqual(runtime_identity(config), identity)
            self.assertEqual(runtime_binding(config), inputs['identity']['runtime'])
            generation = execute('--json')
            report = json.loads((run/'evaluation/report.json').read_text())
            self.assertEqual(report['status'], 'PENDING_REVIEW')
            self.assertEqual(json.loads(generation.stdout), report)
            self.assertIn('3/3 (100%) | generation failures 0', generation.stderr)
            self.assertIn('Generation finished. Scoring and review are pending.', generation.stderr)
            result = execute()
            self.assertIn('Evaluation: PENDING_REVIEW', result.stdout)
            self.assertIn('Review: ' + report['review_path'], result.stdout)
            self.assertIn('Report: ' + str(run/'evaluation/report.json'), result.stdout)
            self.assertNotIn('"per_example"', result.stdout)
            self.assertNotIn('"aggregate"', result.stdout)
            self.assertLess(len(result.stdout), 2000)
            self.assertLess(len(result.stdout), len(json.dumps(report)))
            for banner in ('Synthetic tokenizer library banner', 'Synthetic tokenizer library warning'):
                self.assertNotIn(banner, result.stdout + result.stderr)
                self.assertNotIn(banner, generation.stdout + generation.stderr)
            events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
            self.assertEqual(sum(event['kind'] == 'start' for event in events), 1)
            self.assertEqual(sum(event['kind'] == 'tokenize' for event in events), 3)
            completions = [event for event in events if event['kind'] == 'completion']
            self.assertEqual(len(completions), 3)
            self.assertEqual(len({tuple(event['prompt']) for event in completions}), 3)
            self.assertTrue(all(event['cache_prompt'] is False and event['n_cache_reuse'] == 0
                                for event in completions))
            for event in completions:
                self.assertEqual(event['temperature'], 0)
                self.assertEqual(event['seed'], 42)
            answers = [json.loads(line) for line in (run/'evaluation/answers.jsonl').read_text().splitlines()]
            self.assertEqual([answer['id'] for answer in answers], ['0', '1', '2'])
            self.assertTrue(all(answer['answer']['status'] == 'COMPLETE' for answer in answers))
            self.assertEqual([answer['answer']['format_valid'] for answer in answers], [True, False, True])
            self.assertEqual(len(list((run/'evaluation/results').glob('*.json'))), 3)
            state = json.loads((run/'run.json').read_text())
            self.assertEqual(len(state['evaluation_sessions']), 1)
            self.assertIsNone(state['pending_charge'])
            session = run/'evaluation/capture'/state['evaluation_sessions'][0]['name']
            request = json.loads((session/'request.json').read_text())
            self.assertEqual(request, dict(schema='evaluation-session-v1', identity=inputs['identity'],
                config=config, cases=[dict(id=case['id'], messages=case['messages']) for case in cases]))
            execution = json.loads((session/'execution/configuration.json').read_text())
            self.assertEqual(execution['limits']['phase_seconds'], 30)
            self.assertEqual(execution['limits']['total_seconds'], 30)
            self.assertEqual(json.loads((session/'progress.json').read_text()), dict(index=2, state='COMPLETE'))
            self.assertFalse((session/'execution/request.json').exists())
            self.assertIn('Synthetic tokenizer library banner', (session/'execution/stdout.log').read_text())
            self.assertIn('Synthetic tokenizer library warning', (session/'execution/stderr.log').read_text())
            ledger = [json.loads(line) for line in (session/'resources.jsonl').read_text().splitlines()]
            self.assertEqual(len(ledger), 1)
            self.assertEqual(ledger[0]['status'], 'EXITED')
            self.assertGreater(state['consumed_seconds'], 0)
            self.assertEqual(state['consumed_seconds'], ledger[0]['elapsed_seconds'])

            # Completed commands recover without model or runtime availability.
            server.unlink()
            (root/'model.gguf').unlink()
            (root/'manifest.json').unlink()
            library.unlink()
            for options in ((), ('--report-only',)):
                self.assertEqual(evaluate(*options)['status'], 'PENDING_REVIEW')
                current = json.loads((run/'run.json').read_text())
                self.assertEqual(current['consumed_seconds'], state['consumed_seconds'])
                self.assertEqual(current['evaluation_sessions'], state['evaluation_sessions'])
            full_report = execute('--report-only', '--json')
            self.assertEqual(json.loads(full_report.stdout), json.loads((run/'evaluation/report.json').read_text()))
            self.assertEqual(json.loads(full_report.stdout), report)
            self.assertEqual(full_report.stderr, '')
            self.assertEqual([json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()], events)

            page = (run/'evaluation/review/index.html').read_text()
            self.assertNotIn('<script>bad()', page)
            self.assertNotIn('<textarea', page)
            self.assertLess(page.index('Source cells'), page.index('Reference'))
            self.assertLess(page.index('Reference'), page.index('Answer'))
            decisions = json.loads((run/'evaluation/review/template.json').read_text())
            self.assertEqual([decision['id'] for decision in decisions], ['0', '2'])
            for decision in decisions:
                decision['reviewed'] = True
            review = root/'review.json'
            review.write_text(json.dumps(decisions))
            imported = execute('--decisions', str(review), '--json', action='review-import', expected_returncode=0)
            completed = json.loads(imported.stdout)
            self.assertEqual(completed['status'], 'COMPLETE')
            self.assertTrue(completed['valid_accuracy'])
            self.assertEqual(completed['targets'], 3)
            self.assertEqual(completed['aggregate']['format_valid'], 2)
            self.assertEqual(completed['aggregate']['source_value_correct'], 2)
            repeated = execute('--decisions', str(review), '--json', action='review-import', expected_returncode=0)
            self.assertEqual(json.loads(repeated.stdout), completed)
            decisions[0]['quotation_errors'] = ['issuer_name']
            review.write_text(json.dumps(decisions))
            conflict = execute('--decisions', str(review), action='review-import', expected_returncode=1)
            self.assertIn('Review conflicts with a committed decision', conflict.stderr)
            regenerated = execute('--report-only', '--json', expected_returncode=0)
            self.assertEqual(json.loads(regenerated.stdout), completed)
            self.assertEqual(json.loads((run/'evaluation/report.json').read_text()), completed)
            current = json.loads((run/'run.json').read_text())
            self.assertEqual(current['consumed_seconds'], state['consumed_seconds'])
            self.assertEqual(current['evaluation_sessions'], state['evaluation_sessions'])
            self.assertEqual([json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()], events)
            artifact_root = os.environ.get('PROXYBENCH_TEST_ARTIFACT_DIR')
            if artifact_root:
                artifact = Path(artifact_root)/'evaluation-command'
                shutil.copytree(run, artifact)
                replay = subprocess.run([*command, 'evaluate', '--run-dir', str(artifact), '--report-only', '--json'],
                    cwd=root, env=environment, capture_output=True, text=True, timeout=20)
                self.assertEqual(replay.returncode, 0, replay.stderr)
                self.assertEqual(json.loads(replay.stdout), completed)


if __name__ == '__main__':
    unittest.main()
