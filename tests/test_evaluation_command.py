"""Run the public evaluation command with a synthetic subprocess model server."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

from proxybench.evaluation.workflow import create_run, runtime_binding
from proxybench.extraction.runtime import source_messages
from proxybench.runstate import binding, file_hash
from proxybench.training.labels import FIELDS


TOKENIZER = '''
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
        tokens = list((root / 'response.json').read_bytes()) + [9999]
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
            (root/'manifest.json').write_text(json.dumps(dict(source_commit='synthetic',
                files={'llama-server': file_hash(server)})))
            reference = dict(fields={name: dict(value=None, raw_text=None,
                availability='ABSENT_IN_CONTEXT', origin=None) for name in FIELDS})
            (root/'response.json').write_text(json.dumps(reference))
            config = dict(server=str(server), model=str(root/'model.gguf'),
                tokenizer=str(root/'tokenizer'), runtime_manifest=str(root/'manifest.json'),
                source_commit='synthetic', library_path='', server_arguments=[],
                system_prompt=str(root/'prompt.txt'), input_tokens=4096,
                response_tokens=4096, context_tokens=8192, startup_seconds=5,
                natural_request_seconds=5, limits=dict(cpu_only=True, phase_seconds=10,
                    total_seconds=30, start_host_bytes=0, stop_host_bytes=0,
                    device_margin_bytes=0, heartbeat_seconds=15))
            cases = [dict(id=str(index), reference=reference, source_cells=[],
                messages=source_messages(f'BEGIN MARKED TARGET case {index} END MARKED TARGET', config))
                for index in range(3)]
            inputs = dict(identity=dict(dataset='synthetic', model=file_hash(config['model']),
                runtime=runtime_binding(config), prompt=binding([case['messages'][0] for case in cases])),
                cases=cases)
            run = root/'run'
            create_run(run, inputs, config)
            source = Path(__file__).resolve().parents[1]/'src'
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='',
                               PYTHONPATH=os.pathsep.join((str(root), str(source))))
            command = [sys.executable, '-m', 'proxybench', 'evaluate', '--run-dir', str(run)]

            def evaluate(*extra):
                result = subprocess.run([*command, *extra], cwd=root, env=environment,
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                return json.loads((run/'evaluation/report.json').read_text())

            self.assertEqual(evaluate()['status'], 'PENDING_REVIEW')
            events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
            self.assertEqual(sum(event['kind'] == 'start' for event in events), 1)
            self.assertEqual(sum(event['kind'] == 'tokenize' for event in events), 3)
            completions = [event for event in events if event['kind'] == 'completion']
            self.assertEqual(len(completions), 3)
            self.assertEqual(len({tuple(event['prompt']) for event in completions}), 3)
            self.assertTrue(all(event['cache_prompt'] is False and event['n_cache_reuse'] == 0
                                for event in completions))
            answers = [json.loads(line) for line in (run/'evaluation/answers.jsonl').read_text().splitlines()]
            self.assertEqual([answer['id'] for answer in answers], ['0', '1', '2'])
            self.assertTrue(all(answer['answer']['status'] == 'COMPLETE' for answer in answers))
            self.assertEqual(len(list((run/'evaluation/results').glob('*.json'))), 3)
            state = json.loads((run/'run.json').read_text())
            self.assertEqual(len(state['evaluation_sessions']), 1)
            self.assertIsNone(state['pending_charge'])
            session = run/'evaluation/capture'/state['evaluation_sessions'][0]['name']
            ledger = [json.loads(line) for line in (session/'resources.jsonl').read_text().splitlines()]
            self.assertEqual(len(ledger), 1)
            self.assertEqual(ledger[0]['status'], 'EXITED')
            self.assertGreater(state['consumed_seconds'], 0)
            self.assertEqual(state['consumed_seconds'], ledger[0]['elapsed_seconds'])

            # Completed commands recover without model or runtime availability.
            server.unlink()
            (root/'model.gguf').unlink()
            (root/'manifest.json').unlink()
            for options in ((), ('--report-only',)):
                self.assertEqual(evaluate(*options)['status'], 'PENDING_REVIEW')
                current = json.loads((run/'run.json').read_text())
                self.assertEqual(current['consumed_seconds'], state['consumed_seconds'])
                self.assertEqual(current['evaluation_sessions'], state['evaluation_sessions'])
            self.assertEqual([json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()], events)


if __name__ == '__main__':
    unittest.main()
