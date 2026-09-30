"""CPU checks for shared evaluation servers and durable failure boundaries."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from proxybench.evaluation.worker import run_session
from proxybench.extraction.llama_server import generate
from proxybench.extraction.runtime import generate_answers, source_messages, prompt_tokens
from proxybench.training.adapters import digest


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        text = messages[0]['content'] + '\n' + messages[1]['content']
        return list(text.encode()) if tokenize else text

    def decode(self, tokens, **kwargs):
        return ''.join(chr(token) for token in tokens)


class EvaluationWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.requests, self.starts = [], []
        self.fail_at = None
        self.terminal_override = {}
        self.config = dict(server=str(self.root/'llama-server'), model=str(self.root/'model.gguf'),
                           tokenizer='synthetic', source_commit='synthetic', external_libraries={},
                           runtime_manifest=str(self.root/'manifest.json'), library_path='',
                           server_arguments=[], startup_seconds=5, natural_request_seconds=5,
                           input_tokens=1000, response_tokens=4, context_tokens=1004,
                           system_prompt=str(self.root/'prompt.txt'), limits=dict(phase_seconds=10))
        for name in ('llama-server', 'model.gguf', 'prompt.txt'):
            (self.root/name).write_text('synthetic')
        (self.root/'manifest.json').write_text(json.dumps(dict(source_commit='synthetic',
            files={'llama-server': digest(self.root/'llama-server')})))
        self.cases = [dict(id=str(index), messages=source_messages(
            f'BEGIN MARKED TARGET case {index} END MARKED TARGET', self.config)) for index in range(3)]
        self.save_request()
        worker = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, body):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.send(b'{"status":"ok"}')

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/tokenize':
                    self.send(json.dumps(dict(tokens=list(body['content'].encode()))).encode())
                    return
                worker.requests.append(body)
                chunks = [dict(tokens=[65])]
                if len(worker.requests) == worker.fail_at:
                    chunks.append(dict(error='synthetic generation failure'))
                else:
                    chunks.append(dict(tokens=[99], stop=True, stop_type='eos', truncated=False,
                                       tokens_evaluated=len(body['prompt']), tokens_predicted=2,
                                       timings=dict(cache_n=0, prompt_n=len(body['prompt']))))
                    chunks[-1].update(worker.terminal_override)
                self.send(b''.join(b'data: '+json.dumps(chunk).encode()+b'\n\n' for chunk in chunks))

        def launch(command, **kwargs):
            self.starts.append(command)
            port = int(command[command.index('--port')+1])
            server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
            thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=.01), daemon=True)
            thread.start()
            def terminate():
                server.shutdown()
                server.server_close()
                thread.join()
            return SimpleNamespace(poll=lambda: None, terminate=terminate, wait=lambda **kw: 0)

        for patcher in (
            patch('proxybench.extraction.runtime.subprocess.Popen', side_effect=launch),
            patch.dict(sys.modules, transformers=SimpleNamespace(AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: Tokenizer()))),
            patch.dict(os.environ, PROXYBENCH_REQUEST_FILE=str(self.root/'deadline.json'),
                       PROXYBENCH_PHASE_FILE=str(self.root/'phase.json')),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def save_request(self):
        (self.root/'request.json').write_text(json.dumps(dict(schema='evaluation-session-v1',
            identity={'synthetic': True}, config=self.config, cases=self.cases)))

    def answer(self, index):
        return json.loads((self.root/'answers'/f'answer-{index}.json').read_text())

    def test_existing_inference_api_reuses_server_and_validates_before_launch(self):
        answers = generate_answers([case['messages'] for case in self.cases],
                                   self.root/'inference', self.config)
        self.assertEqual([answer['status'] for answer in answers], ['COMPLETE'] * 3)
        self.assertEqual(len(self.starts), 1)
        self.cases[1]['messages'][0]['content'] = 'changed policy'
        with self.assertRaisesRegex(ValueError, 'canonical prompt'):
            generate_answers([case['messages'] for case in self.cases],
                             self.root/'invalid-inference', self.config)
        self.assertEqual(len(self.starts), 1)

    def test_failure_retains_partial_tokens_and_stops_later_cases(self):
        self.fail_at = 2
        self.assertEqual(run_session(self.root), 1)
        self.assertEqual(len(self.requests), 2)
        answer = self.answer(1)
        self.assertEqual(answer['status'], 'FAILED')
        self.assertEqual(answer['token_ids'], [65])
        self.assertEqual(answer['text'], 'A')
        self.assertIn(b'synthetic generation failure', (self.root/'answers/answer-1.response.bin').read_bytes())
        self.assertFalse((self.root/'answers/answer-2.json').exists())

    def test_every_case_validates_its_prompt(self):
        self.cases[1]['messages'][0]['content'] = 'changed policy'
        self.save_request()
        self.assertEqual(run_session(self.root), 1)
        self.assertEqual(len(self.requests), 1)
        self.assertIn('canonical prompt', self.answer(1)['error'])

    def test_terminal_runtime_failure_stops_before_next_case(self):
        self.terminal_override = dict(tokens_predicted=3)
        self.assertEqual(run_session(self.root), 1)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.answer(0)['status'], 'TOKEN_COUNT_MISMATCH')

    def test_verified_length_stop_can_continue(self):
        self.config['response_tokens'] = 2
        self.save_request()
        self.terminal_override = dict(stop_type='limit')
        self.assertEqual(run_session(self.root), 0)
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(self.answer(0)['status'], 'LENGTH_STOP')

    def test_generation_retains_supervised_deadline_bounded_by_case_remaining(self):
        clock = time.monotonic
        offset = [0]
        def prepare(*args):
            offset[0] += 9
            return prompt_tokens(*args)
        def capture(*args, **kwargs):
            published = json.loads((self.root/'deadline.json').read_text())
            self.assertLessEqual(kwargs['deadline'], 1)
            self.assertGreater(kwargs['deadline'], .5)
            self.assertLessEqual(published['deadline_monotonic'] - time.monotonic(), 1)
            self.assertNotIn('PROXYBENCH_REQUEST_FILE', os.environ)
            answer = generate(*args, **kwargs)
            self.assertTrue((self.root/'deadline.json').exists())
            return answer
        with patch('proxybench.evaluation.worker.time.monotonic', side_effect=lambda: clock()+offset[0]), \
             patch('proxybench.evaluation.worker.prompt_tokens', side_effect=prepare), \
             patch('proxybench.evaluation.worker.generate', side_effect=capture):
            self.assertEqual(run_session(self.root), 0)

    def test_case_deadline_includes_prompt_preparation(self):
        clock = time.monotonic
        offset = [0]
        def prepare(*args):
            published = json.loads((self.root/'deadline.json').read_text())
            self.assertGreater(published['deadline_monotonic'], clock())
            offset[0] = self.config['limits']['phase_seconds'] + 1
            return prompt_tokens(*args)
        with patch('proxybench.evaluation.worker.time.monotonic', side_effect=lambda: clock()+offset[0]), \
             patch('proxybench.evaluation.worker.prompt_tokens', side_effect=prepare):
            self.assertEqual(run_session(self.root), 1)
        self.assertEqual(self.requests, [])
        self.assertEqual(self.answer(0)['status'], 'TIMEOUT')

    def test_startup_deadline_includes_authentication_and_has_no_active_case(self):
        clock = time.monotonic
        offset = [0]
        def authenticate(config):
            self.assertTrue((self.root/'deadline.json').exists())
            offset[0] = self.config['startup_seconds'] + 1
            return 'synthetic'
        with patch('proxybench.evaluation.worker.time.monotonic', side_effect=lambda: clock()+offset[0]), \
             patch('proxybench.extraction.runtime.runtime_identity', side_effect=authenticate):
            with self.assertRaises(TimeoutError):
                run_session(self.root)
        self.assertEqual(self.starts, [])
        self.assertFalse((self.root/'progress.json').exists())

    def test_interrupt_is_not_published_as_completed_failure(self):
        with patch('proxybench.evaluation.worker.prompt_tokens', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_session(self.root)
        self.assertEqual(json.loads((self.root/'progress.json').read_text()), dict(index=0, state='STARTED'))
        self.assertFalse((self.root/'answers/answer-0.json').exists())


if __name__ == '__main__':
    unittest.main()
