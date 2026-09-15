import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from proxybench.extraction.llama_server import assess_response, generate, request_body, require_prompt_tokens


def terminal():
    return dict(stop=True, stop_type='eos', tokens_predicted=2, tokens_evaluated=2, tokens=[],
                truncated=False, timings=dict(cache_n=0, prompt_n=2))


class LlamaCaptureTests(unittest.TestCase):
    def test_completion_requires_terminal_token_counts_and_no_reuse(self):
        kwargs = dict(prompt_length=2, eos=9, maximum=10, forced=False, elapsed=1, deadline=60)
        self.assertEqual(assess_response([1, 9], terminal(), **kwargs), 'COMPLETE')
        self.assertEqual(assess_response([1, 9], None, **kwargs), 'CAPTURE_INCOMPLETE')
        cached = {**terminal(), 'timings': dict(cache_n=1, prompt_n=1)}
        self.assertEqual(assess_response([1, 9], cached, **kwargs), 'PROMPT_OR_CACHE_MISMATCH')
        self.assertEqual(assess_response([1, 8], terminal(), **kwargs), 'TERMINATION_MISMATCH')
        self.assertEqual(assess_response([1], terminal(), **kwargs), 'TOKEN_COUNT_MISMATCH')
        self.assertEqual(assess_response([1, 9], terminal(), **{**kwargs, 'elapsed': 61}), 'TIMEOUT')
        self.assertEqual(assess_response([1, 9], {**terminal(), 'stop_type': 'limit'},
                                        **{**kwargs, 'maximum': 2, 'forced': True}), 'PROBE_COMPLETE')

    def test_integer_prompts_greedy_and_cache_controls(self):
        body = request_body([4, 5], 1792, False)
        self.assertEqual(body['prompt'], [4, 5])
        self.assertFalse(body['cache_prompt'])
        self.assertFalse(body['ignore_eos'])
        self.assertEqual(body['temperature'], 0)
        self.assertEqual(body['repeat_penalty'], 1)
        self.assertEqual(body['stop'], [])
        with self.assertRaises(ValueError):
            request_body(['answer'], 1792, False)
        with patch('proxybench.extraction.llama_server.json_request', return_value={'tokens': [4, 6]}):
            with self.assertRaisesRegex(ValueError, 'token IDs differ'):
                require_prompt_tokens(1, 'source', [4, 5])

    def test_stream_capture_keeps_chunk_bytes_and_rejects_missing_terminal(self):
        class Tokenizer:
            eos_token_id = 9

            def decode(self, tokens, **kwargs):
                return '{}' if tokens == [1] else '{}<end>'

        class Response(io.BytesIO):
            status = 200

        class Connection:
            sock = None
            chunks = []

            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                self.sent = json.loads(kwargs['body'])

            def getresponse(self):
                return Response(b''.join(b'data: ' + json.dumps(c).encode() + b'\n\n' for c in self.chunks))

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory, patch('proxybench.extraction.llama_server.http.client.HTTPConnection', Connection):
            Connection.chunks = [dict(tokens=[1, 9], stop=False), terminal()]
            p = Path(directory) / 'complete.json'
            result = generate(1, Tokenizer(), [4, 5], p, index=0)
            self.assertEqual(result['status'], 'COMPLETE')
            self.assertFalse(result['format_valid'])
            self.assertFalse(result['exact_token_arrivals'])
            self.assertEqual(result['maximum_observed_chunk_tokens'], 2)
            self.assertTrue(p.with_suffix('.response.bin').read_bytes().startswith(b'data: '))
            self.assertEqual(json.loads(p.with_suffix('.request.bin').read_bytes())['prompt'], [4, 5])
            Connection.chunks = [dict(tokens=[1, 9], stop=False)]
            result = generate(1, Tokenizer(), [4, 5], Path(directory) / 'incomplete.json', index=0)
            self.assertEqual(result['status'], 'CAPTURE_INCOMPLETE')
