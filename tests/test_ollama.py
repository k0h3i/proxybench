"""Streaming failures must preserve output without claiming completed extraction."""

import io
import json
from pathlib import Path
import tempfile
import unittest

from proxybench.extraction.ollama import capture_stream


class OllamaCaptureTests(unittest.TestCase):
    def capture(self, events, expected=12):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        stream = io.BytesIO(b''.join(json.dumps(x).encode() + b'\n' for x in events))
        return directory, stream, expected

    def test_complete_unicode_and_exact_tokens(self):
        directory, stream, expected = self.capture([
            {'response': 'Élan'}, {'response': ' vote'},
            {'done': True, 'done_reason': 'stop', 'prompt_eval_count': 12, 'eval_count': 3}])
        result = capture_stream(stream, directory, expected)
        self.assertTrue(result['terminated'])
        self.assertEqual((directory / 'raw.txt').read_text(), 'Élan vote')
        self.assertEqual(len((directory / 'arrivals.jsonl').read_text().splitlines()), 3)

    def test_length_or_prompt_mismatch_cannot_pass(self):
        for reason, tokens in [('length', 12), ('stop', 11)]:
            directory, stream, expected = self.capture([
                {'done': True, 'done_reason': reason, 'prompt_eval_count': tokens}])
            self.assertFalse(capture_stream(stream, directory, expected)['terminated'])

    def test_disconnect_and_server_error_preserve_partial(self):
        for tail in [[], [{'error': 'GPU failure'}]]:
            directory, stream, expected = self.capture([{'response': '{partial'}] + tail)
            with self.assertRaises(RuntimeError):
                capture_stream(stream, directory, expected)
            self.assertEqual((directory / 'raw.txt').read_text(), '{partial')
            self.assertFalse((directory / 'capture.json').exists())

    def test_existing_output_is_not_overwritten(self):
        directory, stream, expected = self.capture([])
        (directory / 'stream.jsonl').write_bytes(b'accepted')
        with self.assertRaises(FileExistsError):
            capture_stream(stream, directory, expected)
        self.assertEqual((directory / 'stream.jsonl').read_bytes(), b'accepted')
