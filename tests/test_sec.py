import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from http.client import IncompleteRead
from urllib.error import HTTPError

from proxybench.sources.sec import SecClient


URL = "https://www.sec.gov/Archives/edgar/data/1/000000000124000001/source.htm"
ACCESSION = "0000000001-24-000001"


class Response(io.BytesIO):
    status = 200
    headers = {"Content-Type": "text/html"}

    def geturl(self):
        return URL


class SecClientTests(unittest.TestCase):
    def test_interrupted_response_is_logged_without_cache_success(self):
        class Interrupted(Response):
            def read(self):
                raise IncompleteRead(b'partial', 100)

        with tempfile.TemporaryDirectory() as root:
            client = SecClient(root, 'Synthetic Test test@example.invalid', opener=lambda *a, **k: Interrupted())
            result = client.fetch(ACCESSION, URL)
            self.assertEqual(result['outcome'], 'network_error')
            self.assertEqual(result['stage'], 'transport')
            self.assertFalse(result['source_exists'])
            self.assertEqual(json.loads((Path(root) / 'data/manifests/retrieval-log.jsonl').read_text())['outcome'], 'network_error')

    def test_storage_failures_preserve_incomplete_cache(self):
        for failing_stage in ('source_write', 'metadata_write'):
            with self.subTest(stage=failing_stage), tempfile.TemporaryDirectory() as root:
                client = SecClient(root, 'Synthetic Test test@example.invalid', opener=lambda *a, **k: Response(b'original'))
                real_open = Path.open

                def failing_open(path, *args, **kwargs):
                    if (failing_stage == 'source_write' and path.name == 'source.htm'
                            or failing_stage == 'metadata_write' and path.name == 'source.htm.retrieval.json'):
                        raise OSError('Synthetic disk full')
                    return real_open(path, *args, **kwargs)

                with patch.object(Path, 'open', failing_open):
                    result = client.fetch(ACCESSION, URL)
                self.assertEqual(result['outcome'], 'storage_error')
                self.assertEqual(result['stage'], failing_stage)
                if failing_stage == 'metadata_write':
                    source = Path(root) / 'data/raw' / ACCESSION / 'source.htm'
                    with self.assertRaises(ValueError):
                        client.fetch(ACCESSION, URL)
                    self.assertEqual(source.read_bytes(), b'original')
                    log = [json.loads(line) for line in (Path(root) / 'data/manifests/retrieval-log.jsonl').read_text().splitlines()]
                    self.assertEqual(log[-1]['outcome'], 'incomplete_cache')

    def test_orphan_metadata_and_unwritable_log_are_not_silent(self):
        with tempfile.TemporaryDirectory() as root:
            metadata = Path(root) / 'data/raw' / ACCESSION / 'source.htm.retrieval.json'
            metadata.parent.mkdir(parents=True)
            metadata.write_text('{}')
            client = SecClient(root, 'Synthetic Test test@example.invalid')
            with self.assertRaises(ValueError):
                client.fetch(ACCESSION, URL)
            self.assertEqual(metadata.read_text(), '{}')
        with tempfile.TemporaryDirectory() as root:
            client = SecClient(root, 'Synthetic Test test@example.invalid', opener=lambda *a, **k: Response(b'original'))
            with patch('proxybench.sources.sec.append_jsonl', side_effect=OSError('Synthetic unwritable log')):
                with self.assertRaisesRegex(OSError, 'unwritable log'):
                    client.fetch(ACCESSION, URL)

    def test_corrupt_metadata_is_logged_without_replacing_bytes(self):
        for contents in ('{', '[]'):
            with self.subTest(contents=contents), tempfile.TemporaryDirectory() as root:
                client = SecClient(root, 'Synthetic Test test@example.invalid', opener=lambda *a, **k: Response(b'original'))
                event = client.fetch(ACCESSION, URL)
                source = Path(root) / event['path']
                metadata = source.with_name(source.name + '.retrieval.json')
                metadata.write_text(contents)
                with self.assertRaises(ValueError):
                    client.fetch(ACCESSION, URL)
                self.assertEqual(metadata.read_text(), contents)
                self.assertEqual(source.read_bytes(), b'original')
                last = (Path(root) / 'data/manifests/retrieval-log.jsonl').read_text().splitlines()[-1]
                self.assertEqual(json.loads(last)['outcome'], 'invalid_cache')

    def test_preserves_original_bytes_and_reuses_cache(self):
        calls = []

        def opener(request, timeout):
            calls.append(request)
            return Response(b"<p>Raw &amp; unchanged\r\n</p>")

        with tempfile.TemporaryDirectory() as root:
            client = SecClient(root, "Synthetic Test test@example.invalid", opener=opener)
            first = client.fetch(ACCESSION, URL)
            second = client.fetch(ACCESSION, URL)
            self.assertEqual(len(calls), 1)
            self.assertEqual(second["outcome"], "cache_hit")
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual((Path(root) / first["path"]).read_bytes(), b"<p>Raw &amp; unchanged\r\n</p>")
            self.assertEqual(calls[0].get_header("User-agent"), "Synthetic Test test@example.invalid")

    def test_denial_is_logged_and_stops_subsequent_requests(self):
        calls = []

        def opener(request, timeout):
            calls.append(request)
            raise HTTPError(URL, 403, "Forbidden", {}, None)

        with tempfile.TemporaryDirectory() as root:
            client = SecClient(root, "Synthetic Test test@example.invalid", opener=opener)
            self.assertEqual(client.fetch(ACCESSION, URL)["http_status"], 403)
            self.assertEqual(client.fetch(ACCESSION, URL)["outcome"], "deferred_after_access_denial")
            self.assertEqual(len(calls), 1)
            events = (Path(root) / "data/manifests/retrieval-log.jsonl").read_text().splitlines()
            self.assertEqual(len(events), 2)
            self.assertEqual(json.loads(events[0])["outcome"], "http_error")

    def test_cache_tampering_does_not_replace_source(self):
        with tempfile.TemporaryDirectory() as root:
            client = SecClient(root, "Synthetic Test test@example.invalid", opener=lambda *a, **k: Response(b"original"))
            result = client.fetch(ACCESSION, URL)
            path = Path(root) / result["path"]
            path.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                client.fetch(ACCESSION, URL)
            self.assertEqual(path.read_bytes(), b"changed")

    def test_identity_and_source_boundaries(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                SecClient(root, "ProxyBench")
            client = SecClient(root, "Synthetic Test test@example.invalid")
            for url in (URL.replace("www.sec.gov", "example.com"), URL + "?x=1", URL.replace("source.htm", "../source.htm")):
                with self.assertRaises(ValueError):
                    client.fetch(ACCESSION, url)
            with self.assertRaises(ValueError):
                client.fetch("0000000001-23-000001", URL)
