import io
import json
from pathlib import Path
import tempfile
import unittest
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
