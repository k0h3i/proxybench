import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from proxybench.annotation.packets import FIELDS
from proxybench.annotation.review import load_suggestions, write_review


class ReviewTests(unittest.TestCase):
    def fixture(self, root):
        manifest = {'packet_id': 'synthetic-1', 'packet_version': 1, 'source_sha256': 'synthetic-hash',
                    'blocks': [{'block_id': 'B1', 'start_byte': 0, 'end_byte': 10}]}
        packets = [{'manifest': manifest, 'source_view': '<p>Synthetic source</p>'}]
        draft = {'packet_version': 1, 'source_sha256': 'synthetic-hash',
                 'packet_fingerprint': hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                 'fields': {f: {'value': '', 'raw_text': '', 'availability': 'ABSENT_IN_CONTEXT',
                                'origin': '', 'evidence_input': 'B1: absent', 'note': ''} for f in FIELDS}}
        path = root / 'drafts.json'
        path.write_text(json.dumps({'draft_set_id': 'synthetic-v1', 'packets': {'synthetic-1': draft}}))
        directory = root / 'data/packets/calibration'
        directory.mkdir(parents=True)
        (directory / 'packet-set.json').write_text(json.dumps(packets))
        return packets, path

    def test_rejects_suggestions_for_changed_context_or_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packets, path = self.fixture(root)
            self.assertIn('file_sha256', load_suggestions(path, packets))
            packets[0]['manifest']['blocks'][0]['end_byte'] = 9
            with self.assertRaises(ValueError):
                load_suggestions(path, packets)
            packets[0]['manifest']['blocks'][0]['end_byte'] = 10
            packets[0]['manifest']['source_sha256'] = 'different'
            with self.assertRaises(ValueError):
                load_suggestions(path, packets)

    def test_rejects_incomplete_field_drafts(self):
        with tempfile.TemporaryDirectory() as directory:
            packets, path = self.fixture(Path(directory))
            data = json.loads(path.read_text())
            del data['packets']['synthetic-1']['fields']['ticker']
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_suggestions(path, packets)

    def test_page_keeps_source_input_separate_and_escapes_suggestions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, path = self.fixture(root)
            source_path = root / 'data/packets/calibration/packet-set.json'
            original = source_path.read_bytes()
            data = json.loads(path.read_text())
            data['packets']['synthetic-1']['fields']['ticker']['note'] = '</script><script>alert(1)</script>'
            path.write_text(json.dumps(data))
            raw_drafts = path.read_bytes()
            page = write_review(root, draft_path=path).read_text()
            self.assertNotIn('</script><script>alert(1)</script>', page)
            self.assertIn('id="assistant-data"', page)
            self.assertIn('id="reveal" class="primary" disabled', page)
            self.assertEqual(source_path.read_bytes(), original)
            self.assertEqual(path.read_bytes(), raw_drafts)
