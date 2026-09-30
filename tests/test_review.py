import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import re

from proxybench.annotation.bindings import review_binding
from proxybench.training.labels import FIELDS
from proxybench.annotation.review import load_suggestions, write_review


class ReviewTests(unittest.TestCase):
    def fixture(self, root):
        manifest = {'split': 'development', 'packet_id': 'synthetic-1', 'packet_version': 1, 'source_sha256': 'synthetic-hash',
                    'blocks': [{'block_id': 'B1', 'start_byte': 0, 'end_byte': 10}]}
        packets = [{'manifest': manifest, 'source_view': '<p>Synthetic source</p>',
                    'model_input': '<p>Synthetic source</p>'}]
        draft = {'packet_version': 1, 'source_sha256': 'synthetic-hash',
                 'packet_fingerprint': hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                 'fields': {f: {'value': '', 'raw_text': '', 'availability': 'ABSENT_IN_CONTEXT',
                                'origin': '', 'evidence_input': 'B1: absent', 'note': ''} for f in FIELDS}}
        draft['input_binding'] = review_binding(packets[0])
        path = root / 'drafts.json'
        path.write_text(json.dumps({'draft_set_id': 'synthetic-v1', 'packets': {'synthetic-1': draft}}))
        directory = root / 'review'
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

    def test_literal_template_markers_round_trip_in_source_and_drafts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packets, path = self.fixture(root)
            markers = '__PACKETS__ __FIELDS__ __DRAFTS__ </script><script>literal</script>'
            packets[0]['source_view'] = '<p>' + markers + '</p>'
            data = json.loads(path.read_text())
            data['packets']['synthetic-1']['input_binding'] = review_binding(packets[0])
            data['packets']['synthetic-1']['fields']['ticker']['note'] = markers
            path.write_text(json.dumps(data))
            source_path = root / 'review/packet-set.json'
            source_path.write_text(json.dumps(packets))
            original = source_path.read_bytes()
            raw_drafts = path.read_bytes()
            page = write_review(root, packet_directory="review",
                                training_contract="Contract <script>data</script>", draft_path=path).read_text()
            payload = json.loads(re.search(r'<script id="packet-data" type="application/json">(.*?)</script>', page, re.S)[1])
            drafts = json.loads(re.search(r'<script id="assistant-data" type="application/json">(.*?)</script>', page, re.S)[1])
            self.assertEqual(payload[0]['source_view'], packets[0]['source_view'])
            self.assertEqual(drafts['packets']['synthetic-1']['fields']['ticker']['note'], markers)
            self.assertNotIn('</script><script>literal</script>', page)
            self.assertIn('&lt;script&gt;data&lt;/script&gt;', page)
            self.assertRegex(page, r'<button\b(?=[^>]*id="reveal")(?=[^>]*\bdisabled)[^>]*>')
            self.assertEqual(source_path.read_bytes(), original)
            self.assertEqual(path.read_bytes(), raw_drafts)

    def test_stale_display_and_model_bundle_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            packets, path = self.fixture(Path(directory))
            for key in ('source_view', 'model_input'):
                original = packets[0][key]
                packets[0][key] += 'changed'
                with self.assertRaises(ValueError):
                    load_suggestions(path, packets)
                packets[0][key] = original
            data = json.loads(path.read_text())
            del data['packets']['synthetic-1']['input_binding']
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                load_suggestions(path, packets)

    def test_existing_page_is_preserved_and_unknown_packets_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packets, path = self.fixture(root)
            output = write_review(root, packet_directory="review", training_contract="Contract", draft_path=path)
            original = output.read_bytes()
            with self.assertRaises(FileExistsError):
                write_review(root, packet_directory="review", training_contract="Contract", draft_path=path)
            self.assertEqual(output.read_bytes(), original)
            packets[0]['manifest']['split'] = 'unknown'
            (root / 'review/packet-set.json').write_text(json.dumps(packets))
            with self.assertRaisesRegex(ValueError, 'training, development, and test'):
                write_review(root, packet_directory="review", training_contract="Contract")
