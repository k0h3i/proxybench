from pathlib import Path
import tempfile
import unittest

from proxybench.annotation.packets import prepare_packet


class PacketTests(unittest.TestCase):
    def test_preserves_utf8_offsets_and_blank_cells_without_predictions(self):
        source = '<p>Fund é</p><table><tr><td>01</td><td>A &amp; B</td><td></td></tr></table>'.encode()
        start = source.index(b"<tr>")
        end = source.index(b"</tr>") + 5
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "source.htm"
            path.write_bytes(source)
            manifest, view = prepare_packet(root, "cal-001", "source.htm", "https://example.invalid", "test",
                                            [(0, len(source))], (start, end))
            self.assertIn("<td></td>", view)
            self.assertEqual(view.count('data-target="true"'), 1)
            self.assertEqual(path.read_bytes(), source)
            self.assertEqual(manifest["target"]["start_byte"], start)
            self.assertIsNone(manifest["predictions"])
            self.assertNotIn("field_values", manifest)
            self.assertEqual((Path(root) / "data/normalized/calibration/cal-001/block-1.source").read_bytes(), source)
            with self.assertRaises(FileExistsError):
                prepare_packet(root, "cal-001", "source.htm", "url", "test", [(0, len(source))], (start, end))

    def test_rejects_partial_target_and_overlapping_context(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "source.txt").write_bytes(b"first record\nsecond record\n")
            for spans, target in [([(0, 5)], (0, 8)), ([(0, 12), (8, 25)], (1, 4)), ([(0, 25)], (3, 3))]:
                with self.assertRaises(ValueError):
                    prepare_packet(root, "bad", "source.txt", "url", "test", spans, target)

    def test_plain_text_target_does_not_mark_neighbor(self):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "source.txt").write_text("Alice FOR, Bob WITHHOLD", encoding="utf-8")
            _, view = prepare_packet(root, "cal-001", "source.txt", "url", "test", [(0, 23)], (0, 9))
            self.assertIn('<mark data-target="true">Alice FOR</mark>, Bob WITHHOLD', view)

    def test_continuation_rows_share_one_target_range(self):
        raw = b'<tr><td>01</td><td>Approve</td></tr><tr><td></td><td>the plan</td></tr>'
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / 'source.htm').write_bytes(raw)
            manifest, view = prepare_packet(root, 'continuation', 'source.htm', 'url', 'test', [(0, len(raw))], (0, len(raw)))
            self.assertEqual(view.count('data-target="true"'), 2)
            self.assertIsInstance(manifest['target'], dict)
            self.assertIn('<table style=', view)
