from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from proxybench.annotation.audit import audit_inventory
from proxybench.annotation.packets import FIELDS


def fixture(root):
    raw = b"Synthetic source"
    (root / "source.html").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    base = {"split": "development", "source_path": "source.html", "source_sha256": digest, "packet_version": 1,
            "accession": "synthetic", "blocks": [{"block_id": "B1", "start_byte": 0, "end_byte": len(raw), "source_slice_sha256": digest}],
            "target": {"start_byte": 0, "end_byte": len(raw), "sha256": digest}}
    labels, packets = {}, []
    for pid in ("accepted", "provisional"):
        m = {**base, "packet_id": pid}
        packets.append({"manifest": m})
        labels[pid] = {"status": "accepted_development_reference" if pid == "accepted" else "policy_updated_provisional_development",
                       "source_sha256": digest, "packet_version": 1,
                       "packet_fingerprint": hashlib.sha256(json.dumps(m, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                       "fields": {k: {"value": "", "raw_text": "", "availability": "ABSENT_IN_CONTEXT", "origin": "", "evidence_input": ""} for k in FIELDS}}
    export = {"session_id": "synthetic-session", "revision": 1, "saved_at": "2026-09-07", "completed_packets": 1,
              "packets": {"accepted": {**deepcopy(labels["accepted"]), "reviewed": True}}}
    write(root / "export.json", export)
    accepted = {"status": "accepted_development_reference", "split": "development", "packets": {"accepted": labels["accepted"]},
                "source_export": "export.json", "source_export_sha256": digest_file(root / "export.json"),
                "source_session_id": "synthetic-session", "source_revision": 1, "review_saved_at": "2026-09-07"}
    provisional = {"status": "policy_updated_provisional_development", "human_accepted": False, "split": "development",
                   "packets": {"provisional": labels["provisional"]}}
    decision = {"latest_accepted_labels": "accepted.json", "latest_provisional_labels": "provisional.json",
                "human_accepted_examples": 1, "provisional_examples": 1}
    for name, data in [("accepted", accepted), ("provisional", provisional), ("decision", decision), ("packets", packets)]:
        write(root / (name + ".json"), data)
    return {"artifacts": {k: {"path": k + ".json", "sha256": digest_file(root / (k + ".json"))}
                          for k in ("accepted", "provisional", "decision", "export")},
            "accepted_ids": ["accepted"], "provisional_ids": ["provisional"], "acceptance_basis": "Synthetic recorded human decision"}


def write(path, data):
    path.write_text(json.dumps(data))


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AuditTests(unittest.TestCase):
    def test_acceptance_states_paths_export_and_counts(self):
        mutations = [("accepted", lambda d: d.update(status="provisional")),
                     ("accepted", lambda d: d["packets"]["accepted"].update(status="provisional")),
                     ("decision", lambda d: d.update(latest_accepted_labels="different.json")),
                     ("decision", lambda d: d.update(human_accepted_examples=2)),
                     ("export", lambda d: d["packets"]["accepted"].update(reviewed=False)),
                     ("accepted", lambda d: d["packets"]["accepted"]["fields"]["ticker"].update(value="edited"))]
        for name, change in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                authority = fixture(root)
                self.assertEqual(audit_inventory(root, authority, ["packets.json"])["accepted_development_examples"], 1)
                p = root / (name + ".json")
                data = json.loads(p.read_text())
                change(data)
                write(p, data)
                # Update the pinned digest to exercise consistency checks as well as hash checks.
                authority["artifacts"][name]["sha256"] = digest_file(p)
                with self.assertRaises(ValueError):
                    audit_inventory(root, authority, ["packets.json"])

    def test_export_fingerprints_with_refreshed_authority(self):
        for location in ("top", "legacy", "both"):
            for mutation in ("intact", "conflict", "missing"):
                with self.subTest(location=location, mutation=mutation), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    authority = fixture(root)
                    exported = json.loads((root / "export.json").read_text())
                    reviewed = exported["packets"]["accepted"]
                    fingerprint = reviewed.pop("packet_fingerprint")
                    if location in ("top", "both"):
                        reviewed["packet_fingerprint"] = fingerprint
                    if location in ("legacy", "both"):
                        reviewed["assistant_draft"] = {"packet_fingerprint": fingerprint}
                    target = reviewed if location == "top" else reviewed["assistant_draft"]
                    if mutation == "conflict":
                        target["packet_fingerprint"] = "f" * 64
                    elif mutation == "missing":
                        reviewed.pop("packet_fingerprint", None)
                        reviewed.pop("assistant_draft", None)
                    write(root / "export.json", exported)
                    with self.assertRaises(ValueError):
                        audit_inventory(root, authority, ["packets.json"])
                    authority["artifacts"]["export"]["sha256"] = digest_file(root / "export.json")
                    accepted = json.loads((root / "accepted.json").read_text())
                    accepted["source_export_sha256"] = authority["artifacts"]["export"]["sha256"]
                    write(root / "accepted.json", accepted)
                    authority["artifacts"]["accepted"]["sha256"] = digest_file(root / "accepted.json")
                    if mutation == "intact":
                        self.assertEqual(audit_inventory(root, authority, ["packets.json"])["status"], "PASS")
                    else:
                        with self.assertRaisesRegex(ValueError, "Review export manifest binding differs"):
                            audit_inventory(root, authority, ["packets.json"])

    def test_integrity_checks_survive_optimized_python(self):
        for mutation in ("hash", "range"):
            for optimized in (False, True):
                with self.subTest(mutation=mutation, optimized=optimized), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    authority = fixture(root)
                    packets = json.loads((root / "packets.json").read_text())
                    if mutation == "hash":
                        packets[0]["manifest"]["source_sha256"] = "f" * 64
                    else:
                        packets[0]["manifest"]["target"]["end_byte"] = 1000
                    write(root / "packets.json", packets)
                    write(root / "authority.json", authority)
                    command = [sys.executable] + (["-O"] if optimized else []) + ["-c",
                        "import json,sys;from pathlib import Path;from proxybench.annotation.audit import audit_inventory;"
                        "r=Path(sys.argv[1]);audit_inventory(r,json.loads((r/'authority.json').read_text()),['packets.json'])", str(root)]
                    result = subprocess.run(command, capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("ValidationError", result.stderr)


if __name__ == "__main__":
    unittest.main()
