"""Validate legacy development inventory against preserved acceptance evidence."""

import hashlib
import json
from pathlib import Path
import sys

from proxybench.annotation.bindings import checked_file
from proxybench.annotation.packets import FIELDS
from proxybench.schemas.records import require, strict_json


AUDIT_VERSION = "pilot-inventory-audit-v3"


def audit_inventory(root, authority, packet_paths):
    """Authority contains pinned paths and hashes from the recorded acceptance."""
    root = Path(root)
    documents = {key: strict_json(checked_file(root, item["path"], item["sha256"]))
                 for key, item in authority["artifacts"].items()}
    accepted, provisional, decision, exported = [documents[key] for key in
                                                ("accepted", "provisional", "decision", "export")]
    artifacts = authority["artifacts"]
    require(bool(authority.get("acceptance_basis")), "Preserved human acceptance basis is required")
    require(accepted.get("status") == "accepted_development_reference", "Accepted label set has contradictory status")
    require(provisional.get("status") == "policy_updated_provisional_development"
            and provisional.get("human_accepted") is False, "Provisional authority is inconsistent")
    require(accepted["split"] == provisional["split"] == "development", "Pilot examples must remain development")
    require(decision["latest_accepted_labels"] == artifacts["accepted"]["path"]
            and decision["latest_provisional_labels"] == artifacts["provisional"]["path"], "Decision label paths differ")
    require(accepted["source_export"] == artifacts["export"]["path"]
            and accepted["source_export_sha256"] == artifacts["export"]["sha256"], "Acceptance export identity differs")
    require(accepted["source_session_id"] == exported["session_id"]
            and accepted["source_revision"] == exported["revision"]
            and accepted["review_saved_at"] == exported["saved_at"], "Acceptance export session differs")
    require(set(accepted["packets"]) == set(authority["accepted_ids"])
            and set(provisional["packets"]) == set(authority["provisional_ids"]), "Label inventory differs from authority")
    require(not (set(accepted["packets"]) & set(provisional["packets"])), "Accepted and provisional IDs overlap")
    require(set(exported["packets"]) == set(accepted["packets"]), "Review export packet inventory differs")
    require(decision["human_accepted_examples"] == len(accepted["packets"])
            and decision["provisional_examples"] == len(provisional["packets"]), "Decision counts differ from authority")
    require(exported["completed_packets"] == len(accepted["packets"]), "Review export is incomplete")
    for pid, label in accepted["packets"].items():
        require(label.get("status") == "accepted_development_reference", "Packet acceptance state differs")
        reviewed = exported["packets"][pid]
        require(reviewed.get("reviewed") is True, "Packet lacks completed review evidence")
        require(label["packet_version"] == reviewed["packet_version"] and label["source_sha256"] == reviewed["source_sha256"],
                "Accepted packet differs from reviewed source")
        fingerprints = []
        if "packet_fingerprint" in reviewed:
            fingerprints.append(reviewed["packet_fingerprint"])
        if isinstance(reviewed.get("assistant_draft"), dict) and "packet_fingerprint" in reviewed["assistant_draft"]:
            fingerprints.append(reviewed["assistant_draft"]["packet_fingerprint"])
        require(bool(fingerprints) and all(value == label["packet_fingerprint"] for value in fingerprints),
                "Review export manifest binding differs")
        for field in FIELDS:
            for key in ("value", "raw_text", "availability", "origin", "evidence_input"):
                require(label["fields"][field][key] == reviewed["fields"][field][key], "Accepted field differs from preserved review export")
    for label in provisional["packets"].values():
        require(label.get("status") == "policy_updated_provisional_development", "Provisional packet status differs")
    rows, seen, hashes = [], set(), {}
    for path in packet_paths:
        raw = (root / path).read_bytes()
        hashes[path] = hashlib.sha256(raw).hexdigest()
        packets = strict_json(raw)
        if isinstance(packets, dict):
            packets = packets["packets"]
        for packet in packets:
            manifest = packet["manifest"]
            pid = manifest["packet_id"]
            require(pid not in seen, "Duplicate packet in inventory")
            seen.add(pid)
            source = checked_file(root, manifest["source_path"], manifest["source_sha256"])
            require(manifest["split"] == "development", "Packet split changed")
            for span in manifest["blocks"] + [manifest["target"]]:
                start, end = span["start_byte"], span["end_byte"]
                require(type(start) is int and type(end) is int and 0 <= start < end <= len(source), "Invalid source range")
                expected = span.get("source_slice_sha256", span.get("sha256"))
                require(hashlib.sha256(source[start:end]).hexdigest() == expected, "Source slice hash differs")
            target = manifest["target"]
            require(sum(b["start_byte"] <= target["start_byte"] < target["end_byte"] <= b["end_byte"]
                        for b in manifest["blocks"]) == 1, "Logical target must lie in exactly one context block")
            human = pid in accepted["packets"]
            labels = accepted if human else provisional
            require(pid in labels["packets"], "Packet lacks matching label authority")
            label = labels["packets"][pid]
            require(set(label["fields"]) == set(FIELDS), "Label field inventory differs")
            fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            require(label["source_sha256"] == manifest["source_sha256"]
                    and label["packet_version"] == manifest["packet_version"]
                    and label["packet_fingerprint"] == fingerprint, "Label manifest binding differs")
            blocks = {b["block_id"]: b for b in manifest["blocks"]}
            for field in label["fields"].values():
                for evidence in field.get("evidence", []):
                    require(evidence["block_id"] in blocks, "Evidence names an unknown block")
                    block = blocks[evidence["block_id"]]
                    require(block["start_byte"] <= evidence["start_byte"] < evidence["end_byte"] <= block["end_byte"],
                            "Evidence leaves the supplied block")
            rows.append({"packet_id": pid, "source_sha256": manifest["source_sha256"], "accession": manifest["accession"],
                         "label_status": "HUMAN_ACCEPTED_LEGACY" if human else "PROVISIONAL", "split": "development",
                         "conversion_status": "NOT_CONVERTED", "field_count": len(label["fields"]),
                         "unresolved_fields": {k: f["availability"] for k, f in label["fields"].items()
                                               if f["availability"] in ("AMBIGUOUS", "UNREADABLE", "CONFLICTING")}})
    require(seen == set(accepted["packets"]) | set(provisional["packets"]), "Packet inventory is incomplete")
    return {"audit_version": AUDIT_VERSION, "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "python_optimization": sys.flags.optimize, "status": "PASS", "packets": rows,
            "accepted_development_examples": len(accepted["packets"]), "provisional_development_examples": len(provisional["packets"]),
            "source_document_count": len({r["source_sha256"] for r in rows}), "packet_hashes": hashes,
            "authority": authority, "limitations": ["Legacy acceptance consistency only; not typed reference admission.",
            "Mechanical source integrity does not establish semantic support or display equivalence."]}
