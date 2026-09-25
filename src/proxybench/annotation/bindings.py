"""Bind review answers and scored references to preserved input artifacts."""

import hashlib
import json
from pathlib import Path

def require(condition, message):
    if not condition:
        raise ValueError(message)



def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def review_binding(packet):
    """Require the exact source-only model bundle when preparing a new review."""
    require(isinstance(packet.get("model_input"), str) and bool(packet["model_input"]),
            "A review packet requires its exact source message")
    return {"binding_version": "review-binding-v1",
            "manifest_sha256": sha256(canonical_bytes(packet["manifest"])),
            "review_view_sha256": sha256(packet["source_view"].encode("utf-8")),
            "model_input_sha256": sha256(packet["model_input"].encode("utf-8"))}


def checked_file(root, path, expected_hash):
    require(isinstance(path, str) and bool(path), "Artifact path is required")
    root = Path(root).resolve()
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root), "Artifact path leaves the supplied workspace")
    raw = resolved.read_bytes()
    require(sha256(raw) == expected_hash, f"Artifact hash mismatch: {path}")
    return raw
