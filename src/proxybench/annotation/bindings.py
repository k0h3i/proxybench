"""Bind review answers and scored references to preserved input artifacts."""

import hashlib
import json
from pathlib import Path

from proxybench.schemas.records import object_keys, require, strict_json, string


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def review_binding(packet):
    """Require the exact source-only model bundle when preparing a new review."""
    require(isinstance(packet.get("model_input"), str) and bool(packet["model_input"]),
            "A new review packet requires its exact model_input text. Preserve legacy packets separately.")
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


def validate_input_binding(binding, root, *, input_id, model_input_sha256, require_review=True):
    pairs = ("source_manifest", "review_view", "model_input")
    keys = {"binding_version", "input_id", "review_tool_revision", "representation_decision", "reviewer", "reviewed_at", "reason"}
    keys.update(key + suffix for key in pairs for suffix in ("_path", "_sha256"))
    object_keys(binding, keys, "input_binding")
    require(binding["binding_version"] == "input-binding-v1" and binding["input_id"] == input_id,
            "Reference input identity mismatch")
    string(binding["review_tool_revision"], "review_tool_revision")
    for key in pairs:
        checked_file(root, binding[key + "_path"], binding[key + "_sha256"])
    require(binding["model_input_sha256"] == model_input_sha256, "Scoring input differs from reference input binding")
    require(binding["representation_decision"] in ("SAME_EVIDENCE", "CHANGED_EVIDENCE", "PENDING"),
            "Invalid representation decision")
    for key in ("reviewer", "reviewed_at", "reason"):
        string(binding[key], key, nullable=binding["representation_decision"] == "PENDING")
    if require_review:
        require(binding["representation_decision"] == "SAME_EVIDENCE", "Input evidence requires a completed representation review")
    return binding


def load_input_binding(reference, root, *, input_id, model_input_sha256):
    raw = checked_file(root, reference["input_binding_path"], reference["input_binding_sha256"])
    return validate_input_binding(strict_json(raw), root, input_id=input_id, model_input_sha256=model_input_sha256)


def save_new_version(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
