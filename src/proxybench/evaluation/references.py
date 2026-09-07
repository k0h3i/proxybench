"""Admit reviewed, input-specific references before scoring."""

from dataclasses import dataclass
from proxybench.normalization.values import LEGACY_GUIDE, policy_version

from proxybench.annotation.bindings import checked_file, load_input_binding
from proxybench.evaluation.evidence import record_citations
from proxybench.evaluation.projection import primary_paths
from proxybench.schemas.records import normalize_record, object_keys, require, string


@dataclass(frozen=True)
class Reference:
    reference_id: str
    input_id: str
    record: dict
    synthetic: bool = False
    guide_version: str = LEGACY_GUIDE


def synthetic_reference(record, *, reference_id="synthetic-reference", input_id="synthetic-input", guide_version=LEGACY_GUIDE):
    return Reference(reference_id, input_id, normalize_record(record, guide_version=guide_version), synthetic=True, guide_version=guide_version)


def admit_reference(value, root, context, *, input_id, model_input_sha256, guide_version):
    policy_version(guide_version)
    text_keys = {"reference_id", "input_id", "label_version", "label_status", "reviewer", "reviewed_at",
                 "guide_version", "original_label_path", "original_label_sha256", "input_binding_path", "input_binding_sha256"}
    object_keys(value, text_keys | {"record", "primary_paths", "conversion_log"}, "reference")
    for key in text_keys:
        string(value[key], key)
    require(value["label_status"] == "HUMAN_ACCEPTED", "Reference conversion is not accepted")
    require(value["input_id"] == input_id and value["guide_version"] == guide_version, "Reference identity or guide differs")
    checked_file(root, value["original_label_path"], value["original_label_sha256"])
    load_input_binding(value, root, input_id=input_id, model_input_sha256=model_input_sha256)
    record = normalize_record(value["record"], guide_version=guide_version)
    expected = primary_paths(record)
    require(isinstance(value["primary_paths"], list) and all(isinstance(p, str) for p in value["primary_paths"])
            and len(value["primary_paths"]) == len(set(value["primary_paths"]))
            and set(value["primary_paths"]) == set(expected), "Reference scoring mask differs from its fields")
    require(isinstance(value["conversion_log"], list), "Conversion log must be an array")
    for entry in value["conversion_log"]:
        object_keys(entry, {"field_path", "old_value", "new_value", "reason"}, "conversion_log")
        string(entry["field_path"], "conversion field path")
        string(entry["reason"], "conversion reason")
    require(all(context.citation_valid(c) for c in record_citations(record, include_enrichments=True)),
            "Reference evidence fails mechanical validation")
    if record["source_anchor"]:
        require(all(context.span_valid(s) for spans in record["source_anchor"].values() for s in spans),
                "Reference anchor leaves permitted input")
    return Reference(value["reference_id"], input_id, record, guide_version=guide_version)
