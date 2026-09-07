"""Validate benchmark-v1 records without repairing or changing raw predictions."""

from copy import deepcopy
import json
import re

from proxybench.normalization.values import ENUMS, scalar, LEGACY_GUIDE, policy_version


SCHEMA_VERSION = "benchmark-v1"
D2_RULE = "pilot-decisions-2026-09-07:D2"
AVAILABILITY = {"PRESENT", "ABSENT_IN_CONTEXT", "AMBIGUOUS", "UNREADABLE", "CONFLICTING", "NOT_APPLICABLE"}
ORIGINS = {"EXTRACTED", "DERIVED", "INFERRED"}
WRAPPER_KEYS = {"value", "raw_text", "availability", "origin", "evidence", "reason", "rule_id"}
IDENTIFIER = {"source_label": "text", "value": "identifier"}
QUANTITY = {"amount": "amount", "unit": "text"}
COMPONENT = {"direction": "direction", "quantity": QUANTITY,
             "disclosed_management_alignment": "disclosed_management_alignment"}
SCOPE = {"name": "name", "scope_type": "scope_type", "members": ["name"]}
FIELD_TYPES = {
    "reporting_scope": SCOPE, "series_identifiers": [IDENTIFIER], "issuer_name": "name",
    "security_identifiers": [IDENTIFIER], "ticker": "identifier", "meeting_date": "date",
    "meeting_type": "text", "proposal_number": "identifier", "raw_description": "text",
    "separate_subject": "text", "proposal_source": "text", "participation": "participation",
    "vote_components": [COMPONENT], "management_recommendation": "management_recommendation",
}
SPAN_KEYS = {"document_id", "source_sha256", "start_byte", "end_byte"}
EVIDENCE_KEYS = SPAN_KEYS | {"view_id", "block_id", "quote", "quote_basis"}


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def object_keys(value, keys, path):
    require(isinstance(value, dict) and set(value) == set(keys), f"{path}: required object keys differ")


def string(value, path, nullable=False):
    require((nullable and value is None) or (isinstance(value, str) and bool(value.strip())),
            f"{path}: expected nonempty string" + (" or null" if nullable else ""))


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValidationError(f"Nonstandard JSON number: {value}")

    def floating(value):
        # Decimal avoids accepting JSON exponent overflow as Python infinity.
        from decimal import Decimal
        return Decimal(value)

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant, parse_float=floating)
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise ValidationError(f"Malformed JSON: {error}") from error


def validate_span(span, path="span", evidence=False):
    object_keys(span, EVIDENCE_KEYS if evidence else SPAN_KEYS, path)
    string(span["document_id"], path + "/document_id")
    require(isinstance(span["source_sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", span["source_sha256"]),
            f"{path}: expected SHA-256 hex string")
    require(type(span["start_byte"]) is int and type(span["end_byte"]) is int
            and 0 <= span["start_byte"] < span["end_byte"], f"{path}: invalid byte range")
    if evidence:
        for key in ("view_id", "block_id"):
            string(span[key], path + "/" + key, nullable=True)
        require(span["quote_basis"] in ("ORIGINAL_DECODED", "PREPARED_VIEW", "UNREADABLE_REGION"),
                f"{path}: unknown quotation basis")
        if span["quote_basis"] == "UNREADABLE_REGION":
            require(span["quote"] is None, f"{path}: unreadable evidence has no quotation")
        else:
            string(span["quote"], path + "/quote")
        if span["quote_basis"] == "PREPARED_VIEW":
            require(span["view_id"] is not None and span["block_id"] is not None,
                    f"{path}: prepared quotation needs view and block IDs")


def validate_anchor(anchor):
    if anchor is None:
        return
    object_keys(anchor, {"subject_spans", "scope_spans"}, "source_anchor")
    for key, spans in anchor.items():
        require(isinstance(spans, list), f"source_anchor/{key}: expected array")
        for span in spans:
            validate_span(span)
    require(bool(anchor["subject_spans"]), "source_anchor: subject spans are required")


def normalize_field(field, kind, path, *, core=True, registry=None, guide_version=LEGACY_GUIDE):
    object_keys(field, WRAPPER_KEYS, path)
    if registry is not None:
        registry[path] = (field, kind)
    for key in ("raw_text", "reason", "rule_id"):
        string(field[key], path + "/" + key, nullable=True)
    require(field["availability"] in tuple(AVAILABILITY), f"{path}: invalid availability")
    require(field["origin"] is None or field["origin"] in tuple(ORIGINS), f"{path}: invalid origin")
    require(isinstance(field["evidence"], list), f"{path}: evidence must be an array")
    for evidence in field["evidence"]:
        validate_span(evidence, path + "/evidence", evidence=True)
        if evidence["quote_basis"] == "UNREADABLE_REGION":
            require(field["reason"] is not None, f"{path}: unreadable evidence requires explanation")
    availability, origin, value = field["availability"], field["origin"], field["value"]
    special_empty = (path == "/fields/vote_components" and availability == "NOT_APPLICABLE"
                     and origin == "DERIVED" and value == [] and isinstance(value, list))
    if availability != "PRESENT":
        require(special_empty or (value is None and origin is None), f"{path}: unresolved value must be null")
        if availability != "ABSENT_IN_CONTEXT":
            require(bool(field["evidence"]) and field["reason"] is not None, f"{path}: missing evidence or reason")
        if special_empty:
            require(field["rule_id"] == D2_RULE, f"{path}: D2 rule ID required")
        return field
    require(value is not None and origin in tuple(ORIGINS) and bool(field["evidence"]),
            f"{path}: present value needs origin and evidence")
    if origin in ("DERIVED", "INFERRED"):
        require(field["rule_id"] is not None, f"{path}: rule ID required")
        if core:
            require(path == "/fields/participation" and origin == "DERIVED"
                    and field["rule_id"] == D2_RULE, f"{path}: invalid core origin")
    if isinstance(kind, str):
        try:
            field["value"] = scalar(value, kind, guide_version=guide_version)
        except ValueError as error:
            raise ValidationError(f"{path}: {error}") from error
        if kind in ENUMS and field["value"] == "OTHER":
            require(field["raw_text"] is not None, f"{path}: OTHER needs source wording")
        if core and origin == "DERIVED":
            require(field["value"] == "VOTED", f"{path}: only VOTED permits derived participation")
    elif isinstance(kind, dict):
        object_keys(value, kind, path + "/value")
        for key, child_kind in kind.items():
            normalize_field(value[key], child_kind, path + "/value/" + key, core=core, registry=registry, guide_version=guide_version)
    else:
        require(isinstance(value, list) and bool(value), f"{path}: present arrays must be nonempty")
        child_kind = kind[0]
        for i, child in enumerate(value):
            child_path = f"{path}/value/{i}"
            if isinstance(child_kind, dict):
                object_keys(child, child_kind, child_path)
                for key, member_kind in child_kind.items():
                    normalize_field(child[key], member_kind, child_path + "/" + key, core=core, registry=registry, guide_version=guide_version)
            else:
                normalize_field(child, child_kind, child_path, core=core, registry=registry, guide_version=guide_version)
    return field


def normalize_record(record, *, guide_version=LEGACY_GUIDE):
    policy_version(guide_version)
    record = deepcopy(record)
    object_keys(record, {"record_id", "source_anchor", "fields", "enrichments"}, "record")
    string(record["record_id"], "record_id")
    validate_anchor(record["source_anchor"])
    object_keys(record["fields"], FIELD_TYPES, "fields")
    registry = {}
    for key, kind in FIELD_TYPES.items():
        normalize_field(record["fields"][key], kind, "/fields/" + key, registry=registry, guide_version=guide_version)
    fields = record["fields"]
    participation, components = fields["participation"], fields["vote_components"]
    did_not_vote = participation["value"] == "DID_NOT_VOTE"
    empty = components["availability"] == "NOT_APPLICABLE" and components["value"] == []
    require(did_not_vote == empty, "D2: DID_NOT_VOTE requires the special empty components, and conversely")
    if participation["origin"] == "DERIVED":
        require(components["availability"] == "PRESENT" and any(
            c["direction"]["value"] in ("FOR", "AGAINST", "WITHHOLD", "ABSTAIN") or
            (c["direction"]["availability"] == "PRESENT" and c["quantity"]["availability"] == "PRESENT"
             and c["quantity"]["value"]["amount"]["availability"] == "PRESENT")
            for c in components["value"]), "D2: derived participation requires a disclosed voting basis")
    if components["availability"] == "PRESENT":
        for component in components["value"]:
            quantity = component["quantity"]
            if quantity["availability"] == "PRESENT":
                amount = quantity["value"]["amount"]
                require(amount["availability"] != "PRESENT" or amount["value"] != "0",
                        "D2: zero quantities are not cast-vote components")
    require(isinstance(record["enrichments"], list), "enrichments: expected array")
    seen = set()
    for enrichment in record["enrichments"]:
        object_keys(enrichment, {"field_path", "field"}, "enrichment")
        path = enrichment["field_path"]
        require(isinstance(path, str) and path in registry, "enrichment: path must identify an existing field wrapper")
        field = normalize_field(enrichment["field"], registry[path][1], path, core=False, guide_version=guide_version)
        require(field["availability"] == "PRESENT" and field["origin"] in ("DERIVED", "INFERRED"),
                "enrichment: expected a derived or inferred value")
        pair = (path, field["rule_id"])
        require(pair not in seen, "enrichment: duplicate path and rule")
        seen.add(pair)
        # A container enrichment cannot introduce absent structure or change membership.
        if not isinstance(registry[path][1], str):
            require(_membership(field) == _membership(registry[path][0]),
                    "enrichment: container structure and membership must remain unchanged")
    return record


def _membership(field):
    value = field["value"]
    if isinstance(value, list):
        return [(_membership(v) if "availability" in v else {k: _membership(f) for k, f in v.items()}) for v in value]
    if isinstance(value, dict):
        return {k: _membership(v) for k, v in value.items()}
    return None


def validate_envelope(response):
    object_keys(response, {"schema_version", "input_id", "status", "records", "failure"}, "response")
    require(response["schema_version"] == SCHEMA_VERSION, "Unknown schema version")
    string(response["input_id"], "input_id")
    require(response["status"] in ("COMPLETE", "ABSTAINED", "FAILED"), "Unknown response status")
    require(isinstance(response["records"], list), "records: expected array")
    failure = response["failure"]
    if response["status"] == "COMPLETE":
        require(failure is None, "COMPLETE response requires null failure")
    else:
        object_keys(failure, {"code", "message", "stage", "truncated"}, "failure")
        for key in ("code", "message", "stage"):
            string(failure[key], "failure/" + key)
        require(type(failure["truncated"]) is bool, "failure/truncated: expected boolean")
        if response["status"] == "ABSTAINED":
            require(not response["records"], "ABSTAINED response cannot contain records")
    return response


def validate_source_audit(audit):
    object_keys(audit, {"audit_id", "field_path", "document_ids", "inspected_spans", "complete_scope",
                        "reviewer", "reviewed_at", "finding", "reason"}, "source_audit")
    for key in ("audit_id", "field_path", "reviewer", "reviewed_at", "reason"):
        string(audit[key], "source_audit/" + key)
    require(audit["field_path"].startswith("/fields/"), "Source audit must name a field path")
    require(type(audit["complete_scope"]) is bool, "complete_scope must be boolean")
    require(audit["finding"] in ("FOUND", "NOT_DISCLOSED", "UNRESOLVED"), "Unknown source audit finding")
    require(isinstance(audit["document_ids"], list) and bool(audit["document_ids"]), "Source audit needs an inspected document inventory")
    for document_id in audit["document_ids"]:
        string(document_id, "source audit document ID")
    require(isinstance(audit["inspected_spans"], list) and bool(audit["inspected_spans"]), "Source audit needs inspected evidence")
    for evidence in audit["inspected_spans"]:
        validate_span(evidence, evidence=True)
        require(evidence["document_id"] in audit["document_ids"], "Audit evidence names an unlisted document")
    return audit
