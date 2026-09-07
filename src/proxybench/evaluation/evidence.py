"""Report mechanical evidence validity separately from source support."""

from dataclasses import dataclass, field
from copy import deepcopy
from decimal import Decimal
import hashlib
import json

from proxybench.schemas.records import FIELD_TYPES, ValidationError, normalize_record, require, validate_span


@dataclass
class SourceContext:
    """Trusted bytes, allowed ranges, and reviewed prepared-text block mappings."""

    documents: dict
    permitted_ranges: dict
    encodings: dict = field(default_factory=dict)
    views: dict = field(default_factory=dict)

    def __post_init__(self):
        require(set(self.documents) == set(self.permitted_ranges), "Every supplied document needs explicit permitted ranges")
        self.hashes = {key: hashlib.sha256(raw).hexdigest() for key, raw in self.documents.items()}
        for key, ranges in self.permitted_ranges.items():
            require(isinstance(self.documents[key], bytes), "Source documents must be preserved bytes")
            for start, end in ranges:
                require(type(start) is int and type(end) is int and 0 <= start < end <= len(self.documents[key]),
                        "Invalid permitted input range")

    def span_valid(self, span):
        key = span["document_id"]
        return (key in self.documents and span["source_sha256"] == self.hashes[key]
                and any(a <= span["start_byte"] < span["end_byte"] <= b for a, b in self.permitted_ranges[key]))

    def citation_valid(self, evidence):
        try:
            validate_span(evidence, evidence=True)
            if not self.span_valid(evidence):
                return False
            basis = evidence["quote_basis"]
            if basis == "UNREADABLE_REGION":
                return True
            if basis == "ORIGINAL_DECODED":
                raw = self.documents[evidence["document_id"]][evidence["start_byte"]:evidence["end_byte"]]
                return evidence["quote"] in raw.decode(self.encodings.get(evidence["document_id"], "utf-8"), errors="strict")
            block = self.views.get((evidence["view_id"], evidence["block_id"]))
            # Each mapping ties this exact prepared block text to a source range.
            return bool(block and all(evidence[k] == block["span"][k] for k in ("document_id", "source_sha256", "start_byte", "end_byte"))
                        and evidence["quote"] in block["text"]
                        and self.span_valid(block["span"]))
        except (ValidationError, KeyError, TypeError, UnicodeError):
            return False


def assertions(record):
    """Retain recognized scalar claims even from parseable malformed records."""
    found = []

    def walk(wrapper, kind, path):
        if not isinstance(wrapper, dict) or "value" not in wrapper:
            return
        value = wrapper["value"]
        if isinstance(kind, str):
            if value is not None:
                found.append({"path": path, "value": json.loads(json.dumps(value, default=str)),
                              "json_type": "number" if isinstance(value, Decimal) else type(value).__name__,
                              "unassessable": isinstance(value, (dict, list)),
                              "origin": diagnostic_value(wrapper.get("origin")),
                              "evidence": diagnostic_value(wrapper.get("evidence")),
                              "assessment": "NOT_ASSESSED"})
        elif isinstance(kind, dict) and isinstance(value, dict):
            for key, child in kind.items():
                walk(value.get(key), child, path + "/value/" + key)
        elif isinstance(kind, list) and isinstance(value, list):
            for i, member in enumerate(value):
                if isinstance(kind[0], dict) and isinstance(member, dict):
                    for key, child in kind[0].items():
                        walk(member.get(key), child, f"{path}/value/{i}/{key}")
                elif not isinstance(kind[0], dict):
                    walk(member, kind[0], f"{path}/value/{i}")

    if isinstance(record, dict) and isinstance(record.get("fields"), dict):
        for key, kind in FIELD_TYPES.items():
            walk(record["fields"].get(key), kind, "/fields/" + key)
    return found


def record_citations(record, *, include_enrichments=False):
    citations = []

    def scan(value):
        if isinstance(value, dict):
            if isinstance(value.get("evidence"), list):
                citations.extend(value["evidence"])
            for key, child in value.items():
                if key != "evidence" and (include_enrichments or key != "enrichments"):
                    scan(child)
        elif isinstance(value, list):
            for child in value:
                scan(child)

    scan(record)
    return citations


def diagnostic_value(value):
    """Represent malformed decimal metadata without losing its numeric type."""
    if isinstance(value, Decimal):
        return {"json_type": "number", "value": str(value)}
    if isinstance(value, dict):
        return {key: diagnostic_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [diagnostic_value(child) for child in value]
    return value


def evidence_report(record, context, *, valid_record=False):
    claims = assertions(record)
    d2 = []
    if valid_record or _valid_d2_fields(record):
        d2 = [claim for claim in claims if claim["path"] == "/fields/participation" and claim["origin"] == "DERIVED"]
        claims = [claim for claim in claims if claim not in d2]
    citations = record_citations(record)
    valid = sum(context.citation_valid(c) for c in citations) if context else None
    # Invalid availability or origin flags cannot hide core assertion slots.
    cited = sum(isinstance(c["evidence"], list) and bool(c["evidence"]) for c in claims)
    return {"assertions": claims, "d2_assertions": d2,
            "semantic_support": {"SUPPORTED": 0, "UNSUPPORTED": 0, "NOT_ASSESSED": len(claims),
                                 "unsupported_rate": None, "assessed_fraction": 0 if claims else None},
            "citations": {"submitted": len(citations), "valid": valid,
                          "validity": valid / len(citations) if citations and valid is not None else None,
                          "asserted_core_values": len(claims), "cited_values": cited,
                          "missing_required": len(claims) - cited,
                          "coverage": cited / len(claims) if claims else None}}


def _valid_d2_fields(record):
    """An unrelated malformed field does not change valid D2 assertion units."""
    if not isinstance(record, dict) or not isinstance(record.get("fields"), dict):
        return False
    fields = record["fields"]
    if not all(key in fields for key in ("participation", "vote_components")):
        return False
    missing = {"value": None, "raw_text": None, "availability": "ABSENT_IN_CONTEXT", "origin": None,
               "evidence": [], "reason": None, "rule_id": None}
    subset = {key: fields[key] if key in ("participation", "vote_components") else missing for key in FIELD_TYPES}
    try:
        normalize_record({"record_id": "diagnostic", "source_anchor": None, "fields": subset, "enrichments": []})
    except (ValidationError, RecursionError):
        return False
    return True


def apply_semantic_reviews(diagnostics, reviews):
    """Attach explicit source-review decisions to a new diagnostic version."""
    result = deepcopy(diagnostics)
    seen = set()
    for review in reviews:
        index, path = review["record_index"], review["field_path"]
        require(type(index) is int and 0 <= index < len(result["records"]), "Semantic review names an unknown record")
        require(isinstance(path, str), "Semantic review field path must be a string")
        require((index, path) not in seen, "Repeated semantic review decision")
        seen.add((index, path))
        claims = [c for c in result["records"][index]["assertions"] if c["path"] == path]
        require(len(claims) == 1 and not claims[0]["unassessable"], "Semantic review needs one assessable scalar assertion")
        require(review["assessment"] in ("SUPPORTED", "UNSUPPORTED"), "Unknown semantic assessment")
        for key in ("reviewer", "reviewed_at", "reason"):
            require(isinstance(review.get(key), str) and bool(review[key].strip()), "Semantic review needs reviewer, time, and source-based reason")
        claims[0]["assessment"] = review["assessment"]
        claims[0]["review"] = deepcopy(review)
    for record in result["records"]:
        counts = {key: sum(c["assessment"] == key for c in record["assertions"])
                  for key in ("SUPPORTED", "UNSUPPORTED", "NOT_ASSESSED")}
        assessed = counts["SUPPORTED"] + counts["UNSUPPORTED"]
        total = sum(counts.values())
        record["semantic_support"] = {**counts, "unsupported_rate": counts["UNSUPPORTED"] / assessed if assessed else None,
                                      "assessed_fraction": assessed / total if total else None}
    return result


def summarize_evidence(reports):
    reports = list(reports)
    counts = {key: sum(record["semantic_support"][key] for report in reports for record in report["records"])
              for key in ("SUPPORTED", "UNSUPPORTED", "NOT_ASSESSED")}
    assessed, total = counts["SUPPORTED"] + counts["UNSUPPORTED"], sum(counts.values())
    return {**counts, "unsupported_rate": counts["UNSUPPORTED"] / assessed if assessed else None,
            "assessed_fraction_of_known_assertions": assessed / total if total else None,
            "response_count": len(reports), "responses_with_unknown_assertion_counts": sum(not r["assertion_counts_known"] for r in reports)}
