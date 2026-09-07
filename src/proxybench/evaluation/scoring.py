"""Score preserved responses against admitted references; never run an extractor."""

from dataclasses import dataclass
from proxybench.normalization.values import LEGACY_GUIDE, policy_version

from proxybench.evaluation.evidence import evidence_report
from proxybench.evaluation.matching import match_records, validate_references
from proxybench.evaluation.projection import canonical, compare_fields, coverage, project
from proxybench.evaluation.references import Reference
from proxybench.schemas.records import FIELD_TYPES, ValidationError, normalize_record, require, strict_json, validate_envelope


def parsed_response(raw, *, guide_version=LEGACY_GUIDE):
    policy_version(guide_version)
    errors, normalized = [], []
    try:
        parsed = strict_json(raw)
    except ValidationError as error:
        return None, None, [], [str(error)], False
    records = parsed.get("records") if isinstance(parsed, dict) else None
    count = len(records) if isinstance(records, list) else None
    try:
        validate_envelope(parsed)
    except (ValidationError, RecursionError) as error:
        return parsed, count, [], [str(error)], False
    for i, record in enumerate(records):
        try:
            normalized.append(normalize_record(record, guide_version=guide_version))
        except (ValidationError, RecursionError) as error:
            normalized.append(None)
            errors.append(f"Record {i}: {error}")
    return parsed, count, normalized, errors, True


def diagnostics(parsed, normalized, context):
    if not isinstance(parsed, dict) or not isinstance(parsed.get("records"), list):
        return {"assertion_counts_known": False, "records": []}
    return {"assertion_counts_known": True, "records": [
        evidence_report(record, context, valid_record=i < len(normalized) and normalized[i] is not None)
        for i, record in enumerate(parsed["records"])]}


def enrichment_diagnostics(reference, prediction):
    findings = []
    expected = {(e["field_path"], e["field"]["rule_id"]): e["field"] for e in reference["enrichments"]}
    actual = {(e["field_path"], e["field"]["rule_id"]): e["field"] for e in prediction["enrichments"]} if prediction else {}
    for path, rule in sorted(set(expected) | set(actual)):
        key = path.removeprefix("/fields/")
        kind = FIELD_TYPES.get(key)
        result = {"field_path": path, "rule_id": rule, "status": "NOT_ASSESSED"}
        # Nested array associations are deliberately not inferred from indices.
        if isinstance(kind, str) and (path, rule) in expected:
            result["status"] = ("MATCH" if (path, rule) in actual and project(expected[path, rule], kind)
                                == project(actual[path, rule], kind) else "ENRICHMENT_MISMATCH")
        findings.append(result)
    return {"findings": findings, "primary_score_includes_enrichments": False, "array_member_accuracy": "NOT_ASSESSED"}


def score_fragment(raw, reference, *, context=None, expected_anchor=None):
    require(isinstance(reference, Reference), "Supply an admitted reference or an explicitly synthetic reference")
    guide_version = reference.guide_version
    record = normalize_record(reference.record, guide_version=guide_version)
    parsed, count, normalized, errors, envelope_valid = parsed_response(raw, guide_version=guide_version)
    fields = {key: False for key in compare_fields(record, record)}
    prediction = normalized[0] if len(normalized) == 1 else None
    complete = envelope_valid and parsed["status"] == "COMPLETE"
    if complete and prediction is not None:
        fields = compare_fields(record, prediction)
    reasons = []
    if not envelope_valid or errors:
        reasons.append("MALFORMED_RESPONSE")
    if envelope_valid and not complete:
        reasons.append("ABSTENTION" if parsed["status"] == "ABSTAINED" else "INCOMPLETE_EXECUTION")
    if count == 0:
        reasons.append("MISSING_TARGET")
    if count is not None and count > 1:
        reasons.append("EXTRA_RECORD")
        # Administrative record IDs do not prevent a duplicate disclosure diagnosis.
        keys = [canonical({k: r[k] for k in ("source_anchor", "fields")}) for r in normalized if r is not None]
        if len(set(keys)) < len(keys):
            reasons.append("DUPLICATE_OUTPUT")
    if complete and prediction is not None and not all(fields.values()):
        reasons.append("FIELD_MISMATCH")
    metadata = []
    if isinstance(parsed, dict) and parsed.get("input_id") != reference.input_id:
        metadata.append("INPUT_ID_MISMATCH")
    anchor_errors = []
    if prediction is not None:
        anchor = prediction["source_anchor"]
        if anchor is None:
            anchor_errors.append("ANCHOR_ABSTENTION")
        elif context and not all(context.span_valid(s) for spans in anchor.values() for s in spans):
            anchor_errors.append("ANCHOR_OUTSIDE_INPUT")
        if expected_anchor is not None:
            _, findings = match_records([prediction], [expected_anchor])
            anchor_errors.extend(f["code"] for f in findings)
    return {"input_id": reference.input_id, "synthetic": reference.synthetic,
            "guide_version": guide_version, "scorer_version": "fragment-" + policy_version(guide_version),
            "scheduled": 1, "passed": complete and prediction is not None and all(fields.values()),
            "field_matches": fields, "included_fields": len(fields), "predicted_records": count,
            "coverage": coverage(record, prediction if complete else None),
            "reasons": reasons, "schema_errors": errors, "metadata_errors": metadata,
            "anchor_errors": anchor_errors, "evidence": diagnostics(parsed, normalized, context),
            "enrichments": enrichment_diagnostics(record, prediction)}


@dataclass(frozen=True)
class FilingReference:
    input_id: str
    references: tuple
    anchors: list
    context: object
    coverage_review: dict
    guide_version: str = LEGACY_GUIDE

    def validate(self):
        policy_version(self.guide_version)
        require(all(r.guide_version == self.guide_version for r in self.references), "Filing guide differs from references")
        review = self.coverage_review
        require(review.get("decision") == "COMPLETE_REFERENCE", "Partial filing references cannot supply recall")
        for key in ("reviewer", "reviewed_at", "reason", "version", "accession"):
            require(isinstance(review.get(key), str) and bool(review[key].strip()), f"Coverage review needs {key}")
        require(review.get("unresolved_coverage") == [], "Unresolved filing coverage prevents reference admission")
        inventory = review.get("documents")
        require(isinstance(inventory, list) and bool(inventory), "Filing document inventory is required")
        require(all(isinstance(d, dict) and isinstance(d.get("document_id"), str) for d in inventory), "Invalid document inventory")
        require(len({d["document_id"] for d in inventory}) == len(inventory), "Duplicate inventory document")
        eligible = {d["document_id"]: d for d in inventory if d.get("disposition") == "ELIGIBLE"}
        require(set(eligible) == set(self.context.documents), "Eligible document inventory differs from original sources")
        for document in inventory:
            require(document.get("disposition") in ("ELIGIBLE", "EXCLUDED", "DUPLICATE")
                    and bool(document.get("reason")), "Each document needs a reviewed disposition")
        for key, document in eligible.items():
            require(document.get("source_sha256") == self.context.hashes[key], "Reviewed source hash differs")
            length = len(self.context.documents[key])
            require(_covers(document.get("reviewed_ranges", []), length), "Review omitted original source ranges")
            require(_covers(self.context.permitted_ranges[key], length), "Complete-filing input omits original document ranges")
        require(all(isinstance(r, Reference) and r.input_id == self.input_id for r in self.references), "Filing reference input mismatch")
        ids = {r.reference_id for r in self.references}
        require(len(ids) == len(self.references), "Duplicate reference IDs")
        validate_references(self.anchors, self.context.span_valid)
        require(ids == {a["reference_id"] for a in self.anchors}, "Logical anchors differ from reference inventory")
        if not self.references:
            require(isinstance(review.get("empty_report_basis"), str) and bool(review["empty_report_basis"].strip()),
                    "Empty reference filing requires reviewed original-source basis")
        return self


def _covers(ranges, length):
    if not isinstance(ranges, (list, tuple)):
        return False
    if not all(isinstance(r, (list, tuple)) and len(r) == 2 and all(type(v) is int for v in r)
               and 0 <= r[0] < r[1] <= length for r in ranges):
        return False
    end = 0
    for start, stop in sorted(ranges):
        if start > end:
            return False
        end = max(end, stop)
    return end == length


def score_filing(raw, filing, *, processing_coverage=None):
    """Processing coverage must come from the runner, never the model response."""
    filing.validate()
    parsed, count, normalized, errors, envelope_valid = parsed_response(raw, guide_version=filing.guide_version)
    pairs, anchor_errors = match_records(normalized, filing.anchors) if envelope_valid else ([], [])
    by_id = {r.reference_id: normalize_record(r.record, guide_version=filing.guide_version) for r in filing.references}
    comparisons = {rid: compare_fields(by_id[rid], normalized[i]) for rid, i in pairs}
    correct = sum(all(result.values()) for result in comparisons.values())
    matched, total = len(pairs), len(filing.references)
    supplied = processing_coverage or {}
    coverage_complete = (supplied.get("unrecovered_truncation") is False
                         and isinstance(supplied.get("documents"), dict)
                         and set(supplied["documents"]) == set(filing.context.documents)
                         and all(_covers(supplied["documents"][key], len(raw_bytes))
                                 for key, raw_bytes in filing.context.documents.items()))
    complete = envelope_valid and parsed["status"] == "COMPLETE"
    exact = complete and not errors and correct == total and count == total and coverage_complete
    covered = {rid: coverage(by_id[rid], normalized[i]) for rid, i in pairs}
    answered = sum(value["answered"] for value in covered.values())
    recoverable = sum(coverage(r)["recoverable"] for r in by_id.values())
    metadata = [] if not isinstance(parsed, dict) or parsed.get("input_id") == filing.input_id else ["INPUT_ID_MISMATCH"]
    return {"input_id": filing.input_id, "reference_records": total,
            "guide_version": filing.guide_version, "scorer_version": "filing-" + policy_version(filing.guide_version), "predicted_records": count,
            "matched_records": matched, "correct_records": correct,
            "localization_precision": matched / count if count else None,
            "localization_recall": matched / total if total else None,
            "strict_precision": correct / count if count else None,
            "strict_recall": correct / total if total else None,
            "exact_filing_success": exact, "correct_empty": exact if not total else None,
            "complete_execution": complete and coverage_complete, "processing_coverage_complete": coverage_complete,
            "processing_coverage": processing_coverage, "pairs": pairs, "field_matches": comparisons,
            "coverage": {"answered": answered, "recoverable": recoverable, "rate": answered / recoverable if recoverable else None},
            "schema_errors": errors, "anchor_errors": anchor_errors, "metadata_errors": metadata,
            "evidence": diagnostics(parsed, normalized, filing.context)}


def summarize_fragments(results):
    results = list(results)
    fields = {}
    for result in results:
        for key, matched in result["field_matches"].items():
            counts = fields.setdefault(key, {"correct": 0, "eligible": 0})
            counts["correct"] += matched
            counts["eligible"] += 1
    for counts in fields.values():
        counts["accuracy"] = counts["correct"] / counts["eligible"]
    passed = sum(r["passed"] for r in results)
    return {"scheduled": len(results), "passed": passed,
            "whole_record_accuracy": passed / len(results) if results else None, "fields": fields}


def summarize_filings(results):
    results = list(results)
    countable = [r for r in results if r["predicted_records"] is not None]
    predicted = sum(r["predicted_records"] for r in countable)
    reference = sum(r["reference_records"] for r in results)
    matched = sum(r["matched_records"] for r in results)
    correct = sum(r["correct_records"] for r in results)
    successful = sum(r["exact_filing_success"] for r in results)
    return {"eligible_filings": len(results), "successful_filings": successful,
            "failed_filings": len(results) - successful,
            "complete_execution_filings": sum(r["complete_execution"] for r in results),
            "exact_filing_success_rate": successful / len(results) if results else None,
            "reference_records": reference, "countable_predictions": predicted,
            "matched_records": matched, "correct_records": correct,
            "precision_response_coverage": len(countable) / len(results) if results else None,
            "localization_precision_on_countable_responses": matched / predicted if predicted else None,
            "strict_precision_on_countable_responses": correct / predicted if predicted else None,
            "localization_recall": matched / reference if reference else None,
            "strict_recall": correct / reference if reference else None}
