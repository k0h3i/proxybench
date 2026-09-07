"""Synthetic acceptance cases; these tests do not score historical model output."""

from copy import deepcopy
import hashlib
import itertools
import json
import unittest

from proxybench.evaluation.evidence import SourceContext, apply_semantic_reviews, assertions, evidence_report, summarize_evidence
from proxybench.evaluation.matching import match_records, validate_references
from proxybench.evaluation.projection import coverage, maximum_assignment
from proxybench.evaluation.references import synthetic_reference
from proxybench.evaluation.scoring import FilingReference, score_filing, score_fragment, summarize_filings, summarize_fragments
from proxybench.normalization.values import scalar
from proxybench.schemas.records import D2_RULE, FIELD_TYPES, SCHEMA_VERSION, ValidationError, normalize_record, strict_json, validate_source_audit


RAW = b"Synthetic source " * 50
HASH = hashlib.sha256(RAW).hexdigest()


def span(start=0, end=10, document="doc", digest=HASH):
    return {"document_id": document, "source_sha256": digest, "start_byte": start, "end_byte": end}


def evidence():
    return {**span(0, len(RAW)), "view_id": None, "block_id": None,
            "quote": "Synthetic", "quote_basis": "ORIGINAL_DECODED"}


def field(value=None, *, availability=None, origin=None, raw_text=None, reason=None, rule_id=None):
    present = value is not None
    return {"value": value, "raw_text": raw_text, "availability": availability or ("PRESENT" if present else "ABSENT_IN_CONTEXT"),
            "origin": origin or ("EXTRACTED" if present else None), "evidence": [evidence()], "reason": reason, "rule_id": rule_id}


def unresolved(status="AMBIGUOUS"):
    return field(availability=status, reason="Synthetic unresolved disclosure.")


def component(direction="FOR", amount=None):
    return {"direction": field(direction), "quantity": field() if amount is None else field({"amount": field(amount), "unit": field("shares")}),
            "disclosed_management_alignment": field()}


def record():
    fields = {key: field() for key in FIELD_TYPES}
    fields.update(issuer_name=field("Synthetic Issuer"), proposal_number=field("01"), raw_description=field("Elect the collective slate."),
                  participation=field("VOTED", origin="DERIVED", rule_id=D2_RULE), vote_components=field([component()]))
    return {"record_id": "synthetic-1", "source_anchor": {"subject_spans": [span(10, 20)], "scope_spans": []},
            "fields": fields, "enrichments": []}


def response(records=None, *, status="COMPLETE", failure=None):
    return {"schema_version": SCHEMA_VERSION, "input_id": "synthetic-input", "status": status,
            "records": [record()] if records is None else records, "failure": failure}


def score(prediction, reference=None, **kwargs):
    raw = prediction if isinstance(prediction, str) else json.dumps(prediction)
    return score_fragment(raw, synthetic_reference(reference or record()), **kwargs)


def context():
    return SourceContext({"doc": RAW}, {"doc": [(0, len(RAW))]})


def anchor_ref(rid="r1", start=10, end=20, zone=(0, 100), scope=None):
    return {"reference_id": rid, "alternatives": [{"subject_spans": [span(start, end)], "subject_zones": [span(*zone)],
              "scope_spans": [span(*scope)] if scope else [], "scope_zones": [span(*scope)] if scope else []}]}


def filing(records=None, anchors=None):
    records = [record()] if records is None else records
    anchors = [anchor_ref()] if anchors is None else anchors
    refs = tuple(synthetic_reference(r, reference_id=a["reference_id"]) for r, a in zip(records, anchors))
    review = {"decision": "COMPLETE_REFERENCE", "reviewer": "Synthetic reviewer", "reviewed_at": "2026-09-07",
              "reason": "Synthetic complete source review", "version": "synthetic-v1", "accession": "synthetic",
              "unresolved_coverage": [], "empty_report_basis": "Synthetic empty original",
              "documents": [{"document_id": "doc", "source_sha256": HASH, "disposition": "ELIGIBLE",
                             "reason": "Synthetic input", "reviewed_ranges": [[0, len(RAW)]]}]}
    return FilingReference("synthetic-input", refs, anchors, context(), review)


PROCESSING = {"documents": {"doc": [[0, len(RAW)]]}, "unrecovered_truncation": False}
FAILURE = {"code": "TIMEOUT", "message": "Synthetic timeout", "stage": "extraction", "truncated": True}


class NormalizationTests(unittest.TestCase):
    def test_finite_dates(self):
        for value, expected in [("05/15/2007", "2007-05-15"), ("15/05/2007", "2007-05-15"),
                                ("2/2/2014", "2014-02-02"), ("FEB 29, 2024", "2024-02-29"),
                                ("29-feb-2024", "2024-02-29"), ("0001-01-01", "0001-01-01")]:
            self.assertEqual(scalar(value, "date"), expected)
        for value in ["6/10/2014", "2014-02-29", "2024-2-01", "2024/01/01", "Jan 1 2024", "0000-01-01", "1-Foo-2024"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                scalar(value, "date")

    def test_string_and_quantity_distinctions(self):
        self.assertEqual(scalar("5,250,000.00", "amount"), "5250000")
        for invalid in [5250000, "0.0.0", "1,00", "-1", "1e3", "01", " 1 "]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                scalar(invalid, "amount")
        self.assertEqual(scalar("  01 A\tB ", "identifier"), "01 A\tB")
        self.assertEqual(scalar(" E\u0301COLE  Fund ", "name"), "école fund")
        self.assertEqual(scalar("&amp;", "text"), "&amp;")
        self.assertEqual(scalar("for", "disclosed_management_alignment"), "WITH_MANAGEMENT")
        with self.assertRaises(ValueError):
            scalar("Take No Action", "participation")

    def test_strict_json(self):
        for raw in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{} trailing', '```json\n{}\n```', '{']:
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                strict_json(raw)


class FragmentTests(unittest.TestCase):
    def test_correct_missing_extra_duplicate_and_incomplete(self):
        self.assertTrue(score(response())["passed"])
        for submitted in [response([]), response([record(), record()]), response(status="FAILED", failure=FAILURE),
                          response([], status="ABSTAINED", failure=FAILURE), "refusal", "{"]:
            with self.subTest(submitted=submitted):
                result = score(submitted)
                self.assertFalse(result["passed"])
                self.assertFalse(any(result["field_matches"].values()))
                self.assertEqual(result["coverage"]["answered"], 0)
        self.assertIn("DUPLICATE_OUTPUT", score(response([record(), record()]))["reasons"])

    def test_recoverable_abstention_and_supported_uncertainty(self):
        r = record()
        r["fields"]["issuer_name"] = field()
        self.assertFalse(score(response([r]))["passed"])
        for key, status in [("reporting_scope", "AMBIGUOUS"), ("proposal_source", "CONFLICTING"), ("meeting_date", "AMBIGUOUS")]:
            ref = record()
            ref["fields"][key] = unresolved(status)
            self.assertTrue(score(response([ref]), ref)["passed"])
            pred = deepcopy(ref)
            pred["fields"][key] = field("2014-06-10" if key == "meeting_date" else "guessed")
            self.assertFalse(score(response([pred]), ref)["passed"])

    def test_enrichment_cannot_replace_extraction_or_mask_fields(self):
        ref = record()
        ref["fields"]["meeting_date"] = unresolved()
        pred = deepcopy(ref)
        pred["enrichments"] = [{"field_path": "/fields/meeting_date", "field": field("2014-06-10", origin="INFERRED", rule_id="assumption")}]
        self.assertTrue(score(response([pred]), ref)["passed"])
        pred["fields"]["meeting_date"] = pred["enrichments"][0]["field"]
        self.assertFalse(score(response([pred]), ref)["passed"])
        pred = record()
        pred["fields"]["vote_components"]["origin"] = "DERIVED"
        pred["fields"]["vote_components"]["rule_id"] = "mask"
        self.assertFalse(score(response([pred]))["passed"])
        with self.assertRaises(ValidationError):
            synthetic_reference(pred)

    def test_enrichment_errors_and_literal_other_text(self):
        ref = record()
        ref["fields"]["ticker"] = field("OTHER")
        ref["fields"]["meeting_date"] = unresolved()
        ref["enrichments"] = [{"field_path": "/fields/meeting_date", "field": field("2014-06-10", origin="INFERRED", rule_id="pilot")}]
        pred = deepcopy(ref)
        pred["enrichments"][0]["field"]["value"] = "2014-10-06"
        result = score(response([pred]), ref)
        self.assertTrue(result["passed"])
        self.assertEqual(result["enrichments"]["findings"][0]["status"], "ENRICHMENT_MISMATCH")

    def test_d2_direction_distinctions_empty_votes_and_zero_components(self):
        for direction in ("WITHHOLD", "ABSTAIN"):
            ref = record()
            ref["fields"]["vote_components"]["value"][0]["direction"] = field(direction)
            pred = deepcopy(ref)
            pred["fields"]["vote_components"]["value"][0]["direction"] = field("AGAINST")
            self.assertFalse(score(response([pred]), ref)["passed"])
        ref = record()
        ref["fields"]["participation"] = field("DID_NOT_VOTE")
        ref["fields"]["vote_components"] = field([], availability="NOT_APPLICABLE", origin="DERIVED", reason="Explicit nonparticipation", rule_id=D2_RULE)
        self.assertTrue(score(response([ref]), ref)["passed"])
        pred = deepcopy(ref)
        pred["fields"]["vote_components"] = field([component()])
        self.assertFalse(score(response([pred]), ref)["passed"])
        ref = record()
        ref["fields"]["vote_components"] = field([component(amount="5250000")])
        pred = deepcopy(ref)
        pred["fields"]["vote_components"]["value"].append(component("AGAINST", "0"))
        self.assertFalse(score(response([pred]), ref)["passed"])

    def test_components_multisets_and_no_invented_quantity(self):
        ref = record()
        ref["fields"]["vote_components"] = field([component("FOR"), component("WITHHOLD")])
        pred = deepcopy(ref)
        pred["fields"]["vote_components"]["value"].reverse()
        self.assertTrue(score(response([pred]), ref)["passed"])
        pred["fields"]["vote_components"]["value"].append(component())
        self.assertFalse(score(response([pred]), ref)["passed"])
        pred = deepcopy(ref)
        pred["fields"]["vote_components"]["value"][0] = component("FOR", "1")
        self.assertFalse(score(response([pred]), ref)["passed"])

    def test_explicit_none_identifiers_and_case(self):
        ref = record()
        ref["fields"]["management_recommendation"] = field("NONE")
        pred = deepcopy(ref)
        pred["fields"]["management_recommendation"] = field()
        self.assertFalse(score(response([pred]), ref)["passed"])
        for value in ("1", 1):
            pred = record()
            pred["fields"]["proposal_number"]["value"] = value
            self.assertFalse(score(response([pred]))["passed"])
        pred = record()
        pred["fields"]["issuer_name"]["value"] = " SYNTHETIC   ISSUER "
        self.assertTrue(score(response([pred]))["passed"])
        pred["fields"]["raw_description"]["value"] = "elect the collective slate."
        self.assertFalse(score(response([pred]))["passed"])

    def test_partial_nested_records_and_array_coverage(self):
        ref = record()
        ref["fields"]["security_identifiers"] = field([
            {"source_label": field("CUSIP9"), "value": field("001234567")},
            {"source_label": field("ISIN"), "value": unresolved("UNREADABLE")}])
        ref["fields"]["reporting_scope"] = field({"name": unresolved("UNREADABLE"),
            "scope_type": field("FUND_GROUP"), "members": field([field("Fund A"), field("Fund B")])})
        ref["fields"]["vote_components"]["value"][0]["quantity"] = field({"amount": unresolved("UNREADABLE"), "unit": field("votes")})
        pred = deepcopy(ref)
        pred["fields"]["security_identifiers"]["value"].reverse()
        pred["fields"]["reporting_scope"]["value"]["members"]["value"].reverse()
        result = score(response([pred]), ref)
        self.assertTrue(result["passed"])
        self.assertEqual(result["coverage"]["rate"], 1)
        pred["fields"]["reporting_scope"]["value"]["members"]["value"].append(field("Fund A"))
        self.assertFalse(score(response([pred]), ref)["passed"])

    def test_other_and_date_wire_values(self):
        ref = record()
        ref["fields"]["management_recommendation"] = field("OTHER", raw_text="Special choice")
        for wording, passes in [(None, False), (" ", False), ("Other choice", False), ("Special   choice", True)]:
            pred = deepcopy(ref)
            pred["fields"]["management_recommendation"]["raw_text"] = wording
            self.assertEqual(score(response([pred]), ref)["passed"], passes)
        ref["fields"]["meeting_date"] = field("2007-05-15")
        pred = deepcopy(ref)
        pred["fields"]["meeting_date"]["value"] = "05/15/2007"
        self.assertTrue(score(response([pred]), ref)["passed"])
        pred["fields"]["meeting_date"]["value"] = "6/10/2014"
        self.assertFalse(score(response([pred]), ref)["passed"])

    def test_anchor_and_evidence_remain_separate_from_values(self):
        pred = record()
        pred["source_anchor"] = None
        result = score(response([pred]), context=context())
        self.assertTrue(result["passed"])
        self.assertIn("ANCHOR_ABSTENTION", result["anchor_errors"])
        pred["fields"]["issuer_name"]["evidence"][0]["quote"] = "not in source"
        result = score(response([pred]), context=context())
        self.assertTrue(result["passed"])
        self.assertLess(result["evidence"]["records"][0]["citations"]["validity"], 1)
        self.assertIsNone(result["evidence"]["records"][0]["semantic_support"]["unsupported_rate"])
        pred["fields"]["issuer_name"]["evidence"][0]["start_byte"] = True
        self.assertFalse(score(response([pred]))["passed"])

    def test_metadata_and_malformed_fields(self):
        pred = response()
        pred["input_id"] = "another-input"
        result = score(pred)
        self.assertTrue(result["passed"])
        self.assertEqual(result["metadata_errors"], ["INPUT_ID_MISMATCH"])
        for mutate in [lambda r: r.pop("source_anchor"), lambda r: r.update(extra=1),
                       lambda r: r["fields"]["issuer_name"].update(availability=[]),
                       lambda r: r["fields"]["ticker"].update(value=False, availability="PRESENT", origin="EXTRACTED")]:
            pred = record()
            mutate(pred)
            self.assertFalse(score(response([pred]))["passed"])

    def test_report_denominators_include_failures(self):
        summary = summarize_fragments([score(response()), score(response([])), score("{")])
        self.assertEqual(summary["whole_record_accuracy"], 1 / 3)
        self.assertEqual(summary["fields"]["issuer_name"]["eligible"], 3)
        self.assertNotIn("participation", summary["fields"])


class FilingTests(unittest.TestCase):
    def test_complete_and_missing_processing_coverage(self):
        f = filing()
        good = score_filing(json.dumps(response()), f, processing_coverage=PROCESSING)
        self.assertTrue(good["exact_filing_success"])
        partial = score_filing(json.dumps(response()), f)
        self.assertEqual(partial["strict_recall"], 1)
        self.assertFalse(partial["exact_filing_success"])

    def test_invalid_envelopes_and_partial_record_recovery(self):
        for submitted in [response(status="ABSTAINED", failure=FAILURE), response(status="UNKNOWN"),
                          {**response(), "schema_version": "unknown"}, response(status="FAILED")]:
            result = score_filing(json.dumps(submitted), filing(), processing_coverage=PROCESSING)
            self.assertEqual(result["predicted_records"], 1)
            self.assertEqual(result["matched_records"], 0)
        for status in ("COMPLETE", "FAILED"):
            submitted = response([record(), None], status=status, failure=FAILURE if status == "FAILED" else None)
            result = score_filing(json.dumps(submitted), filing(), processing_coverage=PROCESSING)
            self.assertEqual(result["predicted_records"], 2)
            self.assertEqual(result["matched_records"], 1)
            self.assertEqual(result["strict_precision"], 0.5)
            self.assertFalse(result["exact_filing_success"])
        for raw in ['{', json.dumps({**response(), "records": {}})]:
            result = score_filing(raw, filing())
            self.assertIsNone(result["predicted_records"])
            self.assertEqual(result["strict_recall"], 0)

    def test_duplicate_order_cannot_select_correct_values(self):
        wrong = record()
        wrong["fields"]["issuer_name"] = field("Wrong fund or issuer")
        result = score_filing(json.dumps(response([wrong, record()])), filing(), processing_coverage=PROCESSING)
        self.assertEqual(result["matched_records"], 1)
        self.assertEqual(result["correct_records"], 0)
        self.assertIn("DUPLICATE_OUTPUT", [e["code"] for e in result["anchor_errors"]])

    def test_overlapping_zones_ambiguity_and_reference_validation(self):
        anchors = [anchor_ref("r1", 10, 20), anchor_ref("r2", 30, 40)]
        validate_references(anchors)
        pred = record()
        pred["source_anchor"]["subject_spans"] = [span(10, 40)]
        pairs, errors = match_records([normalize_record(pred)], anchors)
        self.assertEqual(pairs, [])
        self.assertEqual(errors[0]["code"], "ANCHOR_AMBIGUITY")
        anchors[1]["alternatives"].append(deepcopy(anchors[0]["alternatives"][0]))
        with self.assertRaises(ValidationError):
            validate_references(anchors)

    def test_shared_subject_scopes_and_equivalent_alternatives(self):
        anchors = [anchor_ref("r1", scope=(100, 110)), anchor_ref("r2", scope=(120, 130))]
        validate_references(anchors)
        predictions = []
        for a in anchors:
            r = record()
            r["source_anchor"]["scope_spans"] = a["alternatives"][0]["scope_spans"]
            predictions.append(normalize_record(r))
        self.assertEqual(len(match_records(predictions, anchors)[0]), 2)
        anchors[0]["alternatives"].append(deepcopy(anchors[0]["alternatives"][0]))
        self.assertEqual(len(match_records(predictions, anchors)[0]), 2)
        anchors[0]["reference_id"], anchors[1]["reference_id"] = "z", "a"
        self.assertEqual(len(match_records(list(reversed(predictions)), anchors)[0]), 2)

    def test_directors_collective_records_and_repeated_proposal_numbers(self):
        anchors = [anchor_ref("alice", 10, 20, (10, 20)), anchor_ref("bob", 30, 40, (30, 40))]
        records = [record(), record()]
        records[1]["source_anchor"]["subject_spans"] = [span(30, 40)]
        records[1]["fields"]["separate_subject"] = field("Bob")
        f = filing(records, anchors)
        result = score_filing(json.dumps(response(records[::-1])), f, processing_coverage=PROCESSING)
        self.assertTrue(result["exact_filing_success"])
        combined = record()
        combined["source_anchor"]["subject_spans"] = [span(10, 40)]
        result = score_filing(json.dumps(response([combined])), f, processing_coverage=PROCESSING)
        self.assertEqual(result["matched_records"], 0)
        # Expanding one collective slate into individual outputs remains an extra-record failure.
        self.assertFalse(score(response([record(), record()]))["passed"])

    def test_empty_filings_and_pooled_failures(self):
        empty = filing([], [])
        good = score_filing(json.dumps(response([])), empty, processing_coverage=PROCESSING)
        self.assertTrue(good["correct_empty"])
        self.assertIsNone(good["strict_recall"])
        missing = score_filing(json.dumps(response([])), filing(), processing_coverage=PROCESSING)
        self.assertEqual(missing["strict_recall"], 0)
        summary = summarize_filings([score_filing(json.dumps(response()), filing(), processing_coverage=PROCESSING),
                                     score_filing('{', filing())])
        self.assertEqual(summary["strict_recall"], 0.5)
        self.assertEqual(summary["strict_precision_on_countable_responses"], 1)
        self.assertEqual(summary["precision_response_coverage"], 0.5)

    def test_lost_source_document_reduces_recall_and_processing_coverage(self):
        f = filing([record(), record()], [anchor_ref("r1"), anchor_ref("r2", 200, 210, (200, 300))])
        # A second complete original source exists even if normalization drops it.
        other = b"Second original voting document"
        digest = hashlib.sha256(other).hexdigest()
        f.context.documents["other"] = other
        f.context.hashes["other"] = digest
        f.context.permitted_ranges["other"] = [(0, len(other))]
        f.coverage_review["documents"].append({"document_id": "other", "source_sha256": digest, "disposition": "ELIGIBLE",
             "reason": "Second synthetic original", "reviewed_ranges": [[0, len(other)]]})
        alternative = f.anchors[1]["alternatives"][0]
        alternative["subject_spans"] = [span(0, 6, "other", digest)]
        alternative["subject_zones"] = [span(0, len(other), "other", digest)]
        result = score_filing(json.dumps(response()), f, processing_coverage=PROCESSING)
        self.assertEqual(result["strict_recall"], 0.5)
        self.assertFalse(result["processing_coverage_complete"])
        f.coverage_review["decision"] = "PARTIAL_REFERENCE"
        with self.assertRaises(ValidationError):
            score_filing(json.dumps(response()), f)


class DiagnosticsTests(unittest.TestCase):
    def test_broader_audit_keeps_its_scope_separate(self):
        audit = {"audit_id": "synthetic-audit", "field_path": "/fields/ticker", "document_ids": ["doc"],
                 "inspected_spans": [evidence()], "complete_scope": False, "reviewer": "Synthetic reviewer",
                 "reviewed_at": "2026-09-07", "finding": "NOT_DISCLOSED", "reason": "Not disclosed in the inspected region."}
        self.assertEqual(validate_source_audit(audit)["finding"], "NOT_DISCLOSED")
        self.assertEqual(record()["fields"]["ticker"]["availability"], "ABSENT_IN_CONTEXT")
        audit["inspected_spans"][0]["document_id"] = "unlisted"
        with self.assertRaises(ValidationError):
            validate_source_audit(audit)

    def test_wrong_json_types_do_not_crash_the_scorer(self):
        original = record()
        paths = []
        def walk(value, path=()):
            if isinstance(value, dict):
                for key, child in value.items():
                    paths.append(path + (key,))
                    walk(child, path + (key,))
            elif isinstance(value, list):
                for i, child in enumerate(value):
                    paths.append(path + (i,))
                    walk(child, path + (i,))
        walk(original)
        for path in paths:
            for wrong in (None, False, 0, 0.5, [], {}, "invalid", {"nested": [0.5]}):
                pred = deepcopy(original)
                parent = pred
                for key in path[:-1]:
                    parent = parent[key]
                parent[path[-1]] = wrong
                with self.subTest(path=path, wrong=wrong):
                    result = score(response([pred]), context=context())
                    self.assertIn("passed", result)
                    json.dumps(result, allow_nan=False)
                    json.dumps(score_filing(json.dumps(response([pred])), filing(),
                                           processing_coverage=PROCESSING), allow_nan=False)

    def test_decimal_diagnostics_preserve_numeric_type_and_raw_response(self):
        pred = response()
        f = pred["records"][0]["fields"]["issuer_name"]
        f["origin"] = 0.5
        f["evidence"][0]["quote"] = 0.5
        raw = json.dumps(pred).encode()
        saved = bytes(raw)
        result = score_fragment(raw, synthetic_reference(record()), context=context())
        claim = next(c for c in result["evidence"]["records"][0]["assertions"] if c["path"] == "/fields/issuer_name")
        self.assertEqual(claim["origin"], {"json_type": "number", "value": "0.5"})
        self.assertEqual(claim["evidence"][0]["quote"], {"json_type": "number", "value": "0.5"})
        self.assertFalse(result["passed"])
        json.dumps(result, allow_nan=False)
        self.assertEqual(raw, saved)
        f["value"] = 0.5
        numeric = score(pred)["evidence"]["records"][0]["assertions"][0]
        f["value"] = "0.5"
        textual = score(pred)["evidence"]["records"][0]["assertions"][0]
        self.assertEqual(numeric["value"], textual["value"])
        self.assertNotEqual(numeric["json_type"], textual["json_type"])

    def test_valid_d2_survives_unrelated_malformed_field_for_assertion_counts(self):
        r = record()
        r["fields"]["issuer_name"].update(value=123)
        report = evidence_report(r, context())
        self.assertEqual(len(report["d2_assertions"]), 1)
        self.assertFalse(any(c["path"] == "/fields/participation" for c in report["assertions"]))
        # Retain a wrong numeric primitive without making the diagnostic report unserializable.
        result = score(json.dumps(response([r])).replace('123', '1.23'))
        json.dumps(result)

    def test_malformed_claims_are_not_hidden(self):
        r = record()
        r["fields"]["issuer_name"].update(availability="AMBIGUOUS", origin="INFERRED", value="Unsupported guess")
        r["fields"]["ticker"].update(value={"invalid": "container"})
        r["fields"]["issuer_name"]["evidence"] = []
        claims = assertions(r)
        self.assertTrue(any(c["path"] == "/fields/issuer_name" for c in claims))
        self.assertTrue(next(c for c in claims if c["path"] == "/fields/ticker")["unassessable"])
        report = evidence_report(r, context())
        self.assertEqual(report["semantic_support"]["NOT_ASSESSED"], len(claims) - len(report["d2_assertions"]))
        self.assertGreaterEqual(report["citations"]["missing_required"], 1)

    def test_semantic_review_does_not_convert_citation_validity_into_support(self):
        before = score(response(), context=context())["evidence"]
        review = {"record_index": 0, "field_path": "/fields/issuer_name", "assessment": "UNSUPPORTED",
                  "reviewer": "Synthetic reviewer", "reviewed_at": "2026-09-07", "reason": "Citation exists but does not support the issuer claim."}
        after = apply_semantic_reviews(before, [review])
        self.assertEqual(before["records"][0]["semantic_support"]["UNSUPPORTED"], 0)
        self.assertEqual(after["records"][0]["semantic_support"]["unsupported_rate"], 1)
        summary = summarize_evidence([after, score("{")["evidence"]])
        self.assertEqual(summary["responses_with_unknown_assertion_counts"], 1)
        self.assertLess(summary["assessed_fraction_of_known_assertions"], 1)

    def test_matching_coverage_matches_exhaustive_assignment(self):
        for weights in ([[2, 3], [2, 0]], [[1], [3]], [[0, 4, 2], [3, 0, 4]], [[0, 0], [0, 0]]):
            n, m = len(weights), max(len(weights), len(weights[0]))
            padded = [row + [0] * (m - len(row)) for row in weights]
            expected = max(sum(padded[i][j] for i, j in enumerate(order)) for order in itertools.permutations(range(m), n))
            self.assertEqual(maximum_assignment(weights), expected)
        r, p = normalize_record(record()), normalize_record(record())
        p["fields"]["issuer_name"]["value"] = "wrong but answered"
        self.assertEqual(coverage(r, p)["rate"], 1)

    def test_prepared_evidence_requires_source_mapping(self):
        c = context()
        e = {**evidence(), "view_id": "v1", "block_id": "B1", "quote_basis": "PREPARED_VIEW", "quote": "visible"}
        self.assertFalse(c.citation_valid(e))
        c.views[("v1", "B1")] = {"span": span(0, len(RAW)), "text": "visible text"}
        self.assertTrue(c.citation_valid(e))
        e["source_sha256"] = "f" * 64
        self.assertFalse(c.citation_valid(e))


if __name__ == "__main__":
    unittest.main()
