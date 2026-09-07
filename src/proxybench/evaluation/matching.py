"""Match complete-filing records using reviewed source anchors only."""

from proxybench.schemas.records import object_keys, require, validate_anchor, validate_span


def same_source(a, b):
    return (a["document_id"], a["source_sha256"]) == (b["document_id"], b["source_sha256"])


def inside(span, zone):
    return same_source(span, zone) and zone["start_byte"] <= span["start_byte"] < span["end_byte"] <= zone["end_byte"]


def overlaps(a, b):
    return same_source(a, b) and max(a["start_byte"], b["start_byte"]) < min(a["end_byte"], b["end_byte"])


def matches(anchor, alternative):
    if anchor is None:
        return False
    for role in ("subject", "scope"):
        predicted, minimal, zones = anchor[role + "_spans"], alternative[role + "_spans"], alternative[role + "_zones"]
        if not minimal:
            if predicted:
                return False
        elif (not predicted or not all(any(inside(s, z) for z in zones) for s in predicted)
              or not any(overlaps(s, m) for s in predicted for m in minimal)):
            return False
    return True


def candidates(anchor, references):
    return [r["reference_id"] for r in references if any(matches(anchor, a) for a in r["alternatives"])]


def validate_references(references, span_check=None):
    require(isinstance(references, list), "Anchor references must be an array")
    seen = set()
    for reference in references:
        object_keys(reference, {"reference_id", "alternatives"}, "anchor_reference")
        rid = reference["reference_id"]
        require(isinstance(rid, str) and bool(rid.strip()) and rid not in seen, "Anchor reference IDs must be unique")
        seen.add(rid)
        require(isinstance(reference["alternatives"], list) and bool(reference["alternatives"]), "Anchor alternatives required")
        for alternative in reference["alternatives"]:
            object_keys(alternative, {"subject_spans", "subject_zones", "scope_spans", "scope_zones"}, "alternative")
            for role in ("subject", "scope"):
                spans, zones = alternative[role + "_spans"], alternative[role + "_zones"]
                require(isinstance(spans, list) and isinstance(zones, list), "Anchor spans and zones must be arrays")
                require(bool(spans) == bool(zones) and (role != "subject" or bool(spans)), "Inconsistent anchor zones")
                for span in spans + zones:
                    validate_span(span)
                    if span_check is not None:
                        require(span_check(span), "Reference span lies outside permitted source")
                require(all(any(inside(s, z) for z in zones) for s in spans), "Minimal span lies outside zones")
    for reference in references:
        for alternative in reference["alternatives"]:
            anchor = {k: alternative[k] for k in ("subject_spans", "scope_spans")}
            require(candidates(anchor, references) == [reference["reference_id"]],
                    "Reference minimal anchor does not identify exactly its own logical record")


def match_records(records, references):
    pairs, errors, used = [], [], set()
    for i, record in enumerate(records):
        if record is None:
            errors.append({"prediction": i, "code": "MALFORMED_RECORD"})
            continue
        anchor = record["source_anchor"]
        validate_anchor(anchor)
        ids = candidates(anchor, references)
        if len(ids) == 1 and ids[0] not in used:
            pairs.append((ids[0], i))
            used.add(ids[0])
        else:
            code = ("ANCHOR_ABSTENTION" if anchor is None else "ANCHOR_AMBIGUITY" if len(ids) > 1
                    else "DUPLICATE_OUTPUT" if ids else "UNMATCHED_ANCHOR")
            errors.append({"prediction": i, "code": code})
    return pairs, errors
