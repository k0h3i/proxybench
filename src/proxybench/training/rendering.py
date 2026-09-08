"""Render provisional source-only packets and complete benchmark targets."""

from copy import deepcopy
import hashlib
import html
import json

from proxybench.extraction.bundles import read_bundle
from proxybench.normalization.values import ACTIVE_GUIDE
from proxybench.schemas.records import FIELD_TYPES, D2_RULE, normalize_record, validate_envelope
from proxybench.evaluation.evidence import SourceContext


def absent():
    return dict(value=None, raw_text=None, availability="ABSENT_IN_CONTEXT", origin=None,
                evidence=[], reason=None, rule_id=None)


def present(value, evidence, raw_text=None):
    return dict(value=value, raw_text=raw_text, availability="PRESENT", origin="EXTRACTED",
                evidence=deepcopy(evidence), reason=None, rule_id=None)


def render_record(mapped, *, input_id, group_id, style="text", omit=()):
    if mapped["status"] != "PROVISIONAL_GENERATED" or style not in {"text", "table"}:
        raise ValueError("Only provisional mapped records and supported styles can be rendered")
    cells = [c for c in mapped["cells"] if c["key"] not in omit]
    if not any(c["key"] == "description" for c in cells):
        raise ValueError("The logical target cannot be omitted")
    pieces, offsets, position = [], {}, 0
    prefix = "<table>\n" if style == "table" else ""
    pieces.append(prefix)
    position += len(prefix.encode())
    for cell in cells:
        text = (f'<tr><th>{html.escape(cell["label"])}</th><td>{html.escape(cell["text"])}</td></tr>\n'
                if style == "table" else f'{cell["label"]}: {cell["text"]}\n')
        size = len(text.encode("utf-8"))
        offsets[cell["key"]] = (position, position + size, text)
        pieces.append(text)
        position += size
    if style == "table":
        pieces.append("</table>\n")
    raw = "".join(pieces).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    document_id, view_id = input_id + "-source", input_id + "-view"
    blocks, evidence, lineage = [], {}, []
    by_key = {c["key"]: c for c in cells}
    for i, cell in enumerate(cells):
        start, end, original = offsets[cell["key"]]
        span = dict(document_id=document_id, source_sha256=digest, start_byte=start, end_byte=end)
        block_id = f"block-{i:03d}"
        prepared = f'{cell["label"]}: {cell["text"]}'
        blocks.append(dict(block_id=block_id, span=span, original_text=original, prepared_text=prepared))
        evidence[cell["key"]] = [dict(**span, view_id=view_id, block_id=block_id,
                                      quote=prepared, quote_basis="PREPARED_VIEW")]
        lineage.append({"rendered_span": span, "xml_span": cell["source"], "key": cell["key"]})
    start, end, _ = offsets["description"]
    bundle = dict(input_version="fragment-input-v1", input_id=input_id, track="fragment", view_id=view_id,
                  document_id=document_id, encoding="utf-8", source_sha256=digest,
                  target=dict(start_byte=start, end_byte=end, sha256=hashlib.sha256(raw[start:end]).hexdigest()),
                  blocks=blocks)

    def field(key, value=None, raw_text=None):
        if key not in by_key:
            return absent()
        text = by_key[key]["text"]
        return present(text if value is None else value, evidence[key], text if raw_text is None else raw_text)

    fields = {key: absent() for key in FIELD_TYPES}
    if "scope" in by_key:
        members = absent()
        members.update(availability="NOT_APPLICABLE", evidence=deepcopy(evidence["scope"]),
                       reason="The disclosed reporting scope is an individual fund.")
        fields["reporting_scope"] = present({"name": field("scope"),
                                             "scope_type": field("scope", "INDIVIDUAL_FUND", "Reporting fund"),
                                             "members": members}, evidence["scope"])

    def identifier(key, label):
        return {"source_label": field(key, label, label), "value": field(key)}

    if "series" in by_key:
        fields["series_identifiers"] = field("series", [identifier("series", "Series ID")])
    keys = [key for key in ("cusip", "isin", "figi") if key in by_key]
    if keys:
        fields["security_identifiers"] = present([identifier(k, k.upper()) for k in keys],
                                                 [e for k in keys for e in evidence[k]])
    for target, key in (("issuer_name", "issuer"), ("meeting_date", "date"),
                        ("raw_description", "description"), ("proposal_source", "proponent")):
        fields[target] = field(key)
    components, voting_evidence = [], []
    for j in range(mapped["components"]):
        key = f"direction:{j}"
        if key not in by_key:
            # Omitting one direction would conceal component membership.
            if any(f"direction:{k}" in by_key for k in range(mapped["components"])):
                raise ValueError("Partial vote-component omission is unsupported")
            continue
        direction = by_key[key]["text"]
        quantity_key, alignment_key = f"quantity:{j}", f"alignment:{j}"
        quantity = (field(quantity_key, {"amount": field(quantity_key), "unit": field(quantity_key, "shares", "Shares voted")})
                    if quantity_key in by_key else absent())
        alignment = absent()
        if alignment_key in by_key:
            value = {"FOR": "WITH_MANAGEMENT", "AGAINST": "AGAINST_MANAGEMENT", "NONE": "OTHER"}[
                by_key[alignment_key]["text"]]
            alignment = field(alignment_key, value)
        components.append({"direction": field(key, direction if direction in {"FOR", "AGAINST", "ABSTAIN", "WITHHOLD"} else "OTHER"),
                           "quantity": quantity, "disclosed_management_alignment": alignment})
        voting_evidence.extend(evidence[key])
        if quantity_key in evidence:
            voting_evidence.extend(evidence[quantity_key])
    if components:
        fields["vote_components"] = present(components, voting_evidence)
        known_cast = any(c["direction"]["value"] != "OTHER" or c["quantity"]["availability"] == "PRESENT" for c in components)
        if known_cast:
            fields["participation"] = present("VOTED", voting_evidence)
            fields["participation"].update(origin="DERIVED", rule_id=D2_RULE)
    record = dict(record_id=input_id + "-target", source_anchor={"subject_spans": [blocks[
        next(i for i, c in enumerate(cells) if c["key"] == "description")]["span"]],
        "scope_spans": []}, fields=fields, enrichments=[])
    if "scope" in evidence:
        record["source_anchor"]["scope_spans"] = [{k: evidence["scope"][0][k] for k in
                                                    ("document_id", "source_sha256", "start_byte", "end_byte")}]
    response = dict(schema_version="benchmark-v1", input_id=input_id, status="COMPLETE", records=[record], failure=None)
    validate_envelope(response)
    normalize_record(record, guide_version=ACTIVE_GUIDE)
    read_bundle(json.dumps(bundle))
    context = SourceContext({document_id: raw}, {document_id: [(b["span"]["start_byte"], b["span"]["end_byte"]) for b in blocks]},
                            views={(view_id, b["block_id"]): {"span": b["span"], "text": b["prepared_text"]} for b in blocks})
    for cell_evidence in evidence.values():
        assert all(context.citation_valid(e) for e in cell_evidence)
    return {"source_bytes": raw, "bundle": bundle, "response": response,
            "lineage": {"group_id": group_id, "split": "development", "logical_source": mapped["source"],
                        "rule_version": mapped["rule_version"], "style": style, "omitted": list(omit),
                        "locations": lineage, "human_review": "PENDING", "training_admitted": False}}
