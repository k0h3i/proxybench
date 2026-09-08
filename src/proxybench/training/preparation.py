"""Create new private preparation runs from an explicit bounded source manifest."""

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path

from proxybench.execution.runner import write_json
from proxybench.normalization.npx import map_filing
from proxybench.sources.npx_xml import XMLRejected
from proxybench.training.rendering import render_record


def prepare(manifest, output):
    if not 1 <= len(manifest["filings"]) <= 4 or sum(len(f["selected"]) for f in manifest["filings"]) > 24:
        raise ValueError("Preparation permits at most four filings and twenty-four selected entries")
    if manifest.get("split") != "development" or not manifest.get("group_id"):
        raise ValueError("Explored source groups must be frozen as development")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Freeze grouping and source identities before any renderings exist.
    write_json(output / "source-manifest.json", manifest)
    reports, previews, audit = [], [], []
    for filing in manifest["filings"]:
        sources = {}
        for role in ("primary", "votes"):
            item = filing[role]
            raw = Path(item["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != item["sha256"]:
                raise ValueError("Source hash differs from frozen manifest")
            sources[role] = raw
        try:
            report = map_filing(sources["primary"], sources["votes"],
                                primary_id=filing["primary"]["document_id"], votes_id=filing["votes"]["document_id"],
                                selected=filing["selected"])
        except XMLRejected as error:
            reports.append({"accession": filing["accession"], "status": "FILING_REJECTED", "reason": str(error),
                            "selected": len(filing["selected"]), "parsed": None, "admitted": 0,
                            "sources": {k: filing[k] for k in ("primary", "votes")}})
            continue
        report["accession"] = filing["accession"]
        reports.append(report)
        for mapped in report["records"]:
            if mapped["status"] != "PROVISIONAL_GENERATED":
                continue
            logical_id = f'{filing["accession"]}-{mapped["index"]:05d}'
            for style in ("text", "table"):
                input_id = logical_id + "-" + style
                packet = render_record(mapped, input_id=input_id, group_id=manifest["group_id"], style=style)
                directory = output / "previews" / input_id
                directory.mkdir(parents=True)
                (directory / ("source.html" if style == "table" else "source.txt")).write_bytes(packet["source_bytes"])
                for name in ("bundle", "response", "lineage"):
                    write_json(directory / (name + ".json"), packet[name])
                counts = Counter(f["availability"] for f in packet["response"]["records"][0]["fields"].values())
                previews.append({"input_id": input_id, "logical_id": logical_id, "style": style,
                                 "path": str(directory), "status": "PROVISIONAL_GENERATED",
                                 "rule_version": mapped["rule_version"], "training_admitted": False,
                                 "mechanical_validation": "PASS", "field_states": dict(counts)})
            if mapped["index"] in filing.get("audit", []):
                spans = {"primary": [], "votes": []}
                for cell in mapped["cells"]:
                    p = cell["source"]
                    role = "primary" if p["document_id"] == filing["primary"]["document_id"] else "votes"
                    raw = sources[role][p["start_byte"]:p["end_byte"]].decode()
                    if raw not in spans[role]:
                        spans[role].append(raw)
                source = "\n".join(spans["primary"]) + "\n" + sources["votes"][mapped["source"]["start_byte"]:mapped["source"]["end_byte"]].decode()
                text = render_record(mapped, input_id=logical_id + "-text", group_id=manifest["group_id"])
                table = render_record(mapped, input_id=logical_id + "-table", group_id=manifest["group_id"], style="table")
                audit.append({"logical_id": logical_id, "source": source, "text": text, "table": table})
    if len(audit) > 8:
        raise ValueError("Audit preview cap exceeded")
    write_json(output / "mapping-report.json", reports)
    write_json(output / "preview-manifest.json", previews)
    write_json(output / "audit-status.json", [{"logical_id": a["logical_id"], "status": "PENDING", "active_seconds": None,
                                              "corrections": [], "unresolved": [], "explicit_acceptance": False} for a in audit])
    sections = []
    for i, item in enumerate(audit, 1):
        sections.append(f'<section><h2>Example {i}: {html.escape(item["logical_id"])}</h2>'
                        '<p>Read the original disclosure first. Compare both renderings before opening the generated target.</p>'
                        f'<h3>Original XML disclosure</h3><pre>{html.escape(item["source"])}</pre>'
                        f'<h3>Plain text</h3><pre>{html.escape(item["text"]["source_bytes"].decode())}</pre>'
                        f'<h3>Table</h3>{item["table"]["source_bytes"].decode()}'
                        '<details><summary>Reveal provisional generated target after source inspection</summary>'
                        f'<pre>{html.escape(json.dumps(item["text"]["response"], ensure_ascii=False, indent=2))}</pre></details></section>')
    page = ('<!doctype html><meta charset="utf-8"><title>Modern mapping audit</title>'
            '<style>body{max-width:1100px;margin:2em auto;font:18px sans-serif}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}'
            'th,td{border:1px solid #aaa;padding:.4em;text-align:left}section{margin:3em 0}summary{cursor:pointer}</style>'
            '<h1>Modern mapping audit</h1><p>All targets are provisional development examples. No acceptance is recorded by this page.</p>'
            '<p>Record active review time, corrections, unresolved questions, and explicit acceptance separately.</p>' + ''.join(sections))
    (output / "audit.html").write_text(page)
    return {"filings": len(reports), "previews": len(previews), "audit_examples": len(audit), "admitted": 0}


def record_review(previews, event):
    """Apply an explicit review event to copies, retaining mechanical results."""
    import copy
    if event.get("status") not in {"ACCEPTED", "UNRESOLVED", "RULE_FAILED"}:
        raise ValueError("Unknown review decision")
    if (not isinstance(event.get("active_seconds"), (int, float)) or isinstance(event["active_seconds"], bool)
            or not 0 <= event["active_seconds"] < float('inf') or not event.get("reviewer")
            or not event.get("corrections") and event["status"] == "RULE_FAILED"):
        raise ValueError("Review needs time, reviewer, and correction details for a failed rule")
    matching = [p for p in previews if p["logical_id"] == event["logical_id"]]
    if not matching:
        raise ValueError("Review names an unknown logical record")
    rules = {p["rule_version"] for p in matching}
    output = copy.deepcopy(previews)
    for item in output:
        if event["status"] == "RULE_FAILED" and item["rule_version"] in rules:
            item["status"] = "QUARANTINED"
        elif item["logical_id"] == event["logical_id"]:
            if item["status"] == "QUARANTINED":
                raise ValueError("A failed rule needs correction and a new version before repeat audit")
            item["status"] = "HUMAN_REVIEWED" if event["status"] == "ACCEPTED" else "UNRESOLVED"
        item["training_admitted"] = False
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(json.loads(args.manifest.read_text()), args.output)))


if __name__ == "__main__":
    main()
