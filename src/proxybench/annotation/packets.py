"""Build source views from reviewed byte ranges without filling extraction fields."""

import hashlib
import html
import json
from pathlib import Path
import re


FIELDS = (
    "reporting_scope", "series_identifiers", "issuer_name", "security_identifiers",
    "ticker", "meeting_date", "meeting_type", "proposal_number", "raw_description",
    "separate_subject", "proposal_source", "participation", "vote_components",
    "management_recommendation",
)


def prepare_packet(root, packet_id, source_path, url, accession, spans, target, *, encoding="utf-8"):
    """Spans and target use zero-based, half-open offsets in original source bytes.

    Callers inspect boundaries and supply complete HTML blocks or plain text.
    The target is one HTML row, paragraph, or a text range in a plain-text span.
    This function does not establish that the target is one logical record.
    """
    root = Path(root)
    raw = (root / source_path).read_bytes()
    if not spans or any(not 0 <= start < end <= len(raw) for start, end in spans):
        raise ValueError("Source ranges must be nonempty and within the original file.")
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("Source ranges must be ordered and must not overlap.")
    start, end = target
    if not 0 <= start < end <= len(raw):
        raise ValueError("Target must be nonempty and within the original file.")
    containing = [(a, b) for a, b in spans if a <= start < end <= b]
    if len(containing) != 1:
        raise ValueError("Exactly one source range must contain the whole target.")
    directory = root / "data/packets/calibration" / packet_id
    normalized = root / "data/normalized/calibration" / packet_id
    if directory.exists() or normalized.exists():
        raise FileExistsError("Preserve earlier packet versions. Use a new packet identifier.")
    text_mode = Path(source_path).suffix.lower() == ".txt"
    pieces = []
    mappings = []
    for index, (a, b) in enumerate(spans, 1):
        piece = raw[a:b].decode(encoding, errors="strict")
        if a <= start < end <= b:
            before = raw[a:start].decode(encoding, errors="strict")
            marked = raw[start:end].decode(encoding, errors="strict")
            after = raw[end:b].decode(encoding, errors="strict")
            if text_mode:
                piece = html.escape(before) + '<mark data-target="true">' + html.escape(marked) + '</mark>' + html.escape(after)
            else:
                # The caller supplies a whole row or paragraph, preserving source layout.
                if not marked.lstrip().lower().startswith(("<tr", "<p ", "<p>", "<div")):
                    raise ValueError("HTML targets must start at a complete row, paragraph, or div.")
                if marked.lstrip().lower().startswith("<tr"):
                    marked = re.sub(r"(<tr\b[^>]*)(>)", r'\1 data-target="true"\2', marked, flags=re.I)
                else:
                    opening_end = marked.index(">")
                    marked = marked[:opening_end] + ' data-target="true"' + marked[opening_end:]
                piece = before + marked + after
        elif text_mode:
            piece = html.escape(piece)
        if text_mode:
            piece = "<pre>" + piece + "</pre>"
        elif piece.lstrip().lower().startswith("<tr"):
            piece = '<table style="width:100%;border-collapse:collapse">' + piece + '</table>'
        pieces.append(f'<section><p class="location">Source block {index}: bytes {a}–{b}</p>{piece}</section>')
        mappings.append({"block_id": f"B{index}", "start_byte": a, "end_byte": b,
                         "source_slice_sha256": hashlib.sha256(raw[a:b]).hexdigest(),
                         "line_start": raw[:a].count(b"\n") + 1,
                         "line_end": raw[:b-1].count(b"\n") + 1})
    directory.mkdir(parents=True)
    normalized.mkdir(parents=True)
    for i, (a, b) in enumerate(spans, 1):
        (normalized / f"block-{i}.source").write_bytes(raw[a:b])
    source_view = '''<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<style>body{font:16px/1.5 Georgia,serif;margin:18px;color:#17212b;background:white}
section{margin-bottom:28px;overflow:auto}table{max-width:100%}pre{white-space:pre;overflow:auto}
td,th{padding:4px;border-bottom:1px solid #d8dfe3}font,td,th,div,p{font-size:15px!important}
[data-target], [data-target] td{background:#fff0ad!important;outline:2px solid #915c00}
.location{font:13px system-ui;color:#53616d;border-top:1px solid #abb4bc;padding-top:8px}</style>
</head><body>''' + "\n".join(pieces) + "</body></html>"
    (normalized / "source-view.html").write_text(source_view, encoding="utf-8")
    manifest = {"packet_id": packet_id, "packet_version": 1, "split": "development",
                "status": "awaiting_source_first_review", "source_path": source_path,
                "source_url": url, "accession": accession, "encoding": encoding,
                "source_sha256": hashlib.sha256(raw).hexdigest(), "blocks": mappings,
                "target": {"start_byte": start, "end_byte": end,
                           "sha256": hashlib.sha256(raw[start:end]).hexdigest()},
                "display_transformations": ["Source blocks retain order, with explicit gaps.",
                                            "Source text uses the declared decoding without replacement.",
                                            "Target rows receive an added display attribute and color.",
                                            "Selected row groups receive a display table wrapper.",
                                            "Display fonts and cell spacing improve readability.",
                                            "The display disables scripts and external resources."],
                "predictions": None}
    (directory / "packet.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (normalized / "source-map.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest, source_view
