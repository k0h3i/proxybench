# Local source review

Use existing local source documents to prepare editable browser reviews.
Sol supplies draft labels, and the user reviews the source and corrects each draft.
Browser completion does not by itself approve a label for training.

## Prepare inspected source ranges

The `proxybench.annotation.packets.prepare_packet` function takes ordered source ranges and one target range.
Each range uses byte offsets, counted from zero, with an exclusive end.
The function stores source slices and mappings under `data/normalized/calibration/`.
It stores administrative packet metadata under `data/packets/calibration/`.

For HTML, supply complete source blocks and a complete target row, paragraph, or division.
For text, supply the exact target text range within one context block.
Include fund and issuer headings, column headers, meeting context, and applicable legends.
Retain adjacent context when it establishes a boundary.
Inspect the display before treating a packet as ready for user review.

The prepared display adds highlighting without changing source text.
It adds table wrappers around selected row groups and highlights all continuation rows within the one target range.
The display blocks source scripts and external resources.
The function rejects overlapping ranges, incomplete target coverage, and replacement of an existing packet identifier.
It does not determine whether a physical row represents one logical record.
It does not fill extraction fields or create reference labels.

## Prepare a new development review page

The local packet set can supply a browser review page with a timer and optional hidden assistant drafts.
The page runs from a local HTML file without a server.
It exports draft annotations and active review time to a numbered JSON download.

```bash
PYTHONPATH=src python3 -m proxybench.annotation.review \
  --packet-directory data/packets/review-v2 \
  --drafts artifacts/runs/review-v2/draft-labels.json
```

Replace the example paths with a new inspected review version.
The command reads `packet-set.json` and creates `index.html` in the selected directory.
It refuses an existing page and leaves historical pages unchanged.
Omit `--drafts` when the review has no suggestions.

Each packet requires its original `manifest`, `source_view`, and exact source-only `model_input` text.
The manifest must identify the split as `development`.
`annotation.bindings.review_binding(packet)` computes the binding for a suggestion's `input_binding` field.
A binding records the hashes of the manifest, displayed source, and model input.
Suggestions also require their existing `packet_fingerprint`, source identity, packet version, and complete field checklist.

Saved drafts and revealed suggestions must match the entire binding before reuse.
Changed targets, context ranges, displayed text, or model input invalidate automatic reuse.
Open the generated page in a browser and start the timer before reading the sources or guide.

Read the source and form an answer before selecting `Reveal draft labels`.
The page fills untouched fields and preserves existing answers.
The user can revise suggestions without first typing a separate answer set.
The export preserves the original suggestions, reveal times, earlier answers, and later differences.
Assistant suggestions do not become accepted reference labels automatically.

The visible name `Reporting fund or fund group` distinguishes the reporting fund from the issuer.
New and historical drafts retain the key `reporting_scope`.
Preserve raw dates and document any assumed date order separately from source facts.
The first pilot uses month/day/year or year-month-day, with ambiguous interpretations marked `INFERRED`.
This pilot convention does not establish an SEC-wide rule.

The page pauses when its tab becomes hidden.
The first session stops after 30 active minutes or at the four-hour total limit, based on entered earlier review time.
Count off-page guide reading, corrections, and feedback in the earlier-time entry or the separate review log.
Keep at least 30 minutes of the total budget for corrections.

Download the draft when you stop, even if some fields remain unreviewed.
Preserve that file when importing reviewed versions into `data/annotations/calibration/`.
The browser stores only the current working draft, so downloaded revisions preserve earlier decisions.
Resuming a matching draft retains its active time and input binding.
Answer values and original source wording occupy separate boxes.
Value edits preserve source wording and append the earlier value to correction history.
Record correction reasons in the feedback box before marking the packet reviewed.
These development drafts still require reviewed conversion before typed reference admission.

This page retains the first pilot's reveal procedure and refuses test packets.
Future test review must save a source-only initial answer before revealing suggestions.
That procedure needs a separate annotation allocation.
