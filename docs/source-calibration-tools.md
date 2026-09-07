# Source calibration tools

These helpers support the first local source review.
They do not implement filing discovery, automatic record extraction, or evaluation.
Source selection and logical target boundaries still require inspection.

## Retrieve selected source documents

Before retrieval, read the current [SEC access guidance](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data).
Supply a real client name and contact email in a local file such as `notes/private/sec-identity.txt`.
Keep the file outside Git.

The retrieval command takes an accession, which identifies one filing submission, and its document URL.
It preserves the source under `data/raw/<accession>/` with a hash and retrieval metadata.
It records requests under `data/manifests/retrieval-log.jsonl`.

```bash
PYTHONPATH=src python3 -m proxybench.sources.sec \
  ACCESSION SEC_DOCUMENT_URL --identity-file notes/private/sec-identity.txt
```

Replace `ACCESSION` and `SEC_DOCUMENT_URL` with the inspected filing references.
Use one sequential `SecClient` instance when retrieving several documents through Python.
Do not run concurrent clients.
The helper reuses unchanged cached sources and never overwrites an existing source.
An HTTP 403 or 429 response stops further downloads in that client session.
Inspect a denial before starting another client session.

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

## Open the first review page

The local packet set can supply a browser review page with a timer and optional hidden assistant drafts.
The page runs from a local HTML file without a server.
It exports draft annotations and active review time to a numbered JSON download.

```bash
PYTHONPATH=src python3 -m proxybench.annotation.review

# Include preserved assistant drafts for a source-then-reveal review.
PYTHONPATH=src python3 -m proxybench.annotation.review \
  --drafts artifacts/runs/calibration-assistant-v1/draft-labels.json
```

The command reads `data/packets/calibration/packet-set.json` and writes `data/packets/calibration/index.html`.
It requires an existing inspected packet set.
When drafts are supplied, the command matches their source identities, context ranges, packet versions, and field checklist before building the page.
Open the generated page in a browser and start the timer before reading the sources or guide.

Read the source and form an answer before selecting `Reveal draft labels`.
The page fills untouched fields and preserves existing answers.
The user can revise suggestions without first typing a separate answer set.
The export preserves the original suggestions, reveal times, earlier answers, and later differences.
Assistant suggestions do not become accepted reference labels automatically.

The visible name `Reporting fund or fund group` distinguishes the reporting fund from the issuer.
Existing saved drafts retain the key `reporting_scope`.
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
Resuming a downloaded draft retains its active time and source identities.
Draft field values retain source wording and require review before conversion to the final record schema.

## Validate the helpers

The tests use synthetic sources and do not contact SEC.
They cover cached bytes, retrieval failures, source boundaries, and display mappings.
They do not establish historical coverage or record accuracy.

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
