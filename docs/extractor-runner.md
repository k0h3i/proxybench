# Extractor and development runner

The development runner preserves one attempt per scheduled fragment.
An attempt is one invocation of the declared system.
This stage supports local parser development and replay of recorded bytes.
Comparative evaluation, training, and new annotation require separate decisions.

## Date and version contract

The active guide is `benchmark-v1-date-order-v2`.
It accepts `M/D/YYYY`, `YYYY-MM-DD`, `Mon D, YYYY`, and `D-Mon-YYYY` after outer whitespace removal.
Numeric month and day accept one or two digits in slash dates.
ISO dates require two digits for month and day.
Named months use three English letters, without case sensitivity.
The parser rejects invalid calendar dates and never swaps an invalid month and day.

The original `benchmark-v1` policy retains its earlier unambiguous numeric-date rule.
Public normalization helpers default to that original policy for compatibility.
References carry their guide identity into scoring and normalization.
Unknown policy versions fail before scoring.
The schema remains `benchmark-v1` because the record structure does not change.

An explicit `Meeting Date (DD/MM/YYYY)` source heading permits extractor-side conversion.
The extractor preserves that heading and the date cell as evidence.
The scorer does not read headings to reinterpret submitted strings.

## Parser rules

The first layout contains a marked HTML row and explicit column headings.
The parser reads only the supplied `fragment-input-v1` bundle.
It uses original HTML cells and original byte positions.
Prepared text does not determine column associations.
The supported encoding is UTF-8.

| Field | Declared headings |
|---|---|
| Proposal number | Issue No., Ballot Issue, Proposal No, Proposal Number |
| Description | Description, Proposal |
| Proposal source | Proponent, Proposed By |
| Cast vote | Vote Cast |
| Recommendation | Mgmt Rec, Management Recommendation |
| Alignment | For/Agnst Mgmt |
| Issuer | Issuer |
| Date | Meeting Date, Meeting Date (DD/MM/YYYY) |
| Other context | Ticker, CUSIP, CUSIP9, Security ID, Meeting Type, Meeting Status |

Context uses adjacent label/value cells or aligned heading/value rows.
Context inside one spanning cell remains unsupported.
A single populated issuer cell before meeting context supplies the issuer name.
Fund context uses supplied standalone headings or explicit `Fund:`, `Fund Group:`, `Registrant:`, and `Registrant Name:` lines.
Table rows also accept `Fund Name`, `Fund`, and `Fund Group` with an adjacent value cell.
The latest fund heading controls attribution.
An incomplete later heading leaves the scope unresolved.
Explicit fund groups remain one scope, without invented members.

The target must match one complete row.
Continuation targets, nested tables, row spans, and unsupported column spans cause abstention.
Superscript footnote markers do not form part of header aliases.
Unknown directions remain unresolved and retain their source wording.
Split directions use semicolon-separated components, with optional decimal quantities followed by `shares` or `votes`.
An alignment cell does not establish alignment for each split component.
Those component alignments remain unresolved.
Zero quantities do not create cast votes.
Collective proposal wording stays in one record.
The parser does not infer individual director subjects, proposal categories, or proposal identities.
An explicit shareholder-proposal description conflicts with a `Board of Directors`, `Management`, or `Mgmt` proponent cell.
The parser preserves both cells as evidence for `CONFLICTING` proposal source.
It does not interpret source legends or map `NA` recommendations to `NONE`.

## Capture and execution contract

The runner freezes scheduled input hashes, reference bindings, system identity, run identity, and attempt number before execution.
It rejects duplicate inputs, stale bindings, incorrect capture associations, and existing output directories.
Reference admission occurs before launch, but the extractor receives only source bytes.
Scoring receives references after capture.

Each attempt preserves exact output bytes before JSON parsing.
Process observations, parsed output, normalized output, and scoring reports use separate files.
Malformed numbers use typed diagnostic objects when JSON serialization requires them.
The raw file remains the authority for exact bytes.
Missing responses remain missing and never become reconstructed answers.

Timeouts, crashes, truncation, and capture failures prevent execution eligibility.
Content scores remain separate from execution eligibility.
An echoed input-ID error remains a scoring diagnostic.
It does not change the capture association or field-scoring rules.
Every scheduled input receives a terminal report, with no retry.

The local subprocess receives explicit timeout and stdout/stderr byte limits.
The run records revision, dirty state, Python version, platform, command, and measurement boundaries.
Elapsed time covers launch through capture.
Peak memory and token measurements remain null with reasons when unavailable.
This development runner is not an operating-system security sandbox for arbitrary programs.
Only the inspected deterministic parser runs as a development subprocess.

## Development commands

`execution.runner.ScheduledInput` defines one input and its accepted reference binding.
Its fields are `input_id`, `input_path`, `input_sha256`, `reference_path`, `reference_sha256`, `binding_sha256`, and `split`.
Paths resolve under the supplied workspace root.
The runner accepts only `split: "development"`.
The schedule file contains a JSON array of these objects.

Run a new parser development attempt with this command:

```bash
PYTHONPATH=src python3 -m proxybench.execution \
  data/manifests/development-schedule.json artifacts/runs/new-development-run \
  --run-id new-development-run --timeout-seconds 10 --output-bytes 1048576
```

Use a new destination for every run.
`schedule.json` records the frozen schedule and effective limits before extraction starts.
Numbered attempt directories contain the supplied input, raw output, process observations, normalized records, and scores.
`report.json` lists every scheduled result.
An abstention can have valid process execution while failing whole-record scoring.

For replay, construct `RecordedCapture` objects and pass a `RecordedPredictionAdapter` to `run_development`.
Each capture requires an `Association` with run, system, input, input hash, and attempt identity.
Supply exact bytes as `raw`, or use `None` when no response exists.
The operator must supply original process observations with the capture.
Replay cannot independently establish hosted process outcomes or reconstruct missing observations.
It records original runtime as unavailable.

The scorer preserves a wrong echoed input ID as `INPUT_ID_MISMATCH`.
The adapter rejects a capture whose association names an unscheduled input, system, run, hash, or attempt.
Recorded output beyond the declared byte limit remains intact but loses execution eligibility.
