# Sources and accepted labels

The retained dataset contains 330 training examples and 90 development examples from 36 source files.
Keep its existing split and row order.
The development examples contain known exposure and do not form an untouched test set.
The manifest also retains nine unselected exposed source identities among 45 source assignments.
Future preparation rejects training assignments that conflict with these restrictions.

## Acquisition and source storage

Keep complete ordinary files in the flat `data/raw/` directory.
Keep complete test files separately in `data/raw/test/`.
The source manifest records original locations, friendly filenames, accession identities, encodings, and hashes where known.
Keep amendments and attachments separate.
Deduplicate identical bytes only when every original source location remains recorded.
Do not retain incomplete downloads or SEC error pages.

Use `python -m proxybench fetch-source --help` for future SEC requests.
Set `PROXYBENCH_SEC_IDENTITY` privately before acquisition.
Inspect the complete download, then use `python -m proxybench import-source --help` to admit it.
Supply a declared client identity through private runtime configuration.
The project limits requests to two per second and preserves bounded downloads.
Do not bypass access blocks.
Read the [SEC request guidance](https://www.sec.gov/about/webmaster-frequently-asked-questions) before new acquisition.
Importing a local source does not authorize public redistribution.

## Manual target selection

Select one separately voted subject and its disclosed reporting scope.
Use the original source to choose byte ranges, an ordered sequence of source spans, and the target boundary.
Ranges use zero-based offsets with an excluded end position.
Keep the source encoding and SHA-256 hash with each selection.
A hash identifies exact file content.

Text spans preserve decoded characters and whitespace.
HTML spans preserve ordered cells, empty cells, and row or column spans.
The `historical-cells-v1` renderer decodes character references once and collapses whitespace inside HTML cells.
Generated block labels and omitted-range markers are not source text.
Shared-row subtargets remain unsupported.
Defer a target when complete adjacent blocks cannot represent it.

A selection contains `packet_id`, `source_path`, `source_sha256`, `accession`, `group_id`, `split`, `encoding`, `spans`, and `target`.
It also records `boundary_review`, `reviewer`, and the derivation or source-association metadata required by the label validator.
Each span contains its start, end, and kind: `text`, `row`, `cell`, or `block`.
Standalone HTML cells can provide context but cannot define shared-row targets.
Use project-relative paths without parent traversal.

## Label review and acceptance

Prepare source packets through `python -m proxybench prepare --help`.
Give the labeling agent the original source context and the [label contract](label-contract.md).
Use the [canonical prompt](../configs/model-system-prompt.txt) as the exact system message.
The user message contains source context without another copy of the labeling policy.

Show sources before suggestions in the editable browser.
Keep corrections and uncertainty visible for review.
A reviewed checkbox alone does not grant training acceptance.
Export the review and record explicit acceptance of its exact hash and approved packet IDs.
A changed review requires new acceptance.
Use `python -m proxybench accept --help` to export accepted labels.
Rejected and unreviewed labels cannot enter the dataset.

## Dataset format and boundaries

`data/training-dataset/` contains the following files:

- `training-examples.jsonl`: The ordered training messages.
- `development-examples.jsonl`: The ordered development messages.
- `dataset-manifest.json`: Source selections, file hashes, split assignments, and label rules.

JSONL stores one JSON object on each line.
Each row has three messages in order: `system`, `user`, and `assistant`.
The assistant contains only the accepted fourteen-field label object.
The manifest uses schema `training-dataset-v1` and declares project-relative prompt and contract paths.
It needs no parent dataset or experiment output.

The reader rebuilds each user message from its source selection and compares it with the saved message.
It also checks source hashes, prompt bytes, row hashes, order, duplicate targets, labels, and exposure restrictions.
Assistant labels remain unchanged during path migration.
The source and label files remain private unless the user separately authorizes their exact release list.
See the [release guide](release.md) for raw-source review.

## Test-only preparation

The [test configuration](../configs/testing.json) defines twelve records from twelve distinct filings and SEC filing years, 2013 through 2024.
The 2024 record requires standardized disclosures in text or HTML, including an official SEC HTML view.
Record the reporting period separately from the filing year.
Apply the existing fourteen-field contract without adding standardized-only output fields.
The test configuration fixes the scoring version before any predictions.

Keep accepted test messages and their manifest under `data/testing-dataset/`.
Use temporary files for selections, review records, and source audits before acceptance.
The accepted manifest retains the required review and audit evidence.
Retain the prior dataset and source inventory before acquiring test sources.
Record candidate exclusions, amendment relationships, source comparisons, and coverage gaps in the private selection audit.
Do not weaken year or source-independence requirements to fill a coverage gap.
Keep a hash-bound provider-family audit that covers prior sources, exposed accessions, and each test candidate.
Use `family_id` for a reviewed provider identity shared by related funds.
Do not treat a new registrant identifier as proof of an unexposed provider.

Use `python -m proxybench prepare-test --help` for protected test preparation.
Supply selections, the test configuration, and the source audit.
For context from another document, supply `context_sources` with the same accession and an explicit association review.
Retain each document's original bytes, hash, URL, encoding, and source ranges.

Preparation reserves filing, source-group, registrant, and file identities in `data/test-source-ledger.json` before review.
It also protects other admitted attachments from the same filing.
Keep this ledger even when labels remain drafts.
Ordinary preparation and the training reader reject protected sources, including attached context and matching file bytes.
The ledger also retains reviewed `family_id` values without changing older group reservations.

Review source context before revealing draft labels.
Require independent agent review of drafts and scripts before handoff.
Explicit user acceptance must bind the exact review hash, preparation hash, and ordered packet IDs.
Use `python -m proxybench accept-test --help` only after that acceptance.

The accepted test dataset uses schema `test-dataset-v1`.
Its folder contains only `test-examples.jsonl` and `dataset-manifest.json`.
The manifest retains the exact review, approval, and preparation text with their hashes.
It also retains the accepted audit documents, source identity records, and protection metadata.
Preparation paths in these records identify frozen evidence, not files that evaluation must read.
Eligibility checks use live source and exposure records before acceptance.
Finalized loading uses the accepted evidence and rebuilds inputs directly from the original test files.
It needs no preparation folder, parent training dataset, prior raw filings, or live source inventory.
Raw test sources, canonical policies, and the protection ledger remain required.
Each example keeps the same `system`, `user`, and `assistant` message order as training examples.
The test reader rebuilds sources and checks the exact acceptance record before evaluation.
Evaluation refuses a scoring version that differs from the accepted test configuration.

Evaluation selects the test split from this schema and sends only `system` and `user` messages to the model.
Do not use test results to select prompts, models, thresholds, or replacement records.
Identify later development-driven evaluations as regression testing, which checks behavior after changes, rather than an untouched test.
This small, deliberately varied batch does not establish population accuracy or exclusion from base-model pretraining.
