# Sources and accepted labels

The retained dataset contains 330 training examples and 90 development examples from 36 source files.
Keep its existing split and row order.
The development examples contain known exposure and do not form an untouched test set.
The manifest also retains nine unselected exposed source identities among 45 source assignments.
Future preparation rejects training assignments that conflict with these restrictions.

## Acquisition and source storage

Keep complete original files in the flat `data/raw/` directory.
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
