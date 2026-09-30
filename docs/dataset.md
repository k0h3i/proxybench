# Sources and accepted labels

The retained dataset contains 330 training examples and 90 validation examples from 36 source files.
Keep its existing split and row order.
The validation examples form the development set and contain known project exposure.

## Acquisition and source storage

Keep complete source files in the flat `data/raw/` directory.
The source manifest records original locations, local filenames, and SEC filing identities.
Keep amendments and attachments separate.
Deduplicate identical bytes only when every original source location remains recorded.
Do not retain incomplete downloads or SEC error pages.

### Choose a filing

Open [SEC EDGAR search](https://www.sec.gov/edgar/search/) and filter for `N-PX` or `N-PX/A`.
Choose the filing year and fund that you need.
Open the filing details.
Copy the official HTTPS URL for its complete text submission or an original HTML document.
Keep the filing year separate from the disclosed reporting period.

The repository supports historical text and HTML with one manually marked voting target.
PDF processing and optical character recognition, which converts images to text, remain outside scope.

An accession number identifies an SEC filing.
Record its dashed `0000000000-00-000000` form from the filing details.
Keep each amendment under its own accession number.
Use the original SEC filename, without directories, for the download.
Choose a unique lowercase filename for the imported copy, such as `fund-npx-2014.html`.
That filename must use only letters, numbers, periods, or hyphens.

### Download the original bytes

Run the commands in Bash on Linux or WSL2 from the repository root.
Source preparation requires Python 3.11 or later and no model dependencies.
If `.venv/` is missing, create it with `python3 -m venv .venv`.
Install the CPU package:

```bash
.venv/bin/python -m pip install -e .
```

Read the [SEC request guidance](https://www.sec.gov/about/webmaster-frequently-asked-questions) before new acquisition.
Enter your declared client name and contact email at the hidden prompt.
The prompt keeps the value out of shell history.
Enter the document URL, accession number, and original filename from the filing details:

```bash
read -r -s -p 'SEC client name and contact email: ' PROXYBENCH_SEC_IDENTITY
printf '\n'
export PROXYBENCH_SEC_IDENTITY
read -r -p 'SEC HTTPS document URL: ' PROXYBENCH_SOURCE_URL
read -r -p 'Accession number (0000000000-00-000000): ' PROXYBENCH_ACCESSION
read -r -p 'Original SEC filename: ' PROXYBENCH_ORIGINAL_FILENAME

.venv/bin/python -m proxybench fetch-source \
  --url "$PROXYBENCH_SOURCE_URL" \
  --output "data/downloads/$PROXYBENCH_ACCESSION/$PROXYBENCH_ORIGINAL_FILENAME" \
  --ledger data/sec-download-ledger.json \
  --kind filing
```

The download stays outside `data/raw/` until you inspect it.
A download ledger records requests, byte counts, file counts, and access blocks.
Reuse `data/sec-download-ledger.json` for related requests.
Do not start concurrent download commands.
The downloader limits requests to two per second, each document to 100 MiB, and the ledger to 4 GiB and 60 filings.
Do not bypass access blocks or replace a blocked ledger to continue requests.

The downloader requires at least 4 GiB available host memory.
Free disk space must cover the remaining download allowance plus 2 GiB.
Use `.venv/bin/python -m proxybench fetch-source --help` for the command reference.

### Inspect and import the filing

Read downloaded HTML as text to prevent active content from running.
Compare the downloaded document with its EDGAR filing entry.
Make sure that the document contains the expected final section.
Reject incomplete downloads, empty documents, and SEC error pages.
Preserve the original bytes.
A successful download alone does not establish completeness.

After you inspect the complete document, choose its local filename and enter the actual retrieval date.
The `--complete` flag records your decision that the document is complete.
Import the inspected document:

```bash
read -r -p 'Local lowercase filename: ' PROXYBENCH_LOCAL_FILENAME
read -r -p 'Retrieval date (YYYY-MM-DD): ' PROXYBENCH_RETRIEVAL_DATE

.venv/bin/python -m proxybench import-source \
  --input "data/downloads/$PROXYBENCH_ACCESSION/$PROXYBENCH_ORIGINAL_FILENAME" \
  --project-root . \
  --url "$PROXYBENCH_SOURCE_URL" \
  --accession "$PROXYBENCH_ACCESSION" \
  --filename "$PROXYBENCH_LOCAL_FILENAME" \
  --retrieval-date "$PROXYBENCH_RETRIEVAL_DATE" \
  --complete
```

The command copies unchanged bytes into `data/raw/` and records the source in `data/source-manifest.json`.
The manifest also records the supplied URL, accession number, retrieval date, and original filename.
Use `.venv/bin/python -m proxybench import-source --help` for optional filing metadata.
Continue with [manual target selection](#manual-target-selection), then [label review and acceptance](#label-review-and-acceptance).

Importing a local source does not authorize public redistribution.
Follow the [release guide](release.md) before publishing a project source archive.

## Manual target selection

Select one separately voted subject and its disclosed reporting scope.
Use the original source to choose byte ranges, an ordered sequence of source spans, and the target boundary.
Ranges use zero-based offsets with an excluded end position.
Keep the source encoding and SHA-256 hash with each selection.

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
- `development-examples.jsonl`: The ordered validation messages.
- `dataset-manifest.json`: Source selections, file hashes, split assignments, and label rules.

JSONL stores one JSON object on each line.
Each row has three messages in order: `system`, `user`, and `assistant`.
The assistant contains only the accepted fourteen-field label object.
The manifest uses schema `training-dataset-v1` and declares project-relative prompt and contract paths.
It needs no parent dataset or experiment output.

The reader rebuilds each user message from its source selection and compares it with the saved message.
It also checks source hashes, prompt bytes, row hashes, order, duplicate targets, labels, and exposure restrictions.
Without separate authorization for an exact release list, keep local source copies and label files out of Git.
See the [release guide](release.md) for raw-source review.
