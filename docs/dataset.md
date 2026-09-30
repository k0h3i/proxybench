# Sources and accepted labels

Fine-tuning used 330 training examples for both ProxyType-4B and ProxyType-9B.
Another 90 examples served as development references.
Together, the training and development examples came from 36 source files.
The development examples contain known project exposure.

The acquisition and review steps below cover training and development examples.
For test references, start with the [protected test workflow](#protected-test-workflow) before acquiring sources.

## Acquisition and source storage

Keep complete training and development source files in the flat `data/raw/` directory.
Keep protected test sources in `data/raw/test/`.
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
Before publishing a project source archive, follow the [data notice](../DATA_NOTICE.md#source-redistribution).

## Manual target selection

Select one separately voted subject and its disclosed reporting scope.
Use the original source to choose byte ranges, an ordered sequence of source spans, and the target boundary.
Ranges use zero-based offsets with an excluded end position.
A SHA-256 hash identifies exact file bytes.
Keep the source encoding and SHA-256 hash with each selection.

Text spans preserve decoded characters and whitespace.
HTML spans preserve ordered cells, empty cells, and row or column spans.
The [`historical-cells-v1`](../src/proxybench/annotation/historical.py) renderer decodes character references once and collapses whitespace inside HTML cells.
Generated block labels and omitted-range markers are not source text.
Shared-row subtargets remain unsupported.
Defer a target when complete adjacent blocks cannot represent it.

A selection contains `packet_id`, `source_path`, `source_sha256`, `accession`, `group_id`, `split`, `encoding`, `spans`, and `target`.
It also records `boundary_review`, `reviewer`, and the derivation or source-association metadata required by the label validator.
Each span contains its start, end, and kind: `text`, `row`, `cell`, or `block`.
Standalone HTML cells can provide context but cannot define shared-row targets.
Use project-relative paths without parent traversal.

A JSON array is an ordered list.
Save selections as a JSON array in `data/selections.json`.
Write each span as `[start, end, kind]` and the target as `[start, end]`.
The target must cover complete adjacent blocks.

## Label review and acceptance

### Assign sources to splits

Save the source assignments in `data/assignments.json`.
Use accession numbers as object keys.
Copy existing assignments from the retained dataset manifest before adding entries.
This example uses a fictional accession:

```json
{
  "0000000001-20-000001": {
    "split": "development",
    "group_id": "example-family",
    "development_exposed": true
  }
}
```

Each assignment has these fields:

| Field | Meaning |
|---|---|
| `split` | `training`, `development`, `test`, or `excluded`. Ordinary review selects only training or development examples. |
| `group_id` | The stable source-group identifier shared by related filings. |
| `development_exposed` | `true` when the source group has known development exposure. |

Every selected accession needs an assignment that matches its selection's `split` and `group_id`.
Keep each source group in one split.
Never move an exposed source group into training.

### Prepare and review labels

Use a new directory for each review.
Prepare the selected source packets:

```bash
.venv/bin/python -m proxybench prepare \
  --project-root . \
  --selections data/selections.json \
  --assignments data/assignments.json \
  --output data/label-review
```

Open `data/label-review/index.html` in your browser.
The page presents source context before label suggestions.
Give the labeling agent the original source context and the [label contract](label-contract.md).
Use the [canonical prompt](../configs/model-system-prompt.txt) as the exact system message.
Put source context in the source message without another copy of the labeling policy.

Keep corrections and uncertainty visible for review.
After reviewing the labels, record your approval before exporting the dataset.
Select `Download review draft` and save the downloaded file as `data/review-export.json`.

### Approve labels and export the dataset

Compute the exact review file's hash:

```bash
sha256sum data/review-export.json
```

After reviewing, complete `data/approval.json` with these fields:

```json
{
  "decision": "ACCEPTED",
  "export_sha256": "REPLACE_WITH_REVIEW_SHA256",
  "reviewer": "YOUR_NAME",
  "accepted_at": "YYYY-MM-DD",
  "accepted_packet_ids": ["example-target"]
}
```

Record your name and approval date.
Use the computed hash and the approved packet IDs from `packet-set.json` in the review directory.
Their order determines row order within each split.
If you change the review, record a new approval.
Rejected and unreviewed labels cannot enter the dataset.

Export to a new dataset directory:

```bash
.venv/bin/python -m proxybench accept \
  --project-root . \
  --review-dir data/label-review \
  --review data/review-export.json \
  --approval data/approval.json \
  --assignments data/assignments.json \
  --output data/training-dataset-next
```

The command refuses an existing output directory.
Keep the retained `data/training-dataset/` unchanged during this review.
Use `.venv/bin/python -m proxybench prepare --help` or `.venv/bin/python -m proxybench accept --help` for command help.

## Protected test workflow

Test preparation requires the retained training dataset and source inventory.
Keep the reviewed [test configuration](../configs/testing.json) unchanged.
It defines filing-year coverage, source requirements, and the scoring version.
Never use test answers or scores to select source targets, prompts, or thresholds.

### Save the exposure baseline

Before acquiring test candidates, save the existing source inventory and dataset manifest.
Use unused paths for the baseline and audit file:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from proxybench.annotation.testing import snapshot_test_baseline
from proxybench.runstate import atomic_json

root = Path.cwd()
audit_path = root / "data/test-audit.json"
if audit_path.exists():
    raise FileExistsError(audit_path)
audit = snapshot_test_baseline(root, root / "data/test-baseline")
atomic_json(audit_path, audit)
PY
```

The helper copies the existing manifests and records their paths and hashes in `data/test-audit.json`.
Keep the baseline dataset unchanged through preparation and acceptance.
Never replace the project's known exposure history with empty or synthetic manifests.

### Import and select test sources

Follow the source-selection, download, and inspection steps in the [acquisition procedure](#acquisition-and-source-storage).
For test imports, enter the filing date, form, and registrant identifier from the filing details.
A CIK identifies an SEC registrant.
Use this command instead of the ordinary import:

```bash
read -r -p 'Local lowercase filename: ' PROXYBENCH_LOCAL_FILENAME
read -r -p 'Retrieval date (YYYY-MM-DD): ' PROXYBENCH_RETRIEVAL_DATE
read -r -p 'SEC filing date (YYYY-MM-DD): ' PROXYBENCH_FILING_DATE
read -r -p 'Form (N-PX or N-PX/A): ' PROXYBENCH_FORM
read -r -p 'Registrant CIK: ' PROXYBENCH_CIK

.venv/bin/python -m proxybench import-source \
  --input "data/downloads/$PROXYBENCH_ACCESSION/$PROXYBENCH_ORIGINAL_FILENAME" \
  --project-root . \
  --url "$PROXYBENCH_SOURCE_URL" \
  --accession "$PROXYBENCH_ACCESSION" \
  --filename "$PROXYBENCH_LOCAL_FILENAME" \
  --retrieval-date "$PROXYBENCH_RETRIEVAL_DATE" \
  --filing-date "$PROXYBENCH_FILING_DATE" \
  --form "$PROXYBENCH_FORM" \
  --cik "$PROXYBENCH_CIK" \
  --storage test \
  --complete
```

The command stores unchanged bytes in `data/raw/test/` and records their filing identity in the source manifest.
Prepare test selections using the same byte-range format as [ordinary selections](#manual-target-selection).
Save them in `data/test-selections.json` with `split` set to `test`.
Add `source_url`, `cik`, `filing_date`, `family_id`, and `independence_review` to each selection.
The URL, CIK, filing date, accession, path, and hash must match the imported source.

Record the amendment and copied-content review in `independence_review`.
Use distinct filings, source groups, registrants, and provider families without known project exposure.
For years listed in `standardized_years`, set `standardized_format` to `true`.
Include supporting passages from the selected cells in `standardization_evidence`.

### Audit provider families

A provider family contains filings from one fund provider.
The test configuration requires an audit of provider families before label review.
Save that audit in `data/test-family-audit.json` with these fields:

| Field | Content |
|---|---|
| `schema` | `test-family-audit-v1`. |
| `prior_dataset`, `prior_inventory` | The unchanged baseline bindings from `data/test-audit.json`. Each contains `path` and `sha256`. |
| `baseline_sources` | One entry per prior source, with `path`, `sha256`, `family_ids`, and `evidence`. |
| `baseline_assignments` | Entries keyed by every prior accession, each with `family_ids` and `evidence`. |
| `candidates` | Entries keyed by candidate accession, each with `family_id`, `cik`, `source_sha256`, and `evidence`. |

Use provider identifiers such as `family-example-provider`.
Each prior `family_ids` value and candidate `source_sha256` value is a nonempty list.
Candidate families must differ from prior families and from every other selected test family.
Each `evidence` value is a nonempty array.
Each evidence item contains `source` with `path` and `sha256`, `bytes` with `[start, end]`, and `slice_sha256`.
The slice hash identifies the original bytes in that range.
Candidate evidence must come from the imported filing.

Use the [family audit validator](../src/proxybench/annotation/testing.py) for the complete format and evidence rules.
Add a `family_audit` entry to `data/test-audit.json` with the audit file's project-relative `path` and exact `sha256`.
Preserve its existing baseline entries.

### Prepare, review, and accept test references

Prepare the protected review in a new project-local directory:

```bash
.venv/bin/python -m proxybench prepare-test \
  --project-root . \
  --selections data/test-selections.json \
  --protocol configs/testing.json \
  --audit data/test-audit.json \
  --output data/test-review
```

Preparation records the protected identities in `data/test-source-ledger.json` before exposing packets for review.
Keep this ledger throughout future preparation, training, and evaluation.
Open `data/test-review/index.html` and review every packet against its source.
Save the browser export as `data/test-review-export.json`.
Compute the review and preparation hashes:

```bash
sha256sum data/test-review-export.json data/test-review/preparation.json
```

After reviewing, use the [approval format above](#approve-labels-and-export-the-dataset) in `data/test-approval.json`.
Add `preparation_sha256` with the exact hash of `data/test-review/preparation.json`.
Set `accepted_packet_ids` to every ID in that file's `packet_ids` list, in the same order.
Approve every prepared packet in the recorded order.

Export to a new test dataset directory:

```bash
.venv/bin/python -m proxybench accept-test \
  --project-root . \
  --review-dir data/test-review \
  --review data/test-review-export.json \
  --approval data/test-approval.json \
  --output data/testing-dataset-next
```

The output contains `test-examples.jsonl` and `dataset-manifest.json` under the [`test-dataset-v1`](../src/proxybench/annotation/testing.py) schema.
The manifest retains the accepted review, preparation, and audit metadata.
The training reader rejects this test-only format.
Use `.venv/bin/python -m proxybench prepare-test --help` or `.venv/bin/python -m proxybench accept-test --help` for command help.

## Dataset format and boundaries

`data/training-dataset/` contains the following files:

- `training-examples.jsonl`: The ordered training messages.
- `development-examples.jsonl`: The ordered validation messages.
- `dataset-manifest.json`: Source selections, file hashes, split assignments, and label rules.

JSONL stores one JSON object on each line.
Each row has three messages in order: `system`, `user`, and `assistant`.
The assistant contains only the accepted fourteen-field label object.
The manifest uses the [`training-dataset-v1`](../src/proxybench/training/dataset.py) schema and declares project-relative prompt and contract paths.
It needs no parent dataset or experiment output.

The reader rebuilds each source message from its selection and compares it with the saved message.
It also checks source hashes, prompt bytes, row hashes, order, duplicate targets, labels, and exposure restrictions.
Without separate authorization for an exact release list, keep local source copies and label files out of Git.
See the [data notice](../DATA_NOTICE.md#source-redistribution) for source redistribution requirements.
