# Direct labels for fine-tuning

Sol reads a supplied label contract and a marked target from local source files.
The source includes the primary XML attachment that identifies the reporting fund.
Each agent writes `draft.json` and separate field notes to its assigned directory.
Keep those original drafts unchanged during human review.

`proxybench.training.labels.prepare_review` loads those drafts into the existing editable browser.
It requires the exact `source.txt` and `input.txt` that the agent read.
It keeps source identities and supporting notes outside the training target.
Use the [local review guide](local-review.md) for the browser controls.

## Edit and accept labels

The training review shows plain answer text and labeled controls for structured fields.
Fund details, identifiers, and vote components have separate editable sections.
Select `No value` when information is missing.
An empty text box with `No value` cleared means empty text.
Nested fields have their own availability, origin, and original-wording controls.
The downloaded file preserves the structured labels automatically.

Read the marked source before revealing Sol's draft.
Correct the fields and record reasons for changes.
Mark each completed packet reviewed, then download the review file.
The completion checkbox records completion, not acceptance for training.

After the user explicitly accepts the final labels, record that acceptance in a separate JSON file.
The acceptance record contains these properties:

- `decision`: `ACCEPTED`.
- `export_sha256`: SHA-256 hash of the exact downloaded review file.
- `accepted_packet_ids`: IDs of the packets that the user accepted.
- `reviewer`: The person who accepted the labels.
- `accepted_at`: The time of that acceptance.

A SHA-256 hash identifies the exact file bytes.
A changed review file needs acceptance for its new hash.
Do not create an acceptance record from review completion or agent approval alone.

## Export accepted examples

Run this command after explicit acceptance:

```bash
PYTHONPATH=src python3 -m proxybench.training.labels \
  --run artifacts/runs/direct-sol-labels-v1 \
  --review /path/to/downloaded-review.json \
  --approval /path/to/acceptance.json \
  --output /path/to/accepted-labels.jsonl
```

The exporter refuses changed exports, unreviewed packets, invalid label values, and existing output files.
Each JSONL line contains one `messages` example.
The user message contains the contract and exact source context.
The assistant message contains only the approved label fields.
A separate manifest records acceptance, source hashes, source groups, and raw draft hashes.

The first four targets remain exposed development data.
Their labels do not establish accuracy on unseen filings or a sufficient training dataset.
The exporter does not launch optimizer training.
