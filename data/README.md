# Local data

Keep complete original sources in the flat `raw/` directory.
Keep test sources separately in `raw/test/`.
Keep their locations, names, and hashes in `source-manifest.json`.
The `training-dataset/` directory contains `training-examples.jsonl`, `development-examples.jsonl`, and `dataset-manifest.json`.
The manifest binds exact messages to source selections, label rules, and the existing split.
The `testing-dataset/` directory contains only `test-examples.jsonl` and `dataset-manifest.json` after acceptance.
The test manifest retains accepted review records, source audits, and protection metadata.
Finalized loading needs no preparation folder or parent training dataset.
`test-source-ledger.json` protects test sources and their attachments from training and development use.

These files remain ignored by Git.
This usage guide is the tracked exception.
Follow the [dataset guide](../docs/dataset.md) for preparation and explicit acceptance.
Before selecting public source files, follow the [data notice](../DATA_NOTICE.md#source-redistribution).
A raw-source release does not authorize publishing accepted labels.
