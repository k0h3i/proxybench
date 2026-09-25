# Local data

Keep complete original sources in the flat `raw/` directory.
Keep their locations, names, and hashes in `source-manifest.json`.
The `training-dataset/` directory contains `training-examples.jsonl`, `development-examples.jsonl`, and `dataset-manifest.json`.
The manifest binds exact messages to source selections, label rules, and the existing split.

These files remain ignored by Git.
This usage guide is the tracked exception.
Follow the [dataset guide](../docs/dataset.md) for preparation and explicit acceptance.
Follow the [release guide](../docs/release.md) before selecting public raw files.
A raw-source release does not authorize publishing accepted labels.
