# Local data

This directory stores local project data. Git ignores all contents except this guide. Downloaded filings and reviewed labels remain local until a separate dataset release decision.

Use these subdirectories:

- `raw/`: Original filing documents, preserved without edits.
- `normalized/`: Prepared text, tables, and source mappings.
- `packets/`: Model inputs and source-first review material.
- `annotations/`: Labels, source audits, and review history.
- `manifests/`: Source inventories, hashes, and sampling decisions.

Keep original submissions and amendments separate. Preserve retrieval failures alongside successful downloads in the source inventory. Treat initial calibration examples as development data.
