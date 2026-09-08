# Repository layout

ProxyBench uses a Python source layout under `src/proxybench/`.
Shared documentation lives under `docs/`, and behavior tests live under `tests/`.
The current workflow labels local sources with Sol and uses the editable browser for human review.

| Package | Responsibility |
|---|---|
| `schemas` | Typed records and validation |
| `sources` | Bounded reading of local XML documents |
| `normalization` | Field-value normalization |
| `extraction` | Local model adapters and source bundles |
| `annotation` | Source display, editable review, bindings, and audits |
| `evaluation` | Reference validation and scoring |
| `execution` | General capture, replay, and resource limits |
| `training` | Sequence preparation, compact formats, and local GPU utilities |

## Local storage

Keep reusable code and small synthetic fixtures in Git.
Keep original sources and accepted labels under `data/`.
Keep model output and environments under `artifacts/`.
The Git ignore rules exclude these contents except their usage guides.

Local notes start at `notes/README.md`.
The current plan and label contract live under `notes/training/direct-sol-labeling/`.
Keep source files available to both the labeling agent and the review browser.
