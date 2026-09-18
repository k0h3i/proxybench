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

Tracked project notes start at `notes/README.md`.
They include historical review imports and helper scripts.
Some evidence links point to ignored local data and artifacts, which a fresh clone does not contain.
Before public release, read [the reminder](../notes/project/before-going-public.md).
The current plan and label contract live under `notes/training/direct-sol-labeling/`.
Keep source files available to both the labeling agent and the review browser.

The [historical launcher](../src/proxybench/training/historical_run.py) owns each reviewed training schedule and its saved state.
The [live supervisor](../src/proxybench/execution/live.py) owns process cleanup, progress forwarding, and resource limits.
The [training-label scorer](../src/proxybench/evaluation/training_labels.py) keeps source values separate from origin and quotation diagnostics.
Read the [expanded guide](qwen35-4b-historical-expanded.md) before the expanded user-launched GPU run.
