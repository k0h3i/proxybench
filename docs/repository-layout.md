# Repository layout

ProxyBench uses a Python source layout, with package code under `src/`.
This separates importable code from tests and local files.
Source, review, record validation, and scoring helpers now implement parts of the pipeline.
The [benchmark implementation guide](benchmark-implementation.md) describes their interfaces.

## Package boundaries

The package follows the reviewed extraction architecture. Add behavior to the package that owns the corresponding stage. Shared data types belong in `schemas/`.

| Package | Responsibility |
|---|---|
| `proxybench.schemas` | Documents, record contexts, field evidence, vote records, and run manifests |
| `proxybench.sources` | Filing discovery, document retrieval, caching, and retrieval outcomes |
| `proxybench.normalization` | Text and table preparation with source mappings |
| `proxybench.extraction` | Record segmentation, extraction adapters, and mechanical validation |
| `proxybench.annotation` | Review packets, source-first labeling, and annotation history |
| `proxybench.evaluation` | Common normalization, scoring, comparison protocols, and error reports |
| `proxybench.execution` | Development schedules, isolated comparison execution, subprocess limits, capture, replay, and execution reports |
| `proxybench.training` | Later supervision generation, data preparation, and fine-tuning |

Categorization and proposal linking remain later stages. Their code will be added after their review gates. They cannot overwrite extracted source facts.

## Shared and local files

Shared files include source code, tests, public documentation, and example configuration. Local files include planning notes, downloaded filings, annotations, and model output. This separation keeps pilot work outside ordinary commits.

| Location | Git treatment | Contents |
|---|---|---|
| `notes/` | Ignored in full | Spec, review record, stage plans, and working decisions |
| `data/raw/` | Ignored | Original filing documents |
| `data/normalized/` | Ignored | Prepared text, tables, and source mappings |
| `data/packets/` | Ignored | Model inputs and source-first review packets |
| `data/annotations/` | Ignored | Labels, source audits, and annotation revisions |
| `data/manifests/` | Ignored | Source inventory, file hashes, and sampling decisions |
| `artifacts/runs/` | Ignored | Raw predictions and run metadata |
| `artifacts/models/` | Ignored | Model weights and trained adapters |
| `artifacts/reports/` | Ignored | Generated evaluation and calibration reports |
| `tests/fixtures/` | Eligible for tracking | Small synthetic or deliberately selected shareable examples |

The guides `data/README.md` and `artifacts/README.md` remain eligible for tracking. Empty local subdirectories are not stored by Git. Create the needed directories when preparing a stage.

Keep reusable operating documentation in `docs/`. Keep working notes in `notes/`. If a note supplies instructions needed by other contributors, write a shared guide instead of linking to the ignored note.


## Local note organization

When local notes exist, use `notes/README.md` as the starting point.
The index identifies current work and links to the relevant decisions.
Keep the current decision sheet and annotation guide under `notes/calibration/`.

| Local folder | Purpose |
|---|---|
| `notes/project/` | Scope, specification, and design review |
| `notes/calibration/` | Annotation guide, results, and decisions |
| `notes/benchmark/` | Benchmark decisions, frozen definitions, and implementation reports |
| `notes/handoffs/` | Context for later sessions and historical stage plans |
| `notes/scripts/` | Local preparation and inspection scripts |
| `notes/review-imports/` | Original review downloads |
| `notes/private/` | Local contact details and other private configuration |

Run local scripts from the repository root.
Use `notes/relocations.json` to resolve old paths in preserved generated records.
Keep this local organization outside ordinary commits through the existing ignore rule.
