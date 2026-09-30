# Repository layout

ProxyBench keeps reusable Python code under `src/proxybench/`.
`python -m proxybench` provides the supported command interface.
The package requires Python 3.11 or later and has no mandatory runtime dependencies.
Project Python packages live in the `.venv/` at the repository root.
Native runtimes and run folders remain separate.
Follow the [preparation guide](preparation.md) for their installation.

| Area | Responsibility |
|---|---|
| `sources` | Bounded source acquisition and identity preservation |
| `annotation` | Historical source ranges, safe display, editable labels, and explicit acceptance |
| `training` | Label validation, self-contained datasets, sequence preparation, adapters, checkpoints, and conversion |
| `extraction` | Local inference and raw answer capture |
| `execution` | Process ownership, resource limits, and progress |
| `evaluation` | Selected-model scoring, bound review decisions, and report generation |
| `runstate.py` | Atomic run progress, one writer, input identities, and cumulative resource accounting |

## Storage

Keep portable recipes and the exact model prompt under `configs/`.
Keep shared guides under `docs/` and small synthetic behavior tests under `tests/`.
Keep ordinary source files in flat `data/raw/` and accepted training messages in `data/training-dataset/`.
Keep test sources in `data/raw/test/` and accepted test messages in `data/testing-dataset/`.
The test dataset folder contains only `test-examples.jsonl` and `dataset-manifest.json` after acceptance.
Its manifest retains the accepted review and audit evidence.
Keep protected test identities in `data/test-source-ledger.json`.
Their manifests preserve source identity, labels, and split restrictions.

Keep selected models under `artifacts/models/ProxyType-4B/` and `artifacts/models/ProxyType-9B/`.
Each folder contains `adapter/`, `model-bf16.gguf`, and `model-info.json`.
A selected model can also retain its reviewed report and audit under `evaluation/`.
Keep the pinned BF16 base under `artifacts/models/Qwen3.5-4B/`.
Adapter loading and base-model comparisons share that copy.
The base comparison GGUF stays in this folder as `model-bf16.gguf`.
Evaluation records its hash separately from the original checkpoint files.
[base-model.json](../configs/base-model.json) records its exact revision and file hashes.
The optional 9B base stays under `artifacts/models/Qwen3.5-9B/`.
[base-model-9b.json](../configs/base-model-9b.json) binds its exact revision, checkpoint files, and local model metadata.
The training `--model` parameter selects one supported pinned base without changing the default 4B recipe.
Data and model contents stay ignored, with their usage guides as tracked exceptions.
Private Git history stays unchanged during cleanup.

## Future runs

Create each run in a user-selected workspace outside `artifacts/`.
A run contains only the folders required by its operations.
`run.json` records identities, effective configuration, and progress.
Training creates `adapter/` and `checkpoints/`.
Conversion uses `exports/`, and evaluation uses `evaluation/answers.jsonl`, optional `review/`, and `report.json`.

Keep raw outputs until their intended review is complete.
A completed future run does not delete itself.
Only an explicitly selected final model can enter the retained model folder.
Do not preserve old campaign code or an experiment archive.
Read the [training guide](training.md), [inference guide](inference.md), and [release guide](release.md) for their separate gates.
