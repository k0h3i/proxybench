# Retained models

Keep selected final models and the pinned base under `models/`.
The working model folder is `models/ProxyType-4B/`.
It contains the adapter, final inference GGUF, and portable `model-info.json`.
A selected model can also retain `evaluation/report.json` and `evaluation/agent-review-audit.json`.
These files record the reviewed results for that exact model.

Keep the original BF16 Qwen3.5-4B checkpoint under `models/Qwen3.5-4B/`.
Adapter loading and base-model comparisons share that copy.
The loader compares its files with [base-model.json](../configs/base-model.json) before use.
Keep run history, environments, base caches, and merged conversion intermediates outside this folder.

Use an external workspace for each future run.
Keep temporary model rollback copies only until the bounded loading tests pass.
Do not delete the last working originals before that gate.
Model files remain ignored by Git, with this guide as the tracked exception.
See the [inference guide](../docs/inference.md) and [release guide](../docs/release.md).
