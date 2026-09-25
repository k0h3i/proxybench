# Retained models

Keep only selected final models under `models/`.
Each model folder contains its adapter, final inference GGUF, and portable `model-info.json`.
The working model folder is `models/ProxyType-4B/`.
Do not store run history, generated reports, environments, base caches, or merged conversion intermediates here.

Use an external workspace for each future run.
Keep temporary model rollback copies only until the bounded loading tests pass.
Do not delete the last working originals before that gate.
Model files remain ignored by Git, with this guide as the tracked exception.
See the [inference guide](../docs/inference.md) and [release guide](../docs/release.md).
