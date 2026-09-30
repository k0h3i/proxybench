# Inference with ProxyType models

Inference means generating an answer from a model.
This guide covers ProxyType-4B and ProxyType-9B.

The supported task takes one marked voting target with its source context.
Prepare the fragment through the [dataset workflow](dataset.md).
The model returns the fourteen-field [label contract](label-contract.md).
It does not select every target in a complete filing.
Complete the [preparation guide](preparation.md) in the root `.venv/` before launching GPU work.

## Model formats

An adapter stores learned changes to the base weights.
Merging applies those changes to a copy of the base weights.
GGUF is the converted model format used by llama.cpp.
These are separate steps with different storage and runtime requirements.
A tokenizer converts text into tokens, the units a model processes.

```mermaid
flowchart LR
    B[Pinned base model] --> S[Separate adapter loading]
    A[Trained adapter] --> S
    B --> M[Temporary merged weights]
    A --> M
    M --> C[GGUF conversion]
    C --> G[Final model-bf16.gguf]
    G --> L[Local llama.cpp inference]
```

The file sizes below apply to ProxyType-4B.

| Form | Required files and tradeoffs |
|---|---|
| Separate adapter | About 82 MiB with tokenizer files, plus the matching base weights in `artifacts/models/Qwen3.5-4B/`. The adapter can be replaced independently. |
| Temporary merged model | About 7.9 GiB of base weights with adapter changes applied. Conversion needs temporary disk space and a compatible Python environment. |
| Final GGUF | About 7.9 GiB with its tokenizer information. The supported llama.cpp path needs no separate base weights or adapter loading. |

The Python input renderer uses the retained tokenizer files to preserve exact prompt tokens.
The GGUF also contains tokenizer information for its runtime.
Keep no duplicate merged-weight or tokenizer directories without a supported caller.
Delete temporary merged weights after successful conversion.

Merging removes separate adapter calculations.
Changing to llama.cpp also changes the inference engine and its GPU implementation.
Do not attribute every speed difference to merging or promise identical answers across formats.

## External runtime

Install llama.cpp with the [preparation guide](preparation.md#install-llamacpp-and-its-cuda-libraries) before GGUF inference.
The guide supplies the exact archive, checksum, native dependencies, runtime manifest, and CPU tests.
Use the shell variables from its [path setup](preparation.md#set-paths-and-obtain-model-inputs).
Python package installation alone does not install llama.cpp.

[configs/inference.json](../configs/inference.json) records the engine controls and runtime variables.
The pinned engine revision is `329b6160f513915f1c607dbfae3d5ce864a64a4f`, release `b10909-mix-bea84f7`.
A replacement build requires bounded loading and inference tests.
Keep installed environments and retained models outside run folders.

The default configuration selects ProxyType-4B.
For ProxyType-9B, copy `configs/inference.json` to `configs/inference-9b.json`.
Set `model` to `artifacts/models/ProxyType-9B/model-bf16.gguf` in the copy.
Set `tokenizer` to `artifacts/models/ProxyType-9B/adapter` in the copy.
Pass that copy with `--config` when running inference.
The `--model` option changes only the model path.

## Inputs and answers

Use the exact [system prompt](../configs/model-system-prompt.txt).
The dataset workflow saves prepared packets in `packet-set.json`.
Select one packet and save its `model_input` value as UTF-8 text in `fragment.txt`.
Decode the JSON string before saving it.
Keep exactly one `BEGIN MARKED TARGET` and `END MARKED TARGET` pair.
The command places this context in the user message without repeating the label policy.

Preserve the generation prefix, tokenizer template, response parameters, and token limits.
The server must not substitute another chat template.
Choose a run folder that does not exist yet.
The command creates it and rejects an existing folder.

For ProxyType-4B, run:

```bash
.venv/bin/python -m proxybench infer --model artifacts/models/ProxyType-4B/model-bf16.gguf --input fragment.txt --config configs/inference.json --run-dir ../proxybench-runs/inference-001
```

For ProxyType-9B, use the copied configuration with its matching model and tokenizer paths:

```bash
.venv/bin/python -m proxybench infer --model artifacts/models/ProxyType-9B/model-bf16.gguf --input fragment.txt --config configs/inference-9b.json --run-dir ../proxybench-runs/inference-9b-001
```

The command saves the raw answer before parsing or normalization.
If generation returns an answer, read `answer.json` in the run folder.
The `text` field contains the generated answer.
The `status` field records how generation ended.
A `true` value for `format_valid` requires complete generation and a valid answer structure.
The `format_error` field records any parsing or structure error.

A timeout or worker failure can prevent the command from saving `answer.json`.
If this file is missing, inspect records and logs under `capture/` in the run folder.
Keep failure details when generation times out or the answer violates the contract.
Do not silently repair malformed answers or infer unsupported values.

Inference binds its server to `127.0.0.1` and stops its owned process after work.
Keep process, memory, and input limits enabled.

For separate adapter loading and temporary conversion, use `.venv/bin/python -m proxybench export --help`.
The loader compares the retained base files with their pinned hashes and applies the local Safetensors adapter.
It does not download weights during model execution.
The adapter configuration alone is not proof that the correct revision loaded.
Use the bounded acceptance tests in the [training guide](training.md) before deleting working originals.
