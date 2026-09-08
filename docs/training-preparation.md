# Training preparation

The preparation tools map a bounded sample of modern N-PX XML and create provisional text examples.
They also support a local Qwen baseline and a disposable synthetic adapter test.
An adapter is a small set of trainable model parameters.
Preparation does not authorize training on filing labels, expanded historical annotation, independent evaluation, or publication.

Use `Qwen/Qwen3.5-9B` at revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
The [configuration](../configs/qwen35-preparation.json) retains the multimodal model and limits current inputs to text.
The model identity is fixed, while local fit and usable extraction still require measured results.
Keep source documents, labels, audit pages, weights, and generated output in ignored directories.

The [compact format prototype](compact-fragment-format.md) reduces repeated metadata and evidence in model inputs and targets.
It preserves original source content and reconstructs the existing record fields.
Its contract remains separate from the frozen baseline, and token measurements do not establish training-memory fit.
The current training worker uses Transformers and PEFT, not Unsloth.
The Ollama adapter has synthetic capture tests, but complete-fragment execution remains unmeasured.

## GPU runtime prerequisites

Install the pinned [GPU dependencies](../configs/requirements-gpu-preparation.txt) in the isolated model environment.
This dependency file targets Python 3.12 on Linux x86_64 with CUDA 12.
Qwen requires `flash-linear-attention` and `causal-conv1d` for its optimized linear-attention operations.
Use the official `causal-conv1d` wheel that matches Python, PyTorch, CUDA, and the platform.
The [official releases](https://github.com/Dao-AILab/causal-conv1d/releases) provide these compiled packages.
For this environment, the tested wheel uses Python 3.12, PyTorch 2.10, CUDA 12, Linux x86_64, and C++11 ABI `TRUE`.

Triton compiles GPU operations and their C launcher during execution.
The loader uses an existing C compiler or the pinned local `ziglang` compiler.
It requires a working Triton CUDA backend before loading model weights.
It also requires optimized operations and verifies their bindings in every linear-attention layer.
Each run saves these results and package versions in `runtime.json` and `optimized-layers.json`.
Missing prerequisites stop execution instead of silently selecting slower operations.

Use `proxybench.execution.speed_probe` under the resource supervisor for a separate synthetic speed measurement.
It generates at most 128 tokens and saves each token with its arrival time.
It reports the first-token delay and generation speed separately from model loading.
The first call can include kernel compilation, so repeat the same test to measure execution after compilation.
This short prompt does not establish complete extraction speed for long filing inputs.

## Source preparation

Use the existing SEC client and private identity for retrieval.
The [SEC access guidance](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data) requires a declared client identity and limits request rates.
The client makes sequential requests at least one second apart and stops after access denial.
Use a new cache root for this stage to preserve earlier retrieval logs.

`proxybench.sources.npx_xml.discover_attachments` selects XML attachments by their recorded document roles.
It excludes stylesheet representations and links outside the filing directory.
Preserve the index, attachment relationships, original bytes, URLs, retrieval dates, and hashes.
Do not assume that the primary attachment contains the votes.

The [mapping table](modern-npx-mapping.md) describes the supported subset of the official N-PX 3.1 specification.
The XML reader rejects entity declarations, external resolution, unknown namespaces, unsupported encodings, and oversized structures.
Default limits are 16 MiB per document, 32 nesting levels, 200,000 nodes, and 10,000 physical vote entries.
These are local preparation limits, not the SEC schema limits.

Prepare a manifest with these fields:

| Field | Contents |
|---|---|
| `version` | Frozen selection version |
| `split` | `development` |
| `group_id` | Group shared by related sources and renderings |
| `grouping_policy` | Duplicate, overlap, amendment, and representation rules |
| `filings` | At most four explicit filing selections |
| Filing `accession` | Original SEC accession string |
| Filing `selected` | Zero-based XML entry indices, at most 24 across the manifest |
| Filing `audit` | Selected indices for up to eight audit examples |
| Filing `primary`, `votes` | Objects with `path`, `sha256`, `document_id`, and `url` |

An XML entry is a candidate disclosure, not proof of an independent logical record.
Resolve uncertain subject boundaries and share-class overlap through source review.
Keep originals and amendments separate, but retain their grouping relationships.
Every exposed source group stays outside future untouched evaluation.

Run the builder with a new output path:

```bash
PYTHONPATH=src python3 -m proxybench.training.preparation MANIFEST.json artifacts/runs/NEW_RUN
```

The builder refuses an existing output directory and freezes the source manifest before rendering.
It produces mapping counts, rejection reasons, two source renderings, full targets, evidence locations, and an audit page.
Generated targets use only visible packet content.
The full `benchmark-v1` contract remains unchanged.

## Audit and admission

The audit page shows original XML and both renderings before a collapsed generated target.
Present a workload estimate before requesting user review time.
Include source orientation, inspection, corrections, and disagreement resolution in that estimate.
Modern audit time does not estimate historical annotation time.

Record explicit decisions and active review seconds separately from mechanical results.
`record_review` returns revised copies and never grants training admission.
An accepted review establishes human review status for the source entry and its two renderings.
The caller must preserve the exact review event and its prior manifest.

If a mapping rule fails, quarantine every preview generated by that rule.
Correct the rule, assign a new version, and repeat the audit before acceptance.
Do not relabel quarantined examples as accepted under the failed rule.
The later training gate must approve the frozen admission policy and source manifest.

## Exact token measurements

A token is a unit that the model reads or generates.
Measure the complete prompt and response through the pinned chat template.
Include the contract, source bundle, evidence wrappers, and response termination token.
Report over-limit examples without removing them from the denominator.

```bash
PYTHONPATH=src GPU_PYTHON -m proxybench.training.measure \
  RUN/previews MODEL_DIRECTORY docs/model-extraction-contract.md LENGTH_REPORT.json
```

`GPU_PYTHON` means the Python executable in the isolated model environment.
The mask selects response tokens for training loss and excludes prompt tokens.
`sequence` requires an exact shared prompt prefix and a supervised termination token.
`pad_sequence` excludes padding from loss and refuses truncation.

## Local model probes

Keep optional GPU packages outside the core package environment.
The initial runtime uses Python 3.12, Transformers 5.5.0, PyTorch 2.10.0, PEFT 0.18.1, bitsandbytes 0.49.2, and Accelerate 1.13.0.
The run records exact installed versions, model-file hashes, tokenizer files, processor files, and the chat template.
The [Transformers Qwen guide](https://huggingface.co/docs/transformers/v5.5.0/en/model_doc/qwen3_5) documents the full multimodal model class.

The proposed baseline uses four-bit NF4 weights with double quantization and bfloat16 computation.
Quantization stores base weights at reduced precision.
The worker retains vision weights and reports their actual parameter storage.
No image inputs enter this stage, and no visual performance claim follows.

The initial probe has one frozen prompt and one attempt for each of six accepted development inputs.
It records raw text and token IDs before validation or normalization.
Greedy decoding selects the highest-scoring next token, and thinking is explicitly disabled.
The maximum context is 16,384 tokens, with at most 8,192 generated tokens within that context.

The proposed synthetic test performs two optimizer updates with gradient accumulation of two device steps per update.
Each device step uses one complete synthetic sequence at the maximum context length, with no packing.
The adapters target language-model projection modules only, using rank 8, alpha 16, zero dropout, and AdamW at `0.0001`.
The worker records every actual target module and trainable parameter name.

The response loss uses checkpointed vocabulary projections in chunks of 128 positions.
This computes the complete response loss without retaining a full context-by-vocabulary output tensor.
The worker requires finite loss, finite adapter gradients, a nonzero adapter gradient, and changed adapter weights.
It also requires frozen base parameters and compares deterministic samples of their values before and after updates.

The worker saves a disposable adapter, clears its active tensors, reloads the saved tensors, and compares every tensor exactly.
It then performs a bounded generation with that reloaded adapter.
The test does not require loss to fall over two updates.
The [PEFT quantization guide](https://huggingface.co/docs/peft/developer_guides/quantization) explains the base preparation method.

## Resource supervision

Approve the operational allocation before starting model execution.
The proposed allocation is eight GPU hours across all attempts, with thirty minutes per loading, inference, optimizer, reload, or generation phase.
Downloads and environment setup are outside GPU execution time.
The supervisor records elapsed execution in a shared ledger and refuses concurrent runs that use that ledger.

The supervisor samples device memory, Linux available memory, process memory, and swap counters about once per second.
It requires at least 4 GiB of available host memory before starting.
It stops the process group after two consecutive samples below 1 GiB.
The readiness margin requires at least 2 GiB of free device memory at the observed peak.

Use the supervisor for both worker modes:

```bash
PYTHONPATH=src GPU_PYTHON -m proxybench.execution.resources \
  configs/qwen35-preparation.json SUPERVISION_DIRECTORY \
  --ledger EXECUTION_LEDGER.jsonl -- \
  GPU_PYTHON -m proxybench.training.gpu_probe baseline \
  configs/qwen35-preparation.json MODEL_DIRECTORY INPUT_DIRECTORY WORKER_OUTPUT
```

For the synthetic mode, replace `baseline` with `synthetic` and pass the synthetic sequence file as `INPUT_DIRECTORY`.
Every output directory must be new.
Use the same ledger across both modes and any failed attempts.
After a memory exhaustion, stop the candidate and report the failing operation.

Sampling cannot prevent sudden memory exhaustion and does not describe all Windows memory pressure.
Do not change WSL memory allocation automatically.
Keep framework peak memory separate from device-level observations.
Record loading, optimizer execution, adapter reload, and generation as separate phases.

## Readiness and release

Report mapping coverage, human acceptance, hardware fit, extraction usability, and quality separately.
Usability requires at least one complete response that passes the unchanged contract.
Six assisted development references cannot establish independent historical accuracy.
Missing GPU results or pending human review remain pending, even when synthetic software tests pass.

The intended public release includes the model and excludes the dataset.
Preserve the [base model license](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/LICENSE), provenance, and applicable supervision-output terms for the later release decision.
Model documentation can report aggregate methods and limitations without embedding private examples.
Adapter-only versus merged-weight packaging and publication remain later decisions.
