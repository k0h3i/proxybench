# Training and evaluation

Training uses the private dataset and [portable recipe](../configs/training.json).
The base model is `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Download that revision into an external cache.
Keep optional model libraries in an external environment with the [recorded dependencies](../configs/requirements-training.txt).
The user starts GPU commands.

## Recipe and inputs

The retained final model completed 660 updates over two epochs with seed 42.
The recipe uses batch size 1, accumulation 1, learning rate 0.0001, zero weight decay, and gradient clipping at 1.0.
LoRA trains small weight changes while the base model stays fixed.
Its rank is 8, alpha is 16, and dropout is zero.

Each training example contains system, user, and assistant messages.
The system message must match [model-system-prompt.txt](../configs/model-system-prompt.txt) byte for byte.
Sequence preparation masks the prompt so loss applies to the assistant response.
The total context limit is 5,120 tokens, with 3,328 input tokens and 1,792 response tokens.
A token is a unit of text processed by the model.
Reject an oversized example rather than silently truncating accepted context or labels.

## Run ownership and resume

Use one new external folder for each run.
The commands reject existing destinations and mismatched input identities.
One writer owns each run folder.
The run metadata records input identities, effective configuration, progress, and cumulative resource usage.

```bash
python -m proxybench train --config configs/training.json --run-dir ../proxybench-runs/run-001
python -m proxybench status --run-dir ../proxybench-runs/run-001
python -m proxybench resume --run-dir ../proxybench-runs/run-001
```

Training validates the dataset and prepares sequences before loading the model.
It saves future checkpoints under `checkpoints/` and the completed adapter under `adapter/`.
Only trusted local checkpoints can restore optimizer state.
Interrupted writes cannot count as completed checkpoints.
An explicit resume continues validated saved work and preserves resource totals.
Deleting completed historical optimizer state intentionally gives up resuming that old run.

## Selected-model evaluation

Evaluation scores one selected model against development references.
It does not train a model.
Use the trained run metadata or give an existing model and dataset explicitly.
When a trained adapter needs conversion, evaluation uses the shared export path.

```bash
python -m proxybench evaluate --run-dir ../proxybench-runs/run-001 --config configs/inference.json
python -m proxybench evaluate --run-dir ../proxybench-runs/evaluation-001 --model artifacts/models/ProxyType-4B/model-bf16.gguf --dataset data/training-dataset --config configs/inference.json
python -m proxybench evaluate --run-dir ../proxybench-runs/evaluation-001 --report-only
```

The evaluation folder retains raw answers, token IDs when available, generation status, and terminal failures.
Timeouts and malformed answers stay in the scoring denominator.
Missing answers make the evaluation incomplete.
Invalid reference labels prevent a valid accuracy report.
Field scoring keeps source-value errors separate from origin and quotation diagnostics.

When semantic scoring needs review, the command returns pending status and prints the review location.
Review the exact source cells, reference labels, and new answer in the browser.
Import the exported decisions to complete the report.

```bash
python -m proxybench review-import --run-dir ../proxybench-runs/evaluation-001 --decisions ../review-decisions.json
```

Decisions bind to exact answers, references, sources, model, prompt, and runtime configuration.
Identical repeated imports are safe, while conflicting or stale imports fail.
The report is generated automatically when its required inputs are ready.
Complete scoring inputs can regenerate the report with the model and inference runtime unavailable.
New generation requires new review decisions where semantic review applies.
Regeneration does not promise identical historical answers or scores.

## GPU acceptance and promotion

Prepare bounded model-load tests before deleting the last working originals.
Test the portable adapter with its pinned base and the GGUF without the old base directory.
Use synthetic text and HTML fragments with old run folders and artifact environments unavailable.
The user must launch these GPU tests.
Until they pass, model migration remains pending.

Keep future run output outside `artifacts/`.
Promote only an explicitly selected final model into `artifacts/models/`, with overwrite protection.
Keep the adapter, final GGUF, required tokenizer files, and portable model metadata.
Public release requires the separate [release gate](release.md).

## External environment and bounded load command

The recorded optional environment uses Python 3.12.14.
Install that Python version and a C/C++ compiler outside the repository before the GPU gate.
On Linux, the system `build-essential` package can provide the compiler.
For a nonstandard compiler, set `CC` and `CXX` to its supported external paths.
Keep the old environment and toolchain until their replacements pass acceptance.

```bash
python3.12 -m venv "$HOME/.local/share/proxybench/venv"
source "$HOME/.local/share/proxybench/venv/bin/activate"
python -m pip install -r configs/requirements-training.txt
python -m pip install -e .
export PROXYBENCH_BASE_CACHE="$HOME/.cache/proxybench/base-models"
export PROXYBENCH_RUNTIME="$HOME/.local/share/proxybench/runtime/llama-329b6160"
export PROXYBENCH_CUDA_LIB="/usr/local/lib/ollama/cuda_v12"
python -m proxybench validate-runtime --run-dir ../proxybench-runs/model-load-001
```

The command loads the pinned adapter and final GGUF and requests one synthetic text and one synthetic HTML-context answer from each.
It performs no training updates.
The adapter phase allows at most 900 seconds, and the GGUF phase allows at most 600 seconds.
Before the test, make old run directories, base-model directories, and artifact environments unavailable through a reversible move.
Use only the external environment, runtime, and base cache for this test.
If either load fails, keep the protected originals and fix the supported path.

For future conversion, place the converter source outside the repository:

```bash
git clone https://github.com/unslothai/llama.cpp.git ../proxybench-llama-cpp
git -C ../proxybench-llama-cpp checkout 329b6160f513915f1c607dbfae3d5ce864a64a4f
export PROXYBENCH_CONVERTER_SOURCE="$(realpath ../proxybench-llama-cpp)"
python -m proxybench export --adapter artifacts/models/ProxyType-4B/adapter --config configs/training.json --run-dir ../proxybench-runs/export-001
```

The converter uses the same Python environment and its bundled GGUF code.
After selecting a future trained run with a completed export, promote its retained formats explicitly:

```bash
python -m proxybench promote --run-dir ../proxybench-runs/run-001 --name NewModelName-4B
```

Promotion rejects an existing model name and copies only the supported model formats and metadata.
It does not grant public-release approval or claim that a new model passed loading tests.
