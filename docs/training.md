# Training and evaluation

Training uses the private dataset and [portable recipe](../configs/training.json).
The base model is `Qwen/Qwen3.5-4B`, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
Complete the [preparation guide](preparation.md) before model execution.
It covers Python packages, the compiler, llama.cpp, native libraries, base weights, and private inputs.
Export and evaluation of a new adapter also require the authenticated converter installation in the [preparation guide](preparation.md#obtain-converter-source-for-export).
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
.venv/bin/python -m proxybench train --config configs/training.json --run-dir ../proxybench-runs/run-001
.venv/bin/python -m proxybench status --run-dir ../proxybench-runs/run-001
.venv/bin/python -m proxybench resume --run-dir ../proxybench-runs/run-001
```

Training validates the dataset and prepares sequences before loading the model.
It saves future checkpoints under `checkpoints/` and the completed adapter under `adapter/`.
Only trusted local checkpoints can restore optimizer state.
Interrupted writes cannot count as completed checkpoints.
An explicit resume continues validated saved work and preserves resource totals.
Deleting completed historical optimizer state intentionally gives up resuming that old run.

Resume requires the exact checkpoint named by a `CLEAN_STOP` journal.
The command rejects invalid journal metadata and changed input identities before it starts a worker.
The bounded worker authenticates checkpoint files and reads trusted state on the CPU before it loads the GPU model.
Existing `training-inputs.json` bytes stay unchanged during resume and rejected fresh training.
Rejected worker attempts still count toward resource limits.

`READY`, `BOUNDARY`, `UPDATING`, and `TRAINED` do not authorize ordinary resume.
A periodic checkpoint alone does not authorize rollback.
Final publication can still fail after the journal records `TRAINED`.
Finalization retry and rollback require a separate recovery-policy decision.

## Training measurements and display

An optimizer update applies gradients to the trainable parameters.
An epoch is one pass through the training examples.
The display counts optimizer updates and derives epoch progress from the recorded sample position.

The supervisor renders terminal progress and readable redirected summaries.
The terminal uses up to four live lines and shows metrics only after measurements arrive.
Batch settings appear once at startup or resume.
Redirected output uses one line every ten updates, at epoch boundaries, and when the phase changes.

Library banners and warnings remain in the worker capture logs, whose paths appear at startup.
Resume identity warnings remain visible.
If training fails, the supervisor also displays up to twelve final lines from the captured error output.
Display refreshes do not change safety monitoring, journal writes, or checkpoint frequency.

Worker events record preparation, loading, updates, checkpoint stages, and final publication.
The supervisor charges the complete attempt, including failed attempts and startup.
Token rates use completed loop intervals and separate all sequence tokens from supervised response tokens.
The final loop summary includes journal writes, monitoring, event emission, and checkpoint work.
The recent display rate excludes event emission and states that exclusion beside its window.
The first update stays separate from later updates because compilation can affect its duration.
Ordinary computation measurements use host wall time and do not isolate asynchronous GPU kernel time.
Separate bounded profiling can enable device events through `PROXYBENCH_PROFILE_CUDA=1`.
This mode uses the existing safety synchronization and records event overhead.

The runtime record binds executing code, packages, model, tokenizer, optimizer, hardware, and kernel configuration.
Resume preserves earlier records and records the current attempt separately.
A missing or changed runtime identity requires new acceptance before an exact-continuation claim.
Even a matching identity does not establish CUDA continuation by itself.

At 100 percent of updates, final validation and publication can still be active.
The display reports success only after publication, the worker result, and the worker process succeed.
The [training acceptance guide](training-acceptance.md) describes separate loss and continuation checks.
The [Trainer experiment guide](trainer-experiment.md) describes the separate candidate and its pending adoption tests.
The [implementation evidence](training-implementation.md) records CPU results and unresolved policy decisions.

## Selected-model evaluation

Evaluation scores one selected model against development references.
It does not train a model.
Use the trained run metadata or give an existing model and dataset explicitly.
When a trained adapter needs conversion, evaluation uses the shared export path.
Evaluation loads the merged GGUF once for the remaining examples in each attempt.
Each example uses a separate request with prompt reuse disabled.
The worker saves raw answers before it starts the next example.

```bash
.venv/bin/python -m proxybench evaluate --run-dir ../proxybench-runs/run-001 --config configs/inference.json
.venv/bin/python -m proxybench evaluate --run-dir ../proxybench-runs/evaluation-001 --model artifacts/models/ProxyType-4B/model-bf16.gguf --dataset data/training-dataset --config configs/inference.json
.venv/bin/python -m proxybench evaluate --run-dir ../proxybench-runs/evaluation-001 --report-only
```

The evaluation folder retains raw answers, token IDs when available, generation status, and terminal failures.
Timeouts and malformed answers stay in the scoring denominator.
Missing answers make the evaluation incomplete.
Invalid reference labels prevent a valid accuracy report.
Field scoring keeps source-value errors separate from origin and quotation diagnostics.

One session contains the worker and its model server.
Session captures and resource records live under `evaluation/capture/session-*/`.
The session uses the remaining run allowance and preserves startup, request, and memory limits.
The per-example phase limit includes prompt preparation and token comparison.
Loading and cleanup count once toward the cumulative allowance.

If a request fails, evaluation saves that failure and stops the session.
Unattempted examples remain incomplete.
Use `resume` to process the remaining examples after the failure cause is resolved.
A user stop leaves unfinished requests eligible for resume.
Resume recovers completed captures before it requires model files.
Actual GPU speed improvement and answer comparisons remain pending user-launched validation.

When semantic scoring needs review, the command returns pending status and prints the review location.
Review the exact source cells, reference labels, and new answer in the browser.
Import the exported decisions to complete the report.

```bash
.venv/bin/python -m proxybench review-import --run-dir ../proxybench-runs/evaluation-001 --decisions ../review-decisions.json
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

## Preparation stage before GPU work

Follow the [preparation guide](preparation.md) for installation and CPU acceptance.
The implementing agent completes those steps before handing over a GPU command.
Environment preparation does not change the source-labeling purpose of `.venv/bin/python -m proxybench prepare`.

## Bounded user-launched load command

Run this command only after preparation passes.
Use a new run folder and the prepared runtime and retained model locations.
The user starts this GPU work.

```bash
export PROXYBENCH_RUNTIME="$HOME/.local/share/proxybench/runtime/llama-329b6160"
export PROXYBENCH_CUDA_LIB="$PROXYBENCH_RUNTIME"
.venv/bin/python -m proxybench validate-runtime --run-dir ../proxybench-runs/model-load-001
```

The command loads the pinned adapter and final GGUF and requests one synthetic text and one synthetic HTML-context answer from each.
It performs no training updates.
The adapter phase allows at most 900 seconds, and the GGUF phase allows at most 600 seconds.
Before the test, make old run directories and artifact environments unavailable through a reversible move.
Use `.venv/`, the independent native runtime, and the retained base in `artifacts/models/Qwen3.5-4B/`.
If either load fails, keep the protected originals and fix the supported path.

For future export, first complete the [converter preparation](preparation.md#obtain-converter-source-for-export).
Then run export with the prepared variables:

```bash
.venv/bin/python -m proxybench export --adapter artifacts/models/ProxyType-4B/adapter --config configs/training.json --run-dir ../proxybench-runs/export-001
```

The converter uses the same Python environment and its bundled GGUF code.
After selecting a future trained run with a completed export, promote its retained formats explicitly:

```bash
.venv/bin/python -m proxybench promote --run-dir ../proxybench-runs/run-001 --name NewModelName-4B
```

Promotion rejects an existing model name and copies only the supported model formats and metadata.
It does not grant public-release approval or claim that a new model passed loading tests.
