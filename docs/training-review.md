# Training loop review

The recommendations have merit, especially the changes to resume validation and runtime measurements.
Keep the current loop for the first patch.
Hugging Face Trainer is a reasonable experiment for reducing maintenance when batching and scheduling become necessary.
No evidence from this review establishes a speed or extraction-quality improvement from a trainer migration.

This review covers commit `038e381` and the installed packages on September 26, 2026.
An independent Astra review challenged the findings.
The branch is `review/training-loop-trainer`.
This branch changes documentation only.
No GPU training, profiling, model comparison, or recovery-policy change formed part of this review.

## Baseline evidence

The [recipe](../configs/training.json) and [worker](../src/proxybench/training/runtime.py) agree with the proposed numerical baseline.
The root `.venv/` uses Python 3.12.14, and every installed requirement matches its pinned version.
`pip check` reports no broken requirements.
Package import success does not establish that the patched GPU training path works.

| Item | Evidence and qualification |
|---|---|
| Model | Qwen/Qwen3.5-4B at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The retained adapter configuration and model metadata agree. |
| Dataset and order | CPU dataset validation finds 330 training and 90 development examples. The current ordering function produces 660 updates for two epochs and seed 42. |
| Optimizer | AdamW, constant `1e-4`, zero weight decay, default betas `(0.9, 0.999)`, default epsilon `1e-8`, clipping at 1.0. No scheduler exists. |
| Batch | Physical batch 1 and accumulation 1. The worker explicitly rejects other values. |
| Adapter | Rank 8, alpha 16, dropout 0, no bias training. The actual Safetensors header contains 496 tensors across 248 projections, totaling 16,232,448 parameters. |
| Precision | The worker loads BF16 base weights without four-bit quantization. The retained adapter tensors are F32. BF16 does not mean every trainable tensor uses BF16. |
| Limits | Total 5,120, input 3,328, response 1,792 tokens. Input counts include the system message and chat template. A token is a unit of model text. |
| Actual training lengths | Input 1,871–2,204, response 676–1,402, and combined 2,659–3,598 tokens. These ranges come from CPU preparation with the retained tokenizer. |
| Actual development lengths | Input 1,938–1,993, response 669–966, and combined 2,644–2,950 tokens. No training or parameter selection used these measurements. |
| Installed stack | Torch 2.12.1, Transformers 5.5.0, TRL 0.24.0, Unsloth 2026.9.4, Unsloth Zoo 2026.9.3, PEFT 0.21.0, Accelerate 1.15.0. |
| Retained evidence | Adapter, tokenizer, evaluation, audit, and load-test files match their hashes in `model-info.json`. This review did not rehash the full GGUF or base weights. |
| Historical recipe | Retained metadata records 660 updates and a recipe identity. It does not retain every historical runtime choice or optimizer state. Current defaults alone cannot prove every historical setting. |

The retained evaluation already reports 89 of 90 targets with correct source values and 1,259 of 1,260 correct fields.
It records valid formats for all 90 targets, one derivation error, and AI review against source context.
These are exposed development results, not an independent test result.
The retained load-test record also reports completed adapter and GGUF loading tests.
The [model card](../MODEL_CARD.md) still says loading tests are pending, so that sentence needs a separate documentation correction.

## Classification of the recommendations

The table uses the five requested classifications.
Some sections contain both an existing protection and a proposed experiment.
Those cases appear in separate rows to avoid assigning one status to different claims.

| Proposal | Classification | Repository finding and decision |
|---|---|---|
| 1. Improve the current trainer before replacing it | Intentional tradeoff worth reconsidering | The computational path already avoids full output-score allocation through a fused loss. Retain it while repairing resume validation. Reconsider Trainer when its generic machinery removes measurable maintenance work. |
| 2. Preserve the supplied baseline | Already addressed in current code | The current numerical recipe agrees. Record implicit optimizer choices before a future comparison. The artifact evidence has the limits stated above. |
| 3. Preserve prompt boundaries, masks, accepted answers, termination, and length rejection | Already addressed in current code | `sequences.py` and `preparation.py` enforce these rules. Oversized examples fail without truncation. |
| 3. Preserve adapter-only training, disabled cache, base identity, RNG restoration, and resource accounting | Already addressed in current code | The worker, trajectory code, checkpoint code, and supervisor implement these protections. CPU continuation tests cover a small model. RNG means random-number generator. |
| 4. Move resume checks before expensive setup | Confirmed issue | The worker loads the model, hashes base tensors, attaches the adapter, and creates the optimizer before journal eligibility checks. GPU-related imports also precede these checks. |
| 4. Validate journal status and required fields | Confirmed issue | `previous['checkpoint']` precedes status validation. Ordinary `BOUNDARY` and `UPDATING` records lack that key. Other malformed fields also produce incidental parser or key errors. |
| 4. Preserve existing input records on rejected resume | Confirmed issue | The worker unconditionally writes `training-inputs.json` before rejecting an invalid resume or an existing training run. |
| 5. Measure checkpoint stages | Experiment requiring measurement | Serialization, read-back, tensor comparisons, hashing, synchronization, and publication are real operations. Their runtime shares remain unknown. |
| 5. Separate resume checkpoints from portable artifacts | Intentional tradeoff worth reconsidering | Each periodic checkpoint duplicates adapter weights and tokenizer files. Removing copies changes its artifact contract and requires explicit approval. |
| 5. Add explicit rollback recovery | Intentional tradeoff worth reconsidering | Ordinary resume intentionally requires the exact `CLEAN_STOP` checkpoint. Periodic saves do not authorize recovery after an interrupted update. Preserve that rule in the first patch. |
| 6. Measure full runtime, useful tokens, and memory peaks | Confirmed issue | Detailed measurements are missing. Update timing excludes several costs, and ETA uses those incomplete durations. The supervisor already records total worker elapsed time. |
| 6. Separate startup from steady-state measurements | Experiment requiring measurement | Startup, compilation, and periodic saves can change averages. Record them separately before comparing implementations. |
| 7. Audit durable logging and device synchronization | Experiment requiring measurement | Two ordinary journal writes cause four `fsync` calls per update. A finite-value decision, scalar loss read, and explicit memory synchronization also occur. None is an established bottleneck. |
| 7. Reduce optional reporting and telemetry frequency | Experiment requiring measurement | Console output is optional. The memory callback also enforces a safety margin, so reducing it changes more than reporting frequency. |
| 7. Remove scalar loss reads or buffer the journal without a new contract | Unsupported or inapplicable | Per-update history requires Python loss values. Durable writes enforce the current failure contract. A replacement needs compatible logging and recovery behavior. |
| 8. Cache trainable parameters | Experiment requiring measurement | The loop rebuilds the list for every clipping operation. Reusing the optimizer parameter list is small, but no timing proves a useful gain. |
| 8. Benchmark explicit fused AdamW | Experiment requiring measurement | The current worker does not specify `fused` or `foreach`. The installed Torch dispatch can already choose a foreach implementation. Record the effective implementation before comparison. |
| 8. Cache CPU tensors or use pinned transfers | Experiment requiring measurement | The collator creates tensors on every visit. Measure host memory and transfer cost before adding caching. |
| 8. Configure gradient checkpointing or activation offloading | Experiment requiring measurement | Unsloth checkpointing is hardcoded at loading, adapter attachment, and training preparation. All three sites need consistent configuration. These features trade memory against work. |
| 9. Test custom-loss values and gradients | Confirmed issue | The suite lacks an actual fused-loss comparison with reference causal cross-entropy. CPU sequence tests do not establish GPU kernel parity. |
| 9. Add a manual causal label shift | Unsupported or inapplicable | Installed Unsloth Zoo passes `shift=True` to its loss implementation. Adding another shift changes the target. |
| 9. Record executing code, packages, hardware, and kernel configuration | Confirmed issue | Training identity contains dataset, recipe, and sample order. It does not bind the executing stack or prove identical GPU continuation. |
| 9. Preserve current sampling state | Already addressed in current code | The code saves explicit sample order and position. No scheduler or accumulation state exists today. |
| 9. Save future scheduler and batching state | Experiment requiring measurement | Any future machinery adds state requirements. Implement and test that state with the experiment. |
| 10. Keep batch 1 and accumulation 1 as a valid baseline | Already addressed in current code | This is the supported worker behavior. There is no padding waste inside a one-example batch. |
| 10. Test physical batch 2 or effective batch 4 | Experiment requiring measurement | Peak memory, throughput, and extraction quality decide usefulness. Accumulation alone does not execute examples simultaneously. |
| 11. Add accumulation and partial-window semantics | Experiment requiring measurement | This is a feature experiment, not a baseline bug fix. The supplied update counts are correct when each epoch applies its final partial window. |
| 11. Define weighting for accumulated loss | Experiment requiring measurement | A future batching design needs an explicit objective. Existing token-weighted history does not turn the current per-example optimizer steps into a token-weighted batch objective. |
| 12. Test `5e-5`, then warmup and decay separately | Experiment requiring measurement | Neither constant `1e-4` nor zero warmup is an identified defect. Five percent of 660 updates is 33 updates. |
| 12. Compare epoch one and final, then capacity or regularization | Experiment requiring measurement | Generalization needs extraction results. Keep rank, target coverage, dropout, weight decay, betas, epsilon, clipping, and BF16 fixed in initial comparisons. |
| 12. Switch to four-bit training to increase batch size | Unsupported or inapplicable | This changes numerical behavior and needs separate stack acceptance. No evidence here establishes a Qwen3.5 QLoRA improvement or supported comparison. |
| 13. Preserve context limits and audit allowances | Already addressed in current code | CPU preparation includes template overhead and complete responses. Every accepted example fits. This does not establish semantic completeness of every source selection. |
| 13. Add packing, padding-free execution, attention kernels, or Liger | Experiment requiring measurement | Packing combines examples into shared sequences. Model support, example separation, and loss parity need proof. Batch-one padding is not a current problem. |
| 13. Claim maximum-context extraction from short-context training | Unsupported or inapplicable | Neither the current training data nor this review supports that claim. |
| 14. Evaluate the existing final model | Already addressed in current code | Retained development evaluation and source-review evidence exist. Reuse their identities and limitations before requesting another evaluation. |
| 14. Compare base, epoch-one, and final models | Experiment requiring measurement | A matched three-model comparison is not established by the retained evidence. No epoch-one checkpoint exists in the retained model folder. |
| 14. Save epoch and final resumable checkpoints | Confirmed issue | `train_updates` supports `checkpoint_steps`, with a test, but the worker does not pass it. Interval 96 misses both 330 and 660. Changing cadence remains a separate approval. |
| 14. Use extraction metrics and preserve exposure restrictions | Already addressed in current code | The evaluation pipeline retains source-aware scoring and failures. Dataset validation preserves split and exposure rules. Add format/group breakdowns where current reporting lacks them. |
| 15. Trial Trainer or SFTTrainer with prepared examples | Experiment requiring measurement | Plain Trainer is the smaller initial candidate. Preserve preparation and implement an actual no-logit forward/loss path. Installed imports pass, but GPU integration remains untested. |
| 15. Assume `compute_loss_func` avoids full output scores | Unsupported or inapplicable | Installed Trainer calls the model before this callback. SFTTrainer also reads logits for entropy and accuracy unless its Liger path applies. |
| 16. Follow the proposed order | Intentional tradeoff worth reconsidering | Repair resume first, then add measurements and loss acceptance. Reuse existing evaluation evidence. Keep experiments, save policy, and migration separate. |
| 17. Improve the training display | Confirmed issue | The current output omits epoch progress and uses incomplete timing for its estimate. Follow Hugging Face counter conventions in a separate display and measurement patch. |

## Smallest coherent first patch

Implement read-only resume preflight, meaning validation before model setup or provenance mutation.
Share the eligibility checks between the command entry and the worker.
Use the existing run lock at the command entry.
Keep model-dependent tensor and parameter-map comparisons after model creation.
Keep the current clean-stop rule, save frequency, optimizer, data order, and numerical recipe.

The patch covers these changes:

1. Validate the journal object, status, checkpoint path, completed position, pending position, and required hashes before accessing them.
2. Reject missing journals, unsupported statuses, invalid positions, and inconsistent checkpoint metadata with actionable `ValueError` messages.
3. Compare dataset, recipe, prompt, explicit sample order, existing input records, and checkpoint identity before GPU imports or loading.
4. Compare the command's saved configuration with `run.state['identity']['recipe']`, alongside its existing dataset and prompt checks.
5. Validate the checkpoint inventory and file hashes before deserializing trusted local state.
6. Read trusted state on CPU and validate order, completed position, next position, and history before GPU loading.
7. Keep adapter tensor compatibility, optimizer parameter mapping, and restoration checks after model creation.
8. Restore RNG state after all initialization and restoration work, as the current code does.
9. Write `training-inputs.json` only for a new accepted run, and preserve its original bytes for a valid resume.
10. Reject inconsistent existing files rather than reconstructing missing provenance or overwriting it.

Do not claim that failed attempts leave every run file unchanged.
The supervisor must retain actual resource charges, and reconciliation can update accounting records.
Protect the original identity, inputs, journal, checkpoint, and result records during preflight rejection.
Perform cheap eligibility checks before launch so predictable rejection creates no worker attempt.
Keep expensive file hashing and trusted-state loading within bounded execution, before GPU setup.

Do not deserialize untrusted checkpoints.
The existing state format uses `torch.load(..., weights_only=False)` for Python and NumPy random state.
A matching hash detects changed bytes but does not establish that a checkpoint is trustworthy.

The patch needs these acceptance tests:

| Test | Required result |
|---|---|
| Journal statuses | `READY`, `BOUNDARY`, `UPDATING`, and `TRAINED` fail intentionally, including records without a checkpoint key. No `KeyError` escapes. |
| Malformed records | Missing keys, non-object JSON, invalid paths, negative or excessive positions, Boolean positions, and invalid hashes fail before model loading. |
| Identity changes | Changed dataset, recipe, prompt, sample order, inputs, completion marker, inventory, or file hash fails before deserialization or GPU loading as appropriate. |
| CPU state inconsistency | Wrong order, completed position, next position, or history fails before GPU loading. A valid manifest alone does not establish a valid trajectory. |
| Input preservation | Rejected resume, rejected fresh training, and valid resume preserve existing provenance bytes. No rejected attempt replaces the input record. |
| Entry-point boundary | Replace worker launch with a sentinel. Cheap rejection never reaches it. Independently test that direct worker calls reject before GPU imports or loading. |
| Valid continuation | The existing dropout-based CPU test retains identical losses, parameters, optimizer tensors, and sample order across interruption and resume. |
| Resource accounting | Existing charges never decrease. Cheap eligibility rejection does not launch a worker. Bounded worker rejection preserves provenance and counts its resource cost. |
| Regression | The retained CPU suite passes with CUDA disabled and no dependency skips. No default recipe or checkpoint-policy diff appears. |

Malformed-record handling belongs in the shared validators used by preflight.
Avoid creating a second checkpoint implementation just to change the order of operations.
CPU acceptance can establish this patch without a training campaign.
Exact Qwen/Unsloth CUDA continuation remains a separate user-launched acceptance test.

## What Trainer can improve

Trainer already provides generic batching, partial accumulation windows, scheduling, logging, and distributed-training machinery.
These are useful maintenance benefits when the project needs those features.
Its current [extension points](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) include custom data collation, optimizers, and loss behavior.
The installed Transformers 5.5.0 source, rather than the moving documentation defaults, controls this review's implementation findings.

| Responsibility | Keep or delegate | Acceptance condition |
|---|---|---|
| Source preparation and accepted labels | Keep in ProxyBench | Preserve source identity, complete evidence, exact token IDs, masks, and termination. |
| Collation | Reuse `ResponseCollator` | Preserve padding and `-100` labels without retokenization. |
| Forward and loss | Keep the efficient path through a model wrapper or `Trainer.compute_loss` override | Call the backbone and fused loss without allocating full logits, which are scores for every vocabulary token. Test gradients and evaluation behavior. |
| Optimizer | Pass an explicit optimizer | Match parameter groups, order, precision, betas, epsilon, weight decay, and backend. Do not inherit Trainer's fused-AdamW default. |
| Scheduling | Delegate only with explicit configuration | The fidelity trial uses a constant schedule with no warmup. Save any scheduler state added by the trainer. |
| Sample order | Supply the recorded order explicitly | Trainer's default sampler does not reproduce Python `random.Random(seed)` shuffling merely because the seed matches. Preserve RNG consumption around loader creation and resume. |
| Accumulation | Delegate after loss acceptance | Choose per-example or per-token weighting explicitly. For token averaging, divide by total supervised tokens. Apply each partial window and clip once per optimizer update. |
| Nonfinite protection | Keep explicit rejection | Trainer's filtering of invalid logged losses is not equivalent to aborting before an invalid optimizer update. |
| Checkpoints and resume | Keep the ProxyBench contract | Preserve publication validation, clean-stop identity, failure boundaries, and RNG restoration. Default Trainer saves are not an equivalent replacement. |
| Resource limits | Keep the external supervisor | Preserve host/device margins, bounded phases, one writer, cumulative charges, and stop behavior. |
| Extraction evaluation | Keep the current pipeline | Preserve raw answers, terminal failures, evidence review, and development-exposure limits. |

For this project, plain Trainer is a better first migration candidate than SFTTrainer.
ProxyBench already prepares complete token sequences, so much of SFTTrainer's text preparation duplicates existing work.
[TRL 0.24.0](https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer) supports prepared datasets, but its installed loss method also computes entropy and token accuracy from logits.
Unsloth patches can change that behavior, so inspect and test the effective patched path before claiming incompatibility or savings.
Do not enable Liger solely to bypass those metrics without a separate supported-stack comparison.

A trainer migration has limited value if it requires copying the upstream training loop to preserve all current behavior.
Prefer supported overrides and callbacks, then measure how much project code they actually remove.
If those extensions cannot preserve update and recovery boundaries, keep the custom loop or consider a narrower Accelerate integration.
No speed claim is justified until matched runs retain the same initialization, order, workload, backend, save cadence, and extraction evaluation.

## Training display recommendation

Use Hugging Face progress conventions for the terminal display.
An epoch is one pass through the training examples.
An optimizer update applies the calculated gradients to trainable parameters.
Hugging Face defines `global_step` as completed optimizer updates and `epoch` as completed epochs plus fractional progress.
Its `ProgressCallback` displays progress, while `PrinterCallback` prints logs.
These are documented conventions, not a mandatory screen layout. See the [Hugging Face callback documentation](https://huggingface.co/docs/transformers/main_classes/callback#transformers.TrainerState).

The installed Transformers 5.5.0 `trainer_callback.py:35–58` confirms those counter meanings.
Its `ProgressCallback` at lines 624–695 advances a `tqdm` bar using `global_step` and `max_steps`.
The version-specific online callback page was unavailable, so the installed source supplies the version-specific comparison.
The [TRL 0.24.0 metric documentation](https://huggingface.co/docs/trl/v0.24.0/en/sft_trainer#logged-metrics) lists loss, epoch, update count, learning rate, token count, and gradient norm.
Adopt useful fields without adding its logits-based metrics to the efficient loss path.

The current [loop output](../src/proxybench/training/trajectory.py) reports steps, losses, learning rate, elapsed time, an estimate, and memory.
It does not show epoch progress or batch configuration.
The [supervisor](../src/proxybench/execution/live.py) separately prints phase and resource-budget heartbeats.
Consolidate their visible status into one display while preserving saved worker output and safety messages.
This display improvement does not require a Trainer migration.

Use these display fields:

| Field | Proposed meaning |
|---|---|
| Phase | Preparation, loading, training, saving checkpoint, validating final adapter, or publishing final adapter. Show the actual active operation. |
| Epoch | Current epoch and total epochs, plus completion within that epoch. Store fractional epoch progress separately for logs. |
| Updates | Completed optimizer updates divided by planned updates, plus a percentage. Baseline total: 660. |
| Examples | Completed examples in the current epoch divided by its size. Baseline size: 330. Derive this from sample position when batching exists. |
| Batch configuration | Physical batch size, accumulation count, and effective batch size. Show once at startup and resume, and keep it available in detailed status. |
| Training loss | Latest update loss and the existing response-token-weighted mean over the last 12 updates. Label the window and weighting explicitly. |
| Learning rate and gradient norm | Current learning rate and the gradient norm before clipping. Reuse existing values rather than calculating gradients again. |
| Time | Elapsed time for this attempt, cumulative charged time, remaining resource budget, and a labeled estimate of remaining loop time. |
| Throughput | Non-padding sequence tokens per second and supervised response tokens per second. State the measurement window in detailed output. |
| Memory | Current allocated memory, peak allocated/reserved memory, and free device memory. Keep units consistent. |
| Last save | Last fully published checkpoint and its update number. Display resume eligibility separately, based on the clean-stop validator. |

This example shows the baseline after update 396, with 66 examples completed in the second epoch.
The numbers represent counters only, not a measured training run.
Loss, time, and memory fields appear when measurements are available.

```text
Phase: training
Epoch 2/2: 20% complete (66/330 examples)
Optimizer updates: 396/660 (60%)
Batch: 1 example | Accumulation: 1 | Effective batch: 1 example
```

At this position, the fractional epoch value is `1.20`.
At update 330, report epoch 1 complete and fractional epoch `1.00`.
At update 660, report epoch 2 complete and fractional epoch `2.00`.
Do not derive epoch progress from update count after introducing unequal or partial accumulation windows.
Use the recorded sample position and actual epoch boundaries instead.

The display patch needs these behaviors:

1. Let the supervisor own terminal rendering because it captures worker output through pipes.
2. Use a compact progress bar for an interactive terminal and plain lines for redirected output.
3. Refresh the interactive display at most once per second, with immediate phase, stop, failure, and completion messages.
4. Print plain progress summaries every 10 completed updates and at epoch boundaries, without changing per-update history.
5. Restore counters from accepted resume state before showing progress, and start attempt timing from the new launch.
6. Show `estimating` until complete-loop timing supports an estimate, and label final-publication time as excluded until separately estimated.
7. Keep the phase active during checkpoint saving and final publication, even when the update bar reaches 100%.
8. Show successful completion only after final validation, artifact publication, and the worker result succeed.

The refresh and summary intervals above are proposed project defaults, not Hugging Face requirements.
Keep safety monitoring, durable journal writes, and checkpoint cadence independent from display frequency.
Do not infer resume eligibility from a checkpoint's existence or a progress percentage.
Optional display measurements must not introduce extra per-update GPU synchronization solely to refresh the screen.

Accept the display patch with CPU tests and a simulated clock.
Cover initial state, updates 329/330/331 and 659/660, resumed counters, missing measurements, phase changes, and a failed final publication.
Test terminal and redirected output, long saves, time-budget stops, and preservation of saved output when terminal output is slow.
If accumulation is later introduced, test distinct sample, microbatch, and optimizer counters, including the last partial window.
A microbatch is one forward-and-backward batch within accumulation.
Keep this patch after resume safety and alongside complete timing, without changing the training recipe or recovery policy.

## Measurements and acceptance after the first patch

Add measurement fields in a separate patch without changing checkpoint or recovery policy.
The supervisor already measures whole worker attempts, including startup and saving.
Connect that accounting to explicit stage measurements rather than replacing it with update timing.
Report failed-attempt time separately and preserve cumulative totals.

Record preparation, model loading, initial base hashing, adapter attachment, restoration, full-loop time, and final publication.
Within the loop, separate computation, journal writes, safety monitoring, reporting, and checkpoint stages.
Track total non-padding sequence tokens separately from supervised response tokens.
Use matching time intervals for token-rate numerators and denominators.
Record peak allocated and reserved GPU memory, and state when peak counters reset.
Separate first-update compilation from later updates, without removing startup from the overall result.

Device execution is asynchronous, so CPU timestamps alone do not isolate GPU computation.
Use device events or deliberate synchronization in a bounded profiling mode, and account for measurement overhead.
Do not remove a safety barrier based only on an assumed duplicate synchronization.
The [PyTorch tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) describes synchronization costs, but it does not establish this workload's bottleneck.

Loss acceptance must exercise the installed helper, not only a Python imitation.
Unsloth Zoo 2026.9.3 `loss_utils.py:187` passes `shift=True` and `filter_eps='auto'` to cut-cross-entropy.
Use reference logits with the next-token labels shifted exactly once.
Test first-response supervision, termination, ignored prompt labels, padding, unequal response lengths, and adapter gradients.
Choose numerical tolerances for the actual numeric types and approximate gradient filtering before evaluating results.
CPU tests establish sequence semantics, but the actual fused CUDA loss needs user-launched GPU acceptance.

Bind future exact-resume claims to code, package, model, tokenizer, optimizer, hardware, driver, and relevant kernel configuration.
Treat changes in these items as requiring new acceptance rather than inheriting an exactness claim.
The [PyTorch reproducibility notes](https://docs.pytorch.org/docs/main/notes/randomness.html) explain that identical seeds do not guarantee identical results across releases or platforms.
Do not silently enable different deterministic kernels in the baseline while describing it as unchanged.

One further failure boundary needs a separate design decision.
The loop writes `TRAINED` before the worker compares frozen base weights and publishes the final adapter.
A failure during final publication leaves no ordinary clean-stop resume path.
Do not accept `TRAINED` or roll back automatically as part of the first patch.
Review finalization retry and rollback together with checkpoint cadence, while retaining failed-attempt charges.

## Evidence locations and completed checks

The main findings come from the repository and installed source listed below.
Line numbers refer to the reviewed revision and installed versions.
The online Transformers 5.5.0 documentation page was unavailable through the browser tool, so installed source supplied the version-specific behavior.

| Source | Relevant locations |
|---|---|
| [Training worker](../src/proxybench/training/runtime.py) | Lines 144–233: setup, provenance mutation, resume, memory, loss, and saving. Lines 396–415: outer resume and resource reconciliation. |
| [Trajectory](../src/proxybench/training/trajectory.py) | Lines 75–148: checkpoint publication and restoration. Lines 151–214: updates, journals, timing, and explicit checkpoint steps. |
| [Checkpoint publication](../src/proxybench/training/checkpoints.py) | Manifest validation, portable adapter publication, read-back, and durability. |
| [Sequence preparation](../src/proxybench/training/preparation.py) and [sequence masks](../src/proxybench/training/sequences.py) | Template boundaries, accepted-answer comparison, length accounting, and termination. |
| [Adapter helpers](../src/proxybench/training/adapters.py) | Tensor construction, padding, broad projection coverage, and adapter comparisons. |
| [Resource supervision](../src/proxybench/execution/resources.py) and [run state](../src/proxybench/runstate.py) | Durable writes, elapsed accounting, ownership, and cumulative limits. |
| [Trajectory tests](../tests/test_trajectory.py) | Lines 34–64: exact CPU continuation. Lines 114–121: explicit epoch/final checkpoint positions. |
| Installed `transformers/trainer.py` | Lines 1003–1032: sampling. Lines 1689–1778: partial windows and update callbacks. Lines 1938–2020: forward before custom loss. |
| Installed `transformers/training_args.py` | Linear schedule and batch-size defaults, plus fused AdamW default for the installed Torch version. |
| Installed `trl/trainer/sft_trainer.py` | Lines 790–797: preparation bypass. Lines 1080–1179: logits-based metrics and scalar reads. |
| Installed `unsloth_zoo/loss_utils.py` | Lines 187–217: loss normalization, internal causal shift, and gradient-filter configuration. |

The complete retained CPU suite passed: 118 tests, with CUDA disabled and no skips.
The first sandboxed run blocked local test sockets, and the approved rerun outside that sandbox passed.
CPU dataset preparation passed for all 420 examples without truncation.
Trainer and SFTTrainer imports passed with CUDA disabled.
No GPU performance, loss-parity, or exact-resume claim follows from these CPU results.
