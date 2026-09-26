# Training acceptance

These tests cover loss semantics and continuation after a clean stop.
Loss measures prediction error.
A gradient measures how a parameter affects that error.
The CPU tests do not establish CUDA kernel parity.
A kernel is a computation that runs on the GPU.

Use the [preparation guide](preparation.md) before GPU acceptance.
Require the pinned environment, declared inputs, and complete CPU suite to pass without dependency skips.
The user starts every GPU command below.
These commands do not start a full training campaign or change the retained model.
The [training review](training-review.md) explains the evidence gaps.
The [GPU acceptance results](training-gpu-acceptance.md) record the September 26 loss pass and exact-continuation failure.

## CPU evidence

The reference computes ordinary causal cross-entropy with one target shift.
Cross-entropy measures the error in the target token probability.
The shift pairs each target token with the preceding hidden state.
The installed Unsloth Zoo helper already applies `shift=True`.
Do not shift its inputs again.

The CPU tests compare reference loss and gradients with independently enumerated target pairs.
They cover the first response token, termination, ignored prompts, right padding, and unequal sequence lengths.
The first response uses the last prompt position as its predictor.
Prompt targets and padding remain excluded.
The mean uses the number of supervised tokens, rather than the number of examples.

Run the focused tests and inspect the installed helper without initializing CUDA:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s tests -p test_loss_acceptance.py -v
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m proxybench.training.loss_acceptance inspect
```

The inspection reads installed source and records its SHA-256, an identity for file bytes.
It requires the internal shift and the expected gradient filter argument.
It does not import Unsloth or prove that the installed GPU implementation executes successfully.

## User-launched CUDA loss comparison

After all preparation and CPU acceptance pass, choose a new external directory.
If the directory exists, the command refuses to reuse it.
Run from the repository root:

```bash
.venv/bin/python -m proxybench.training.loss_acceptance loss \
  "$HOME/proxybench-runs/loss-acceptance-001"
```

The command calls the actual installed `fused_linear_cross_entropy` helper.
It compares loss and gradients against the FP32 reference.
The BF16 gradients cover hidden states, output weights, and both matrices of a synthetic rank-8 adapter.
An adapter is a small trainable addition to a frozen model.
FP32 and BF16 are numerical storage formats with different precision.
Each format uses both disabled filtering and the helper default, `auto`.
The helper receives original labels without a manual shift.

The installed kernel accepts BF16 or FP16 hidden states for backward calculation.
It rejects FP32 hidden-state gradients with `Backwards requires embeddings to be bf16 or fp16`.
FP32 cases compare forward loss only and record `gradient_status=UNSUPPORTED_DTYPE`.
They do not establish FP32 gradient agreement.
BF16 cases exercise the baseline training precision and require all gradient comparisons.

The test uses five synthetic mask cases with 64 hidden dimensions and 2,048 output tokens.
It uses seed 42 and disables TF32, a reduced-precision matrix multiplication mode.
The reference matches the helper input cast before FP32 multiplication.
The command does not load model weights or private examples.
These small tensors do not establish parity for every production shape.

The acceptance budgets are fixed before execution:

| Format | Maximum absolute loss error | Maximum relative gradient norm error |
|---|---:|---:|
| FP32 | `2e-5` | Unsupported by the installed backward kernel |
| BF16 | `5e-3` | `2e-2` |

Relative gradient norm error divides the difference norm by the reference norm.
The FP32 allowance covers differences in parallel arithmetic.
The BF16 allowance covers rounding from its seven stored fraction bits and differences in arithmetic order.
These are acceptance budgets, not measured error claims.
Ignored predictor gradients must remain exactly zero in every case.

The installed `auto` filter uses `torch.finfo(dtype).eps / 32`.
It permits the kernel to omit small gradient contributions.
The BF16 threshold is `0.000244140625`, and the FP32 threshold is about `3.73e-9`.
The comparison retains the same budgets with filtering disabled and enabled.
If a case fails, inspect its recorded errors before proposing a different budget.
Do not loosen a budget solely to obtain a passing result.

The supervisor limits this command to 600 seconds and preserves the configured memory margins.
It retains stdout, stderr, memory samples, and cumulative resource usage in the external directory.
`loss-report.json` records each case, the helper identity, thresholds, errors, and overall status.
An incomplete report or a nonzero exit does not pass acceptance.

## User-launched continuation comparison

After the CUDA loss comparison passes, use another new external directory.
Make sure that the pinned base and accepted training dataset remain available.
Run from the repository root:

```bash
.venv/bin/python -m proxybench.training.loss_acceptance continuation \
  "$HOME/proxybench-runs/continuation-acceptance-001"
```

This command calls the actual training worker in three fresh supervised processes.
The uninterrupted arm stops cleanly after four updates.
The second arm stops cleanly after two updates, reloads its saved state, and stops after four total updates.
Acceptance stop hooks limit execution without changing the recipe or sample order.
Each process uses the actual backbone, fused loss, adapter, optimizer, and checkpoint code.

Both arms use the complete 660-position sample order for the retained 330-example training split.
They execute its first four positions.
The two-epoch recipe, response masks, adapter initialization, optimizer settings, and periodic save interval remain unchanged.
The interval remains 96, so this short comparison does not cross a periodic save.
Only clean stops create checkpoints during these arms.
The comparison does not select prompts or thresholds with development examples.

The comparison requires exact agreement in these records:

- Initial trainable parameter identities and training input identities.
- Recorded runtime identities, including code, packages, hardware, and kernel configuration.
- Final adapter tensors, optimizer state, and parameter mapping.
- Random-number states, full sample order, and next sample position.
- Per-update loss, sample index, step number, and supervised token count.

Random-number state determines the next random values.
Wall-clock timing and memory measurements remain evidence but do not participate in exact agreement.
No numerical tolerance excuses continuation differences.
A difference leaves actual-stack continuation acceptance incomplete.
This test does not establish longer-run determinism, extraction quality, or speed improvement.

Each subprocess has a 1,800-second limit.
All three share a 5,400-second cumulative limit and retain the configured memory margins.
If a resource stop prevents a required boundary, the command fails acceptance.
The command preserves its private checkpoints and logs for review.
`continuation-report.json` records a successful exact comparison.
GPU results remain pending until the user runs these commands successfully.

## Separate timing experiment

After the preceding acceptance passes, use another new external directory for device-event measurements.
A device event records time on the GPU execution stream.
This optional mode adds measurement overhead and keeps it separate from the ordinary baseline.
The existing safety synchronization makes each completed event available without an extra display synchronization.

```bash
PROXYBENCH_PROFILE_CUDA=1 .venv/bin/python -m proxybench.training.loss_acceptance continuation \
  "$HOME/proxybench-runs/timing-acceptance-001"
```

The same limits bound this experiment.
Runtime identity records that device-event profiling is active.
The complete worker cost still includes startup, compilation, event overhead, checkpoint work, and cleanup.
Four updates cannot establish steady-state speed for the complete workload.
