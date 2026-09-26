# GPU acceptance results

The user authorized these tests on September 26, 2026.
The GPU loss comparison passed its supported cases.
The exact continuation comparison failed.
The baseline recipe and retained models stayed unchanged.
No Trainer experiment or separate profiling run formed part of this acceptance.

## Environment and preparation

The tests used the NVIDIA GeForce RTX 3090 and the pinned Python environment.
The recorded stack included Torch `2.12.1+cu130`, Unsloth `2026.9.4`, and Unsloth Zoo `2026.9.3`.
The training worker selected AdamW's `foreach` implementation with FP32 adapter parameters and learning rate `1e-4`.
The dependency consistency test passed.
After the acceptance-harness correction below, all 195 CPU tests passed with CUDA disabled and no skips.

## Loss comparison

The first run stopped because the installed cut-cross-entropy kernel rejects backward calculation with FP32 hidden states.
Its error was `Backwards requires embeddings to be bf16 or fp16`.
The run retained its error logs and 12.74 seconds of resource charges.
This failure exposed an invalid precision assumption in the acceptance harness.

The corrected harness compares FP32 forward loss without requesting unsupported gradients.
It marks those cases with `gradient_status=UNSUPPORTED_DTYPE`.
BF16 cases still require agreement for loss, hidden gradients, output-weight gradients, and both synthetic adapter matrices.
The numerical tolerances remain unchanged for every supported comparison.
The reference still shifts targets once, and the installed fused helper receives unshifted labels.

All 20 corrected cases passed.
They cover five mask cases, two numerical formats, and filtering both disabled and enabled.
Ten FP32 cases establish forward-loss agreement only.
Ten BF16 cases establish loss and gradient agreement within the declared limits.

| Measurement | Largest observed error | Acceptance limit |
|---|---:|---:|
| FP32 forward loss, absolute | `0.00000190735` | `0.00002` |
| BF16 loss, absolute | `0.00224877` | `0.005` |
| BF16 gradients, relative norm | `0.0143594` | `0.02` |

The corrected loss worker completed in 27.23 seconds.
The report covers small synthetic tensors and does not establish agreement for every production shape.
FP32 hidden-state gradients remain unsupported by this installed kernel.

## Exact continuation

The uninterrupted worker completed four updates and saved a clean checkpoint.
The second worker completed two updates and saved a clean checkpoint.
The third worker restored that checkpoint, completed two more updates, and saved another clean checkpoint.
All three workers exited successfully and left no owned processes running.
Their combined resource charge was 267.45 seconds, within the 5,400-second limit.

The comparison requires exact equality, not approximate agreement.
Initial trainable weights and training input identities matched.
Recorded runtime identities, sample order, parameter mapping, completed position, and final random state matched.
Adapter weights, optimizer moments, and losses did not match.
The acceptance report therefore records `FAILED` with `Saved tensor differs`.

| Update | Uninterrupted loss | Split-run loss |
|---|---:|---:|
| 1 | `0.08819703012704849` | `0.08819712698459625` |
| 2 | `0.06445515155792236` | `0.06391091644763947` |
| 3 | `0.036427438259124756` | `0.0366729199886322` |
| 4 | `0.021446730941534042` | `0.020720846951007843` |

The two fresh starts differ before the checkpoint interruption.
This result does not isolate a restoration defect.
All 496 adapter tensors differ after four updates, with maximum absolute difference about `0.000736343`.
The 992 optimizer moment tensors also differ.
No tolerance change or deterministic-kernel change concealed these differences.
Exact CUDA continuation remains unaccepted until a separate investigation resolves the divergence.

## Optional acceleration warning

The pinned environment does not include `causal-conv1d`.
The generated Qwen implementation therefore uses Torch for that convolution path.
Unsloth supplies bundled flash-linear-attention code, and this run compiled its forward and backward kernels.
The Transformers warning covers several optional functions and appears when any one is absent.
It does not mean that all linear-attention operations use the Torch fallback.

The pinned preparation procedure did not audit every optional acceleration package.
No incompatibility with `causal-conv1d` was established during this run.
Adding it requires a separate environment and performance comparison.
The warning alone does not explain the observed continuation differences.

## Retained evidence

The external run folders retain reports, checkpoints, raw output, runtime identities, memory samples, and resource records.
Their identifiers are listed below:

- `loss-acceptance-20260926T1603`: Original harness failure and resource charge.
- `loss-acceptance-20260926T2005`: Corrected loss report with 20 passing cases.
- `continuation-acceptance-20260926T2008`: Three clean worker exits and the failed exact comparison.

The [acceptance guide](training-acceptance.md) describes the commands and their limits.
The [implementation report](training-implementation.md) records the preceding CPU evidence.
These short runs do not establish extraction accuracy, full-campaign determinism, final adapter publication, or a speed improvement.
