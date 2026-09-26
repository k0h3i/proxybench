# Training implementation evidence

This implementation follows the [training review](training-review.md).
It keeps the custom loop as the default.
The separate Trainer candidate remains an experiment.
No GPU training, evaluation, profiling, loss comparison, or continuation test ran during implementation.

## Completed implementation

Resume validates journal structure, input identities, checkpoint metadata, and CPU state before model loading.
Cheap rejection occurs before worker launch.
Expensive file authentication stays inside bounded execution.
Existing input records remain unchanged, and failed worker attempts retain their resource charges.
The resume milestone is commit `7a8cc18`.

Measurements cover complete attempts, loop work, checkpoint stages, final publication, useful tokens, and memory peaks.
Runtime records identify code, packages, models, tokenizers, optimizer behavior, hardware, and kernel configuration.
The supervisor controls terminal progress and redirected summaries.
Successful completion requires final publication, a saved worker result, and successful process exit.
The measurement, display, and loss-acceptance milestone is commit `bedb8f5`.

The Trainer candidate reuses prepared sequences, the efficient loss path, explicit AdamW, and recorded sample order.
CPU comparisons require identical losses, parameters, optimizer state, and random state on a small dropout model.
Separate loop identities reject accidental changes between the candidate and custom loop during resume.
The candidate milestone is commit `02022d9`.
Its added code does not establish a maintenance or performance benefit.

## Validation evidence

The final retained CPU suite passed 194 tests with CUDA disabled and no skips.
The same 194 tests passed from a clean source checkout without private models or dataset files.
Local socket restrictions required an approved suite execution outside the sandbox.
Portable command help and CPU source-inspection commands also passed from that checkout.

Python 3.12.14 and all pinned dependencies were available in the root environment.
The pinned installation command and `pip check` passed.
The compiler, prepared native libraries, runtime identity, base file hashes, and converter identity and help passed CPU preparation.
Sequence preparation accepted all 330 training and 90 development examples without truncation.
These results do not establish a working CUDA training path.

The wheel build passed, and its 49 files contained no private inputs or workstation paths.
Changed guide links, Markdown code fences, and `git diff --check` passed.
A fresh Astra agent reviewed the combined changes with `xhigh` reasoning effort.
The review found resume, telemetry, and measurement issues that the implementation corrected and covered with regression tests.
No review finding remains open.

## Pending acceptance and decisions

The [acceptance guide](training-acceptance.md) provides bounded user-launched loss, continuation, and timing commands.
Actual fused CUDA loss and adapter-gradient agreement remain pending.
Actual-stack exact continuation remains pending.
No GPU speed improvement or extraction-quality improvement is established.

The [Trainer guide](trainer-experiment.md) provides its separate bounded commands and source-identity restrictions.
Matched GPU correctness, runtime, memory, and maintenance evidence must support adoption before it replaces the custom loop.
Extraction evaluation remains project-specific and retains raw predictions and terminal failures.
No development examples moved into training.

The baseline remains 330/90, two epochs, 660 updates, seed 42, batch 1, accumulation 1, and constant AdamW at `1e-4`.
The other pinned recipe values remain unchanged.
The worker still uses its configured periodic checkpoint interval of 96 and does not add epoch-boundary saves.
Periodic checkpoints still do not authorize rollback.
Failure after `TRAINED` during final publication still lacks ordinary clean-stop recovery.
Checkpoint cadence, rollback, finalization retry, and public release remain separate policy decisions.
