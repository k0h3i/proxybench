# Fragment comparison controller

The controller compares the bounded HTML parser with Sol-in-Codex on six development fragments.
A schedule fixes the input order and permits one attempt per system and fragment.
The controller keeps reference admission and scoring outside both extraction environments.
Use the [model contract](model-extraction-contract.md) for the complete response structure and extraction rules.

## Prepare the package

Call `proxybench.execution.comparison.freeze` from a trusted preparation script.
Supply the accepted schedule, approved parser snapshot, snapshot hash manifest, model contract, CLI executable, and model cache.
Supply an original synthetic session transcript and the recorded user authorization.
The function refuses an existing destination and admits all six references before creating the package.

The package contains exact source bundles, prompts, parser files, model cache, application instructions, and a manifest.
A manifest records file hashes and the execution configuration.
The separate execution schedule contains no reference paths.
Keep the package under ignored local artifacts because its trusted schedule includes reference bindings.

Make sure that synthetic tests pass before freezing a package.
Demonstrate denied reference access, process cancellation, and original Sol capture in the actual operating environment.
Keep those observations beside the run package.
Freeze changes to the controller before execution because the controller checks its own file hash.

## Isolation and model configuration

The Linux controller requires `bubblewrap`, a local filesystem isolation tool.
Each extraction runs in a fresh mount and process namespace, which limits visible files and processes.
The parser sees system libraries and the approved Python package.
Its network namespace has no external connection.

Sol sees the CLI executable, system libraries, certificate files, and a fresh private state directory.
That directory contains subscription authentication and the frozen model cache, but no prior sessions or user configuration.
The controller removes temporary authentication after the attempt and never includes it in exported observations.
The repository, accepted references, review notes, and prior predictions are absent from both namespaces.

The CLI uses `gpt-5.6-sol` with `model_reasoning_effort="medium"`.
It ignores user configuration and rules, disables memory, and disables the listed tool features.
Any recorded tool call stops the schedule.
The current CLI can emit a warning that disabled Code Mode is unavailable.
That warning is retained as an application observation.

Each original session must match the frozen base instructions, developer instructions, and environment text.
It must contain the exact delivered prompt once and record the required model and effort.
The model cache is frozen because it can affect model instructions and capabilities.
Hosted implementation details that the CLI does not expose remain unknown.

## Execute the schedule

Run the frozen package with the existing subscription authentication:

```bash
PYTHONPATH=src python3 -m proxybench.execution.comparison artifacts/runs/RUN_ID \
  --auth /path/to/private/auth.json
```

Execution uses the frozen model cache in the package.
The controller refuses an existing attempts directory.
Do not resume a partial schedule or reuse its output path.
A later retry requires a separate run and user decision.

The parser runs first for the first, third, and fifth inputs.
Sol runs first for the other inputs.
All twelve slots remain in the report, including administrative stops and failed attempts.
Scoring starts after every slot reaches a terminal state.

| Limit | Value |
|---|---|
| Source bundle | 262,144 bytes |
| Preparation per attempt | 20 seconds |
| Parser execution | 10 seconds |
| Sol execution | 300 seconds |
| Finalization per attempt | 10 seconds |
| Complete schedule | 2,400 seconds |
| Required time before each pair | 370 seconds |
| Parser stdout and stderr, separately | 1,048,576 bytes |
| Sol native final response | 1,048,576 bytes |
| Sol event stdout and stderr, separately | 8,388,608 bytes |
| Combined original session transcripts | 8,388,608 bytes |

Preparation and finalization use interruptible deadlines in the single-threaded Linux controller.
Execution uses the earlier process deadline or global deadline.
Cancellation kills the process group and waits for termination of the namespace owner.
The controller launches no further extraction until cancellation and cleanup finish.

## Capture and failure rules

The CLI writes the final response through its native `-o` export.
The controller preserves those bytes, event stdout, stderr, and original session JSONL files.
It compares the final export with the original final message without selecting an answer from reference scores.
It applies no repair or model-specific normalization.

Timeouts retain observed parser stream bytes as partial output.
Sol responses without an on-time durable final export remain missing in primary scoring.
Late native responses stay separate and cannot replace primary output.
Byte limits can truncate diagnostic captures, and such attempts remain ineligible.

Binding changes, added executable files, incorrect capture associations, tool calls, and missing provenance stop the complete schedule.
Provenance means the original observations that identify and describe an attempt.
If finalization cannot export that evidence, the controller preserves the original session files in quarantine.
Cleanup failures also stop the schedule and remain explicit in its report.

## Read the results

`report.json` separates content matches, execution eligibility, and protocol validity.
Eligibility means that the original attempt satisfied its execution and capture requirements.
An integrity incident makes controlled comparison counts unavailable for both systems.
Content scores remain diagnostic, with six scheduled inputs per system.

These reused, assisted development examples support descriptive differences only.
They do not establish independent accuracy, complete-filing recovery, or model superiority.
Mechanical citation checks do not measure whether each passage supports the asserted meaning.
Training, new annotation, and untouched testing remain separate stages.
