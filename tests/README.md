# CPU tests

End-to-end (E2E) tests exercise complete user workflows.
The [evaluation command test](test_evaluation_command.py) runs a synthetic server through the public command.
It covers generation, review, repeated imports, conflicting decisions, and report recovery.
The [supervisor tests](test_live_supervisor.py) run real worker processes through limits, interruptions, and failures.

Retained isolated tests cover failures that these workflows do not exercise.
They include source boundaries, stale acceptance, changed source identities, corrupted checkpoints, and training calculations.
Tests use small synthetic sources and model substitutes.
They do not measure model accuracy.
GPU loading and inference require a separate user launch.

Follow the [testing rules](../AGENTS.md#validation) before adding tests.
If a system needs an isolated test, list its failure modes before writing its code.
Keep the test only when retained E2E tests miss its failure.

Run the retained CPU suite from the repository root and save its evidence:

```bash
export PROXYBENCH_TEST_ARTIFACT_DIR="$(mktemp -d /tmp/proxybench-cpu-tests.XXXXXX)"
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v \
  > "$PROXYBENCH_TEST_ARTIFACT_DIR/cpu-tests.log" 2>&1
cat "$PROXYBENCH_TEST_ARTIFACT_DIR/cpu-tests.log"
printf '%s\n' "$PROXYBENCH_TEST_ARTIFACT_DIR"
```

The evaluation command test saves its completed run under `evaluation-command/` in that directory.
The saved run contains inputs, raw answers, review decisions, execution logs, and the final report.
The test rebuilds the report from that saved run and compares it with the accepted result.

Make sure that the saved report can be rebuilt without the synthetic server or model:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python -m proxybench evaluate \
  --run-dir "$PROXYBENCH_TEST_ARTIFACT_DIR/evaluation-command" --report-only --json \
  > "$PROXYBENCH_TEST_ARTIFACT_DIR/replayed-report.json"
cmp "$PROXYBENCH_TEST_ARTIFACT_DIR/evaluation-command/evaluation/report.json" \
  "$PROXYBENCH_TEST_ARTIFACT_DIR/replayed-report.json"
```

If a CPU test fails, read `cpu-tests.log` before using the saved artifacts.
Do not skip required CPU checks merely because private experiment data are unavailable.
