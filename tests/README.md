# Behavior tests

Run the retained CPU suite from the repository root:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Tests use small synthetic sources and model substitutes where possible.
They protect source boundaries, label meaning, review bindings, dataset identity, process limits, and run recovery.
GPU loading and inference require a separate user launch.
Do not skip required CPU checks merely because private experiment data are unavailable.
