# Shared configuration

Store portable experiment configuration here.
Keep credentials, personal SEC contact details, and machine-specific paths in ignored local files or environment variables.
Record the effective configuration with each run.

The [Qwen3.5-4B configuration](qwen35-4b-smoke.json) defines the bounded adapter test.
Its [dependency file](requirements-qwen35-4b-smoke.txt) records the isolated Python runtime.
Read the [execution guide](../docs/qwen35-4b-smoke.md) before running a GPU phase.

The [historical pilot configuration](qwen35-4b-historical-pilot.json) pins the accepted 96/24 release and 192-update recipe.
It uses the existing local environment and a two-hour total GPU ceiling.
The [pilot guide](../docs/qwen35-4b-historical-pilot.md) provides user-run commands and stop/resume instructions.

The older `qwen35-preparation.json` and `requirements-gpu-preparation.txt` describe the earlier 9B probe.
They do not define the selected 4B test.
Preserve them as records of earlier work.
