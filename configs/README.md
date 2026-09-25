# Portable configuration

`training.json` records the supported training recipe and local base-model path.
`base-model.json` records the pinned base revision and SHA-256 hashes for its files.
`inference.json` records the supported llama.cpp version and generation controls.
`model-system-prompt.txt` contains the exact canonical prompt bytes.
`requirements-training.txt` pins Python model packages for `.venv/`.
`requirements-llama-cuda.txt` pins CUDA 12 libraries for the native llama.cpp runtime.
Install those libraries into a separate target through the [preparation guide](../docs/preparation.md).
The guide also installs the llama.cpp executable, compiler, and required system libraries.

Keep project Python packages in the root `.venv/`.
Keep runtime binaries and future runs outside the repository.
Keep one base-model copy under `artifacts/models/Qwen3.5-4B/`.
Supply private SEC contact identity through runtime arguments or environment variables.
Do not add personal email, credentials, or workstation paths to tracked configuration.
Read the [training guide](../docs/training.md) and [inference guide](../docs/inference.md).
