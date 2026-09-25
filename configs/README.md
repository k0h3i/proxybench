# Portable configuration

`training.json` records the supported training recipe.
`inference.json` records the supported llama.cpp version and generation controls.
`model-system-prompt.txt` contains the exact canonical prompt bytes.
The requirement files pin the optional environments used by retained commands.

Keep environments, runtime binaries, base caches, and future runs outside the repository.
Supply private SEC contact identity through runtime arguments or environment variables.
Do not add personal email, credentials, or workstation paths to tracked configuration.
Read the [training guide](../docs/training.md) and [inference guide](../docs/inference.md).
