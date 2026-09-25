# ProxyType-4B

ProxyType-4B is the working name for a model that extracts one marked voting target from historical SEC N-PX text and HTML.
Name clearance and public release remain pending.
The output combines source values with standardized labels under the [label contract](docs/label-contract.md).

## Model and training

The base model is `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
The retained final adapter completed 660 updates over 330 training examples, with two epochs and seed 42.
LoRA stores learned weight changes separately from the base model.
The recipe uses rank 8, alpha 16, zero dropout, batch size 1, and learning rate 0.0001.
See [configs/training.json](configs/training.json) for the full recipe.

Each example contains system, user, and assistant messages.
The [exact system prompt](configs/model-system-prompt.txt) defines the model instruction.
Its SHA-256 hash is `fb828305c494f90092180ae4e0dea4290b00f7f90fbb47ec34f8dff1810f3a93`.
The dataset contains 330 training and 90 development examples from 36 source files.
Known development exposure restrictions remain part of the private dataset manifest.
The development split is not an untouched test set.

## Formats and limits

The adapter requires the pinned base model and its matching runtime.
The final BF16 GGUF contains merged weights and tokenizer information for llama.cpp inference.
The [inference guide](docs/inference.md) explains these formats and their loading paths.
Portable loading tests remain pending until the user runs the bounded GPU commands.

The project makes no independently benchmarked accuracy claim.
The model can omit facts, associate the wrong source cells, or return malformed answers.
A manually marked target does not establish automatic complete-filing recovery.
The model does not provide investment advice.
PDF processing, OCR, categorization, and proposal linking remain outside the supported scope.

The accepted dataset remains private.
Code, source filings, and model weights have separate rights and release decisions.
Finalize license terms, attribution, owner privacy review, and the exact upload list before publication under the [release guide](docs/release.md).
