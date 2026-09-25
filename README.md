# ProxyBench

ProxyBench extracts one marked proxy-voting target from historical SEC Form N-PX text and HTML.
ProxyType-4B is the working name for its trained Qwen3.5-4B model.
The model returns source values and standardized labels under the [label contract](docs/label-contract.md).
Target selection remains a manual step.
The project does not recover every record from a complete filing.

## Setup and use

Start with the [preparation guide](docs/preparation.md).
It lists required software, installation commands, model inputs, and CPU acceptance.
The model workflow requires Python 3.12.14, a root `.venv/`, a C/C++ compiler, and an NVIDIA driver.
Install Python model packages from [requirements-training.txt](configs/requirements-training.txt).
Install llama.cpp from the pinned release with the guide's commands.
Its separate CUDA libraries use [requirements-llama-cuda.txt](configs/requirements-llama-cuda.txt).
The guide also covers OpenMP, OpenSSL, the base weights, and converter source.

The private model weights and accepted dataset are separate inputs.
Export needs the converter source identity change recorded in the [preparation plan](docs/environment-cleanup-plan.md).
Complete preparation before the user starts GPU work.
The CPU source-preparation package supports Python 3.11 or later without model dependencies.

Use the [dataset guide](docs/dataset.md) to prepare source context and accepted labels.
Read the [training guide](docs/training.md) for training, resume, and evaluation.
Read the [inference guide](docs/inference.md) for model loading and new fragments.
The [repository guide](docs/repository-layout.md) explains storage and package boundaries.

## Scope and release status

The retained dataset contains 330 training examples and 90 development examples from 36 source files.
The development set is exposed reference data, not an untouched test set.
No independently benchmarked accuracy claim accompanies this model.
PDF processing, OCR, proposal linking, and complete-filing recovery remain outside this workflow.

Keep source documents, labels, weights, and generated output outside ordinary commits.
Only explicitly accepted labels become training examples.
Use one external folder per future run.
Public release requires separate approval under the [release guide](docs/release.md).
Read the [model card](MODEL_CARD.md), [data notice](DATA_NOTICE.md), and [security policy](SECURITY.md) before distribution.
