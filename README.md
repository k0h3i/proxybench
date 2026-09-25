# ProxyBench

ProxyBench extracts one marked proxy-voting target from historical SEC Form N-PX text and HTML.
ProxyType-4B is the working name for its trained Qwen3.5-4B model.
The model returns source values and standardized labels under the [label contract](docs/label-contract.md).
Target selection remains a manual step.
The project does not recover every record from a complete filing.

## Setup and use

Linux and WSL are supported with Python 3.11 or later.
The CPU package has no runtime dependencies.
Install optional model libraries in an external environment before model execution.
The user starts GPU work through the commands in the [training guide](docs/training.md).

```bash
python3 -m venv ../proxybench-env
source ../proxybench-env/bin/activate
python -m pip install -e .
python -m proxybench --help
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

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
