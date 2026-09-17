# ProxyBench

ProxyBench studies extraction of historical SEC Form N-PX proxy-voting records.
The current labeling workflow uses Sol agents to read local filings and draft structured labels.
The user edits those drafts in the existing browser review interface.
Only explicitly accepted labels can become fine-tuning examples.

The package includes local source readers, editable review pages, record validation, scoring, and local model utilities.
The [training-label guide](docs/training-labels.md) explains draft review and export of explicitly accepted labels.
Read the [local review guide](docs/local-review.md) and [repository guide](docs/repository-layout.md).
The [Qwen3.5-4B guide](docs/qwen35-4b-smoke.md) describes the bounded local training test.
The [historical pilot guide](docs/qwen35-4b-historical-pilot.md) gives the CPU-tested commands for the accepted 96/24 dataset and 192-update recipe.
When local notes exist, start with `notes/README.md` for the current plan and label contract.

## Faster local inference

On 15 development examples, the merged llama.cpp model reduced median answer time from 137.72 to 18.92 seconds.
Both methods used the same RTX 3090 GPU, and existing extraction errors remained.
The [inference guide](docs/qwen35-4b-inference.md) explains the change with a before-and-after diagram, measured results, and limitations.

## Setup and tests

The package requires Python 3.11 or later and declares no runtime dependencies.
GPU training libraries require a separate environment.
From the repository root, run these commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Keep original filings, labels, model weights, and generated output outside ordinary commits.
The private repository includes [project notes](notes/README.md).
Before making it public, follow the [public-release reminder](notes/project/before-going-public.md).
The usage guides in `data/` and `artifacts/` remain tracked exceptions.
