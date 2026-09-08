# ProxyBench

ProxyBench studies extraction of historical SEC Form N-PX proxy-voting records.
The current labeling workflow uses Sol agents to read local filings and draft structured labels.
The user edits those drafts in the existing browser review interface.
Only explicitly accepted labels can become fine-tuning examples.

The package includes local source readers, editable review pages, record validation, scoring, and local model utilities.
The [training-label guide](docs/training-labels.md) explains draft review and export of explicitly accepted labels.
Read the [local review guide](docs/local-review.md) and [repository guide](docs/repository-layout.md).
When local notes exist, start with `notes/README.md` for the current plan and label contract.

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
The root `notes/` directory is ignored by Git.
The usage guides in `data/` and `artifacts/` remain tracked exceptions.
