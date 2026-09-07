# ProxyBench

ProxyBench studies extraction of historical SEC Form N-PX proxy-voting records. It tests whether modern structured filings provide useful supervision for a small fine-tuned model. Later stages add proposal categorization and links to issuer ballot items.

This repository currently contains a Python package structure and project documentation. The extraction pipeline and training code are not implemented yet. The next stage is source exploration and annotation calibration, which establishes labeling rules through examples.

## Repository layout

The Python package lives under `src/proxybench/`. Shared documentation lives under `docs/`. Local notes, datasets, and experiment output are ignored by Git.

| Path | Purpose |
|---|---|
| `src/proxybench/` | Source retrieval, normalization, extraction, annotation, evaluation, and later training |
| `tests/` | Future behavior tests and small shareable fixtures |
| `configs/` | Shareable experiment configuration |
| `docs/` | Shared project and repository documentation |
| `data/` | Local source documents, prepared inputs, labels, and manifests |
| `artifacts/` | Local model files, run output, and reports |
| `notes/` | Local plans, specifications, review records, and stage notes |

Read the [repository guide](docs/repository-layout.md) for module boundaries and storage rules. Git retains the usage guides in `data/` and `artifacts/`, but ignores their generated contents. The entire root `notes/` directory is ignored.

## Package setup

The scaffold requires Python 3.11 or later and declares no runtime dependencies. The training environment will be selected after the pilot and a GPU compatibility test. Support for this scaffold does not establish support for future training libraries.

To create an isolated environment and install the local package, run these commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The package has no command-line entry point yet. Add runnable components during the relevant project stage. Keep environment-specific values and SEC contact information outside shared configuration.
