# ProxyBench

ProxyBench studies extraction of historical SEC Form N-PX proxy-voting records. It tests whether modern structured filings provide useful supervision for a small fine-tuned model. Later stages add proposal categorization and links to issuer ballot items.

This repository contains source retrieval, review, record validation, and scoring helpers.
The [benchmark implementation guide](docs/benchmark-implementation.md) describes the current interfaces and limits.
A bounded HTML parser and development runner now preserve and score fragment outputs.
The [extractor guide](docs/extractor-runner.md) explains their rules and limits.
Comparative evaluation and training retain separate review gates.
Local pilot labels and conversion candidates stay outside Git.

## Repository layout

The Python package lives under `src/proxybench/`. Shared documentation lives under `docs/`. Local notes, datasets, and experiment output are ignored by Git.

| Path | Purpose |
|---|---|
| `AGENTS.md` | Repository instructions for coding agents |
| `src/proxybench/` | Source retrieval, normalization, extraction, annotation, evaluation, and later training |
| `tests/` | Behavior tests and small synthetic fixtures |
| `configs/` | Shareable experiment configuration |
| `docs/` | Shared project and repository documentation |
| `data/` | Local source documents, prepared inputs, labels, and manifests |
| `artifacts/` | Local model files, run output, and reports |
| `notes/` | Local plans, specifications, review records, and stage notes |

Read the [repository guide](docs/repository-layout.md) for module boundaries and storage rules. Git retains the usage guides in `data/` and `artifacts/`, but ignores their generated contents. The entire root `notes/` directory is ignored. When local notes exist, start with `notes/README.md` for current work.

## Package setup

The scaffold requires Python 3.11 or later and declares no runtime dependencies. The training environment will be selected after the pilot and a GPU compatibility test. Support for this scaffold does not establish support for future training libraries.

To create an isolated environment and install the local package, run these commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The package has no installed command-line entry point yet. The [source calibration guide](docs/source-calibration-tools.md) describes the retrieval module and packet helper. Keep environment-specific values and SEC contact information outside shared configuration.
