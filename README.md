# ProxyBench

ProxyBench extracts one marked proxy-voting target from historical SEC Form N-PX text and HTML.
You select the target and its surrounding source context manually.
ProxyType-4B is the working name for its trained Qwen3.5-4B model.

The repository provides tools for source preparation, label review, training, and local model use.
The model weights and accepted dataset remain private and are separate inputs.

## Task and workflow

Each answer describes one voting target, including its fund or group, issuer, proposal, and disclosed votes.
The required answer format preserves source wording and distinguishes missing information from explicit statements.
The [label contract](docs/label-contract.md) defines all fourteen fields and their rules.

Follow the guide for your task:

- [Prepare data](docs/dataset.md): Select source context, review labels, and explicitly accept training examples.
- [Train and evaluate](docs/training.md): Train a model, resume a run, or evaluate a selected model.
- [Use the model](docs/inference.md): Load ProxyType-4B and extract a record from a new marked fragment.

## Model and data

The retained dataset contains 330 training examples and 90 development examples from 36 source files.
The development set is exposed reference data, not an untouched test set.
The project makes no independently benchmarked accuracy claim.

The [model card](MODEL_CARD.md) describes ProxyType-4B, its training recipe, and its model formats.
The [data notice](DATA_NOTICE.md) describes the source collection and its distribution status.

## Getting started

Start with the [preparation guide](docs/preparation.md) for software, required inputs, installation, and CPU acceptance tests.
Complete preparation before you start GPU work.
Portable model loading tests remain pending.

The CPU source-preparation package supports Python 3.11 or later without model dependencies.
The model workflow uses Python 3.12.14 and the pinned dependencies in the preparation guide.
The [repository guide](docs/repository-layout.md) explains where code, private data, models, and run outputs belong.

## Limits and release status

ProxyBench does not recover every record from a complete filing.
PDF processing, image-to-text conversion, categorization, and proposal linking remain outside this workflow.
Model answers can contain errors and need review.

Public release requires separate approval under the [release guide](docs/release.md).
Read the [security policy](SECURITY.md) before handling untrusted filings or reporting sensitive material.
