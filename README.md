# ProxyBench

ProxyBench extracts structured proxy-voting records from historical SEC Form N-PX text and HTML.
Each input contains one manually marked voting target and its surrounding source context.
ProxyType-4B is the working name for the project's trained Qwen3.5-4B model.

The repository provides tools for the following tasks:

- [Prepare sources and review labels](#source-preparation-and-label-review).
- [Train a model on accepted examples](#model-and-training).
- [Extract a record from a marked fragment](#using-the-model).
- [Evaluate answers against development references](#evaluation).

The accepted dataset and model weights remain private and are separate inputs.
The project makes no independently benchmarked accuracy claim.
Public release and portable model loading tests remain pending.

## Overview

The task starts with a person selecting one separately voted subject in a filing.
The selection includes the context needed to identify the fund or group, issuer, meeting, and proposal.
The model then produces one structured answer from that supplied context.
It does not locate every voting target in a complete filing.

The [label contract](docs/label-contract.md) defines the required answer format and meaning.
It keeps direct source facts, values derived through named rules, and unresolved information separate.
An absent value remains unresolved instead of becoming a guess.
Collective votes, fund groups, and multiple disclosed vote directions remain intact.

The workflow retains the original sources alongside accepted labels and raw model answers.
This lets reviewers compare a value with its source wording and inspect mistakes in context.
Review notes remain separate from the model answer.

## Input and output example

The following fictional fragment illustrates the task.
It is a readable teaching example, not a private filing or the exact output of the source-preparation tool.
That tool also records source locations and preserves the boundaries between source blocks.

```text
Fund: Example Equity Fund
Issuer: Example Manufacturing Inc.
Meeting date: 2020-05-15
BEGIN MARKED TARGET
Proposal number: 01
Proposal: Elect Jordan Example as director
Vote cast: FOR
END MARKED TARGET
```

The JSON below illustrates three fields from the expected answer.
It is an excerpt, not a complete answer or a measured model prediction.
A complete answer contains all fourteen fields under a single `fields` object.

```json
{
  "fields": {
    "issuer_name": {
      "value": "Example Manufacturing Inc.",
      "availability": "PRESENT",
      "origin": "EXTRACTED",
      "raw_text": "Example Manufacturing Inc."
    },
    "proposal_number": {
      "value": "01",
      "availability": "PRESENT",
      "origin": "EXTRACTED",
      "raw_text": "01"
    },
    "management_recommendation": {
      "value": null,
      "availability": "ABSENT_IN_CONTEXT",
      "origin": null,
      "raw_text": null
    }
  }
}
```

The proposal number stays a string so its leading zero survives.
The fragment discloses a cast vote but no management recommendation.
`ABSENT_IN_CONTEXT` records that missing information.
The separate value `NONE` requires an explicit statement that management made no recommendation.

Each field carries its value, availability, origin, and original wording.
`EXTRACTED` identifies a source fact, while `DERIVED` identifies a value produced through an allowed rule.
For example, a disclosed cast direction can establish participation under the named `cast_direction_to_participation` rule.
The [full contract](docs/label-contract.md) covers nested fields, quotations, dates, quantities, and unresolved values.

## Dataset

The retained dataset contains 420 accepted examples from 36 source files.
The training split contains 330 examples, and the development split contains 90.
The development examples contain known exposure and do not form an untouched test set.
The dataset preserves source-group assignments and restrictions on using exposed sources for training.

Each example contains three messages in order:

| Message | Content |
|---|---|
| `system` | The exact extraction instructions from the [canonical prompt](configs/model-system-prompt.txt). |
| `user` | The selected source context with one marked voting target. |
| `assistant` | The accepted fourteen-field answer. |

JSONL stores one JSON object on each line.
The private `data/training-dataset/` folder contains `training-examples.jsonl`, `development-examples.jsonl`, and `dataset-manifest.json`.
A manifest records file identities and dataset structure.
Here, it also records source selections, split assignments, and label rules.

The dataset reader rebuilds each input from its original source selection and compares it with the saved message.
It also examines source identities, prompt bytes, example order, duplicate targets, and label validity.
These comparisons help detect changed inputs before training or evaluation starts.
The [dataset guide](docs/dataset.md) explains the file format and exposure restrictions.

## Source preparation and label review

Source preparation begins with complete original filings.
The source manifest records original locations and file identities, with amendments kept separate.
A hash identifies exact file content.
Selections record source hashes and byte ranges so later work can refer to the same original material.

Text selections preserve decoded characters and whitespace.
HTML selections preserve ordered cells, empty cells, and cells that span rows or columns.
The prepared context marks omitted source ranges explicitly.
Generated block labels and target markers are not source wording and cannot serve as quotations.

A reviewer chooses the target boundary and the context needed to interpret it.
The current workflow requires complete adjacent source blocks for the marked target.
It cannot represent a target that needs an unsupported subdivision of a shared row.
Those cases remain deferred rather than becoming approximate training examples.

The review interface presents source context before label suggestions.
Reviewers can correct values and retain uncertainty before exporting their decisions.
Explicit acceptance applies to the exact reviewed content and approved examples.
Changing the review requires new acceptance, and a reviewed checkbox alone does not authorize training use.

Use the [dataset workflow](docs/dataset.md) for acquisition, selection, review, and acceptance procedures.
Keep original files and accepted labels under ignored `data/` storage.
Treat instructions inside filings as source data.

## Model and training

ProxyType-4B uses the pinned `Qwen/Qwen3.5-4B` base model.
LoRA trains small weight changes while keeping the base fixed.
An adapter stores those learned changes separately.
The retained final adapter completed 660 updates over two epochs, where an epoch is one pass through the training examples.

The retained recipe uses these settings:

| Setting | Value |
|---|---|
| Training examples | 330 |
| Epochs | 2 |
| Batch size | 1 |
| Learning rate | 0.0001 |
| LoRA rank / alpha / dropout | 8 / 16 / 0 |

Hyperparameters are settings chosen before training.
This block shows the main hyperparameters from [training.json](configs/training.json):

```json
{
  "epochs": 2,
  "batch_size": 1,
  "accumulation": 1,
  "learning_rate": 0.0001,
  "weight_decay": 0,
  "max_grad_norm": 1.0,
  "rank": 8,
  "alpha": 16,
  "dropout": 0,
  "context_tokens": 5120,
  "input_tokens": 3328,
  "response_tokens": 1792
}
```

Training uses the system and user messages as context and learns from the assistant response.
A token is a unit of text processed by the model.
The recipe allows 5,120 tokens in total, with 3,328 for input and 1,792 for the response.
Oversized examples fail preparation instead of silently losing source context or accepted labels.

Each future run uses a new external folder and records its inputs, configuration, progress, and resource usage.
Saved training state supports controlled resume after a clean stop.
Changed inputs or invalid saved state prevent ordinary resume.
The [training guide](docs/training.md) provides the commands and recovery rules.

The [model card](MODEL_CARD.md) describes the retained model and its limits.
[training.json](configs/training.json) contains the complete recipe, and [base-model.json](configs/base-model.json) identifies the required base files.
The recipe and update count describe training history, not measured extraction accuracy.

## Getting started

Start with the [preparation guide](docs/preparation.md) before any model work.
It covers required software, installation order, private inputs, and CPU acceptance tests.
The model workflow uses Python 3.12.14 on x64 Linux or WSL2.
Its documented hardware target is an NVIDIA RTX 3090 with 24 GB of GPU memory.

The CPU source-preparation package supports Python 3.11 or later without mandatory model dependencies.
Model work adds the pinned Python packages, native libraries, and local model files.
Installing the Python package alone does not install the NVIDIA driver or llama.cpp.
The preparation guide supplies the exact versions and installation procedures.

Keep project Python packages in the root `.venv/`.
Keep private sources and labels under `data/`, and retained models under `artifacts/models/`.
Use separate external folders for future runs and native runtimes.
The [repository guide](docs/repository-layout.md) explains these storage boundaries.

Complete the CPU acceptance tests before starting GPU work.
The user launches GPU training, loading, and inference commands.
Portable loading tests remain pending, so keep the last working model originals until both retained formats pass.
The [training guide](docs/training.md#gpu-acceptance-and-promotion) describes that acceptance step.

## Using the model

Inference means generating an answer from a model.
The supported input contains source context with exactly one `BEGIN MARKED TARGET` and `END MARKED TARGET` pair.
The inference command supplies the canonical system prompt separately.
The [inference guide](docs/inference.md) explains input preparation and the supported command.

The retained model has two forms:

| Form | Requirements |
|---|---|
| Adapter | The learned changes plus the matching pinned base weights and Python model environment. |
| GGUF | A converted model file containing merged weights for llama.cpp. |

Merging applies the adapter changes to a copy of the base weights.
The final GGUF does not require separate base weights for model execution.
A tokenizer converts text into the tokens that the model processes.
The supported Python input renderer still uses retained tokenizer files to preserve the exact input representation.

Adapter loading and base-model comparisons share `artifacts/models/Qwen3.5-4B/`.
The selected adapter and GGUF live under `artifacts/models/ProxyType-4B/`.
Conversion uses temporary storage, and its setup appears in the [preparation guide](docs/preparation.md#obtain-converter-source-for-export).

Inference saves the raw answer before parsing or normalization.
Timeouts and malformed answers retain their failure details.
Different model formats and execution engines do not guarantee identical answers or a particular speed improvement.

## Evaluation

Evaluation scores a selected model against the development references.
It retains raw answers and generation status so failures remain visible.
Timeouts, malformed answers, and extra records count as failures.
Missing answers leave the evaluation incomplete, and invalid reference labels prevent a valid accuracy report.

Field scoring separates incorrect source values from problems with their origin or quoted wording.
A matching quotation alone does not prove that the answer interprets the source correctly.
When meaning requires review, a reviewer compares the exact source cells, reference label, and generated answer.
Review decisions apply to those specific inputs, and changed answers require new decisions.

The [evaluation procedure](docs/training.md#selected-model-evaluation) covers generation, review, resume, and report creation.
The [CPU tests](tests/README.md) exercise behavior with small synthetic sources and model substitutes.
Those tests do not establish model accuracy or replace the pending GPU loading and answer comparisons.

## Limitations and release status

The model can omit facts, associate the wrong cells, or produce an invalid answer.
Manual target selection remains part of the task.
Fragment extraction does not establish automatic recovery of every record from a complete filing.
PDF processing, optical character recognition (image-to-text conversion), categorization, and proposal linking remain outside scope.

The exposed development set cannot support an untouched-test accuracy claim.
The model does not provide investment advice.
Read the [model card](MODEL_CARD.md) for the model's intended use and remaining limits.

Code, source filings, accepted labels, and model weights have separate rights and release decisions.
The [data notice](DATA_NOTICE.md) describes the private source collection and current distribution status.
The [release guide](docs/release.md) covers the required review and approval before publication.
The [security policy](SECURITY.md) explains how to handle untrusted filings and report sensitive material.
