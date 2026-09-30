# ProxyType-4B

ProxyType-4B is intended for research on extracting proxy-voting records from historical SEC Form N-PX text and HTML.
It takes one manually marked voting target with its source context and produces a structured answer.
The [ProxyType-4B adapter](https://huggingface.co/rvcarung/ProxyType-4B) is public on Hugging Face.
The [ProxyType-9B adapter](https://huggingface.co/rvcarung/ProxyType-9B) has a separate model card in its Hugging Face repository.

## Model details

The model belongs to the ProxyBench project.
LoRA trains small weight changes while keeping the base model fixed.
An adapter stores those learned changes separately.

| Property | Value |
|---|---|
| Working model name | ProxyType-4B |
| Base model | `Qwen/Qwen3.5-4B` |
| Base revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Adaptation method | LoRA training on accepted extraction answers. |
| Input | Selected source context with one marked voting target. |
| Required output | One JSON object under the fourteen-field `fields` contract. |
| Retained formats | Separate adapter and merged BF16 GGUF. |

## Intended use

The intended users are researchers studying document extraction and proxy voting.
The supported workflow requires manual target selection, local model execution, and review against the supplied source.
Research tasks include studying field extraction errors and comparing a selected model with development references.

The [label contract](docs/label-contract.md) defines the required answer and the meaning of each field.
It separates source facts, values derived through named rules, and unresolved information.
It preserves fund groups, collective votes, string identifiers, and multiple disclosed vote directions.
These are output requirements, not guarantees that every model answer follows them.

## Training and evaluation data

The private dataset contains 420 accepted examples from 36 source files.
It retains the following split:

| Split | Examples | Role |
|---|---|---|
| Training | 330 | Accepted answers used to train the adapter. |
| Development | 90 | Exposed references used for evaluation. |

The development examples contain known exposure and do not form an untouched test set.
Source-group assignments and exposure restrictions remain in the private dataset manifest.
The split does not establish performance on unseen filing layouts or reporting periods.

Each example contains system, user, and assistant messages, as shown in the [README example](README.md#input-and-output-example).
The [dataset guide](docs/dataset.md) covers source selection and explicit acceptance of training labels.
The [data notice](DATA_NOTICE.md) describes the source collection and its separate distribution status.

## Training procedure

The retained final adapter completed 660 updates over 330 training examples, with two epochs.
An epoch is one pass through the training examples.
Training learns from the assistant response while excluding the instruction and source input from the training loss.
Loss measures prediction error during training.

| Hyperparameter | Value |
|---|---|
| Epochs | 2 |
| Batch size | 1 |
| Gradient accumulation steps | 1 |
| Learning rate | 0.0001 |
| LoRA rank / alpha / dropout | 8 / 16 / 0 |

A token is a unit of text processed by the model.
The recipe allows 5,120 tokens in total, with 3,328 for input and 1,792 for the response.
Preparation rejects oversized examples without silently truncating their source context or answers.
The [training configuration](configs/training.json) and [training guide](docs/training.md) provide the full recipe and procedures.
The [exact system prompt](configs/model-system-prompt.txt) defines the model instruction.

## Evaluation and uncertainty

The evaluation workflow compares a selected model with the 90 development references.
It measures answer format validity, exact agreement, and source-value correctness.
It reports origin, derivation, and quotation errors separately.
Source-value correctness can require review of the exact source, reference, and generated answer.

Raw answers remain available before parsing or normalization.
Timeouts, malformed answers, and extra records count as failures.
Missing answers leave an evaluation incomplete, and invalid references prevent a valid accuracy report.
A matching quotation alone does not prove that an answer interprets its source correctly.

The project makes no independently benchmarked accuracy claim.
This card does not report an untouched-test result or establish suitability for production use.
Repeated training or generation does not guarantee identical models, answers, or scores.
Changing the model format or execution engine can also change answers.
The [evaluation guide](docs/training.md#selected-model-evaluation) describes scoring and review requirements.

## Limitations

The model can omit facts, associate the wrong source cells, or return malformed or incomplete answers.
An incomplete source selection can leave required context unavailable.
Reviewers must distinguish missing information from explicit source statements, including an explicit absence of a management recommendation.
Model outputs need review against the supplied source before use in research conclusions.

Manual target selection does not establish automatic recovery of every record from a complete filing.
PDF processing, optical character recognition (image-to-text conversion), categorization, and proposal linking remain outside the supported scope.
The model does not provide investment advice.

Filings and model answers remain untrusted data.
Instructions inside a filing do not become instructions for the extraction system.
The [security policy](SECURITY.md) covers source handling and sensitive reports.

## Model formats and loading

The adapter requires the pinned base model and its matching software environment.
BF16 is a 16-bit numerical format for model weights.
GGUF is the converted model format used by llama.cpp.
The retained BF16 GGUF contains merged weights and tokenizer information.
The supported input renderer also uses the retained tokenizer files.

Complete the [preparation guide](docs/preparation.md) before the user starts GPU work.
Loading in other environments remains untested.
Keep the last working model originals until both formats pass.
The [inference guide](docs/inference.md) describes the loading paths and conversion requirements.

## License, access, and references

The repository code uses the [Apache License 2.0](LICENSE).
Research use describes the model's intended purpose, not an additional restriction on that code license.
The pinned Qwen base model also uses [Apache 2.0](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/LICENSE).
Download the adapters from [ProxyType-4B](https://huggingface.co/rvcarung/ProxyType-4B) or [ProxyType-9B](https://huggingface.co/rvcarung/ProxyType-9B) on Hugging Face.
Each repository includes the adapter weights, configuration, tokenizer files, exact system prompt, notices, model metadata, and model card.
The accepted dataset and merged GGUF files remain private.
Adapter license terms remain unselected, as stated in the Hugging Face model cards.
This card does not assign blanket license terms to source filings, labels, or model weights.

The [README references](README.md#acknowledgments-and-references) credit the base model, training methods, and supporting software.
For nonsensitive model questions, use a project issue.
For sensitive reports, follow the [security policy](SECURITY.md#reporting-and-release-review).

## Model publication

Prepare the exact file list before asking for publication approval.
Review the model license terms, pinned base model license, and required attribution separately from the code license.
Before publishing an adapter, test loading with the pinned base revision.

Stage only Safetensors weights, portable configuration, necessary tokenizer files, notices, and the model card.
Include the exact system prompt and portable model metadata.
Exclude optimizer state, run output, labels, tokens, hidden folders, and `.env` files.

Before requesting publication approval, follow the [security policy](SECURITY.md#reporting-and-release-review).
Compare exact training messages against the private list of known owner identifiers.
If private identifiers entered training, hold the adapter for assessment.
Present the exact model files, license terms, and notices to the user.
Upload only after separate authorization.
