# Historical dataset preparation

The historical adapter prepares source passages for one reviewed voting target.
It preserves original byte locations, source hashes, separate cells, and omitted ranges.
It does not discover logical records or infer labels.
The [dataset plan](../notes/training/sol-dataset-expansion/plan.md) defines the calibration, review, and release gates.

## Source preparation

Use `proxybench.annotation.historical.prepare_historical` with a reviewed selection and a frozen policy.
A selection contains ordered byte ranges, their block types, one target range, and its boundary evidence.
Supported block types are complete HTML rows, single paragraphs or headings, and plain text blocks.
Standalone HTML cells can supply context when a filing omits their opening row tag.
They cannot serve as voting targets.
The adapter rejects nested tables and partial row targets.
Defer a shared row when the adapter cannot represent its separate subjects.

The source view decodes HTML entities and removes inline tags.
It preserves empty cells and records `rowspan` and `colspan` attributes.
The [quotation policy](../notes/training/sol-dataset-expansion/label-policy-v3.md) defines all text operations.
The browser and model use the same cell text.
The model input also includes the frozen policy.

Use `check_packet` before accepting or exporting a label.
It reconstructs the packet from the original source and rejects changed evidence.
Use `literal_support` to test each quotation against one source cell.
This test cannot establish that a value has the correct meaning or association.
The reviewer must inspect every field, including nested values and uncertainty.

## Review and local publication

The existing editable browser accepts training, development, and legacy development packets when supplied with a training contract.
It displays each packet's partition, which identifies its dataset role.
The historical pilot browser still permits development packets only.
Both workflows exclude test packets.

Use `proxybench.training.dataset.publish` after source review and all applicable plan gates pass.
The function requires the exact review export, its acceptance hash, source assignments, and evidence files.
Evidence includes the original draft, review history, and a passed quality report.
The acceptance record identifies the reviewer and their delegated authority.
Review completion alone does not accept a label.

The writer rejects changed sources, unsupported quotations, overlapping targets, duplicate identities, and source groups that cross partitions.
It measures the exact chat sequence with the supplied tokenizer, without truncation.
Use the saved Qwen tokenizer on the CPU with GPU visibility disabled.
The limits are 3,328 prompt tokens, 1,792 response tokens, and 5,120 combined tokens.

Publication first writes a staging directory and tests its contents with `read_release`.
One directory rename publishes the completed release.
An interrupted publication retains its staging directory and publication lock.
Use a new output version after inspecting that incomplete attempt.
Keep all dataset directories and their evidence under ignored local storage.

## Current limits

The writer enforces mechanical bindings and sequence limits.
The coordinator still applies semantic review, family diversity, acquisition limits, audit requirements, and the full execution budget.
A passed quotation test is not a passed quality gate.
Modern XML expansion still uses its existing source adapter and needs a later admission step.
The original 15-row smoke worker and its pinned training recipe remain unchanged.

## Bounded SEC downloads

Run `python -m proxybench.sources.sec` with one URL and output path, or a local JSON selection file.
Each selection entry supplies `url`, `output`, and an optional `kind` of `index` or `filing`.
Supply `--state` for the persistent download ledger and `--identity-file` for a private runtime file containing `user_agent`.
Keep the identity file, downloaded documents, and ledger in ignored local storage.

The client permits official SEC HTTPS addresses and sends at most two requests per second.
It caps downloads at 60 filings, 4 GiB total, and 100 MiB per document.
It requires 4 GiB available memory and free disk space for the remaining allowance plus 2 GiB.
It preserves incomplete responses and stops after an access block or retry instruction.
Cached files must match their saved hashes before reuse.
