# Benchmarking

This document records reviewed results and the conditions needed to compare models.
The frozen test set contains twelve manually marked N-PX voting targets from twelve distinct filings.
It contains one filing from each SEC filing year between 2013 and 2024.
The 2024 target uses standardized disclosures in an official SEC HTML view.
Filing year and reporting period remain separate metadata.
This test measures extraction from supplied context, not discovery of every record in a complete filing.

The [dataset guide](dataset.md) describes source selection, acceptance, and exposure restrictions.
The [label contract](label-contract.md) defines the fourteen output fields.
The [test configuration](../configs/testing.json) fixes the test requirements and scoring version.
Original test files remain in private `data/raw/test/` storage.
Accepted messages and their manifest remain in private `data/testing-dataset/` storage.

## Scoring rules

The scorer, which compares answers with accepted labels, is `source-cells-v1`.
Schema validity means that the answer follows the required structure and value rules.
Source-value correctness means that all primary field comparisons pass.
These comparisons retain availability and values but generally ignore origin tags and quotations.
They normalize whitespace, selected name capitalization, and list order without removing duplicate members.
`OTHER` values still require their source wording.

Primary scoring always excludes `reporting_scope.scope_type`.
It also excludes `participation` when the reference derives participation.
Separate diagnostics cover origins, derivations, and quotations.
The field score counts fourteen top-level comparisons per record, not every nested value independently.
A field can pass its primary comparison while its quotation or derivation diagnostic fails.

Exact matching requires equality of the complete parsed label, including list order, origins, and quotations.
JSON object key order does not affect exact matching.

Semantic review means that a reviewer compares meaning with the supplied source.
Review can accept equivalent wording for `separate_subject` under the frozen rules.
It does not relax `raw_description` or remove disclosed qualifications.
Literal quotation support and semantic support remain separate requirements.
All twelve records remain in record denominators, and all 168 comparisons remain in the field denominator.
A malformed answer fails the record and all fourteen fields.

## Reviewed results

The first run finished generation on September 29, 2026.
Astra reviewed the supplied source context, accepted labels, and all twelve raw answers at xhigh reasoning effort.
Eleven valid answers required bound review decisions, which apply only to those exact inputs and answers.
The malformed answer failed automatically.
The imported decisions changed the report from `PENDING_REVIEW` to `COMPLETE`.

| Date | Run | Model | Format / backend | Schema-valid records | Source-value correct records | Correct primary fields | Exact records | Report status |
|---|---|---|---|---|---|---|---|---|
| 2026-09-29 | `test-eval-001` | ProxyType-4B | BF16/F32 GGUF | 11/12 (91.7%) | 8/12 (66.7%) | 151/168 (89.9%) | 2/12 (16.7%) | `COMPLETE` |
| 2026-09-29 | `qwen-base-test-eval-001` | Qwen3.5-4B base | BF16/F32 GGUF | 0/12 (0%) | 0/12 (0%) | 0/168 (0%) | 0/12 (0%) | `COMPLETE` |
| 2026-09-29 | `luna-low-test-eval-001` | GPT-6 Luna (requested) | Managed agent, low effort requested | 5/12 (41.7%) | 0/12 (0%) | 60/168 (35.7%) | 0/12 (0%) | `COMPLETE` |

All three reports contain zero missing answers, zero pending reviews, and zero invalid references.
All twelve answers in each local GGUF run ended normally without truncation or a generation timeout.
Luna returned twelve final messages, but its service stop reasons are unknown.
All three reports declare `valid_accuracy: true`, which means that the workflow permits reporting these scores.
It does not establish population accuracy or independence from model pretraining.
Luna used different instruction delivery and runtime controls, as described in its section below.

### Field results

Each row counts correct primary comparisons out of twelve.
Each malformed answer contributes one failure to every row.
ProxyType-4B produced one malformed answer, the base model produced twelve, and Luna produced seven.
The exclusions for derived values above still apply.

| Field | ProxyType-4B | Qwen3.5-4B base | GPT-6 Luna, agent-mediated |
|---|---|---|---|
| `reporting_scope` | 10/12 | 0/12 | 4/12 |
| `series_identifiers` | 11/12 | 0/12 | 5/12 |
| `issuer_name` | 11/12 | 0/12 | 5/12 |
| `security_identifiers` | 9/12 | 0/12 | 5/12 |
| `ticker` | 11/12 | 0/12 | 5/12 |
| `meeting_date` | 11/12 | 0/12 | 5/12 |
| `meeting_type` | 11/12 | 0/12 | 5/12 |
| `proposal_number` | 11/12 | 0/12 | 5/12 |
| `raw_description` | 11/12 | 0/12 | 5/12 |
| `separate_subject` | 11/12 | 0/12 | 3/12 |
| `proposal_source` | 11/12 | 0/12 | 3/12 |
| `participation` | 11/12 | 0/12 | 5/12 |
| `vote_components` | 11/12 | 0/12 | 1/12 |
| `management_recommendation` | 11/12 | 0/12 | 4/12 |

### Findings for `test-eval-001`

Four records failed the primary record score.
The years below identify SEC filing years, not meeting years.
The original answers remain unchanged.

| Filing year | Failure |
|---|---|
| 2017 | The answer combined five separate SEDOLs (security identifiers) into one value. |
| 2022 | The answer omitted a disclosed ISIN (an international security identifier). |
| 2023 | The answer omitted the final word of a fund name on a continuation line. |
| 2024 | The vote quantity lacked the required nested amount and unit fields. |

The 2024 answer was valid JSON but failed the label schema.
The validator reported `vote_components[0].quantity: unexpected or missing keys`.
It ended normally after 938 output tokens, so this failure was not a response-limit cutoff.

One wording difference in the 2019 `separate_subject` passed semantic review under the existing equivalence rule.
No `OTHER` equivalences applied.
Several otherwise correct records differed from references through supported container quotations or null quotations.
Those differences help explain why exact matching was lower than primary source-value correctness.

The report counts two origin diagnostics and two derivation diagnostics.
The origin diagnostics reflect changed identifier membership, not invented origin tag values.
Both derivation diagnostics reflect quotation differences inside `reporting_scope.scope_type`, not incorrect scope type values.
Among eleven schema-valid answers, no quotation failed the literal source-cell test.
One quotation failed semantic support because its source passage disclosed separate identifiers, not one combined identifier.
The malformed answer did not receive quotation review.

## First run conditions and evidence

ProxyType-4B uses the final adapter after 660 updates over two training epochs.
Its base is Qwen3.5-4B at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
GGUF is a model file format for llama.cpp.
The evaluated GGUF contains merged BF16 and F32 weights.
BF16 and F32 specify numeric weight precision.
The run used llama.cpp release `b10909-mix-bea84f7`, source commit `329b6160f513915f1c607dbfae3d5ce864a64a4f`, on an NVIDIA GeForce RTX 3090.

Generation used temperature 0, top-k 1, top-p 1, min-p 0, seed 42, and repetition penalty 1.
Presence and frequency penalties were zero.
The run disabled prompt-cache reuse and used one server load for twelve separate requests.
Input, response, and context limits were 3,328, 1,792, and 5,120 tokens respectively.
The per-request timeout was 60 seconds.

Recorded worker execution was 180.65 seconds, including model loading, requests, and cleanup.
This duration excludes semantic review and CPU report preparation.
The answers contain 10,376 output tokens, including end tokens.
The recorded decode-rate metric is null, so this document makes no measured decoding-throughput claim.

Hashes identify exact content or evaluator identities.
Evaluator identities below are bound representations, not hashes of raw JSON file bytes.
Private evidence remains in `../proxybench-runs/test-eval-001/`, relative to the repository root.
Its retained files include `run.json`, `evaluation/inputs.json`, `evaluation/answers.jsonl`, `evaluation/report.json`, and `evaluation/review/decisions.json`.

| Evidence | Identity or SHA-256 |
|---|---|
| Evaluator dataset identity | `38a2692a7c8d3326ed1b6e0648a01879b4bd9cfa3478119d11d12df630ca5cf9` |
| Evaluator prompt identity | `a49ac8190d23ccf78dbfa3139060b076ace854636c5f725ddaee1ea612a88181` |
| Evaluator runtime identity | `5ba6b6be0c6b77f62aa727fd720f0f7f98391f4fef08c9bad8dad6b57064e9ce` |
| Model GGUF SHA-256 | `530fc41aeeee46c251a7ddf3f780f307537ffab07a1a9e9bd50297b62b746bf2` |
| `test-examples.jsonl` SHA-256 | `90c3751bc5dab8741b5a7566528607d87fe8b74a5709302e481e6ae817fd5125` |
| Canonical prompt file SHA-256 | `fb828305c494f90092180ae4e0dea4290b00f7f90fbb47ec34f8dff1810f3a93` |
| Raw `answers.jsonl` SHA-256 | `5b7bb704157faeebd659e9da4983ef0a71c65e6f6ca349596914bd6a8b01f1f3` |
| Final `report.json` SHA-256 | `54fc759e1acb21e6324193fbca1ab3ea1b941dca3d55e961c573416bae377c87` |
| Imported `decisions.json` SHA-256 | `eb1f2c382a703cb8f8d54d70f5b8ef48fd96cd7d5b570d0ba1fa011bfffc08bd` |

## Base Qwen comparison

The base run finished generation on September 29, 2026.
Astra audited all twelve original answers at xhigh reasoning effort and found no scoring or binding blocker.
The run used the retained Qwen3.5-4B base revision listed above without the trained adapter weights.
Its model file is `artifacts/models/Qwen3.5-4B/model-bf16.gguf`.
The shared tokenizer files render the same inputs without loading the adapter weights.

Both runs contain identical source cells, case order, references, system messages, and user messages.
Their actual rendered prompts and prompt token IDs also match.
The native runtime, RTX 3090, generation configuration, token limits, and cache policy remain the same.
The saved effective configuration differs only by model path.
The runtime identity differs because it also binds that configuration.
This is a fixed-task comparison with greedy generation and thinking disabled, not an optimized comparison of general model capability.

All twelve base answers violate the output contract.
Eight are bare JSON objects that fail the label schema.
Four include Markdown fences and fail strict JSON parsing.
Their contents also fail the schema when Astra inspects them without the fences for diagnosis only.
Neither review nor scoring removes fences or repairs the original answers.

Every field must include `availability`, which records whether its value is present in the source, missing, or uncertain.
For example, `"availability": "PRESENT"` means that the source provides the value.
`"availability": "ABSENT_IN_CONTEXT"` means that the supplied context does not provide it.
The 2013 base answer omitted this required entry from `series_identifiers`.

The table groups the first validation failure, not every defect in each answer.
Its years identify SEC filing years.
All twelve outputs remain failures under the original scoring rules.

| Filing years | Records | First failure |
|---|---|---|
| 2013, 2017, 2018, 2019, 2024 | 5 | `series_identifiers: unexpected or missing keys` |
| 2015, 2016, 2020 | 3 | `reporting_scope: unexpected or missing keys` |
| 2014, 2021, 2022, 2023 | 4 | Markdown fences prevent strict JSON parsing. |

The report is legitimately `COMPLETE` because every answer is terminal and none passes the schema gate.
There are no schema-valid answers that require bound semantic review decisions.
Astra's separate audit does not change the raw answers or their automatic failure scores.
The zero origin, derivation, and quotation diagnostics do not establish correctness because those comparisons require schema-valid answers.
The 0/168 field score does not mean that every extracted fact was wrong.

An answer can contain the correct issuer name but omit `availability` in another field.
The scorer rejects the entire answer and marks all fourteen fields as failures before comparing their individual values.
All twelve base answers failed the required format, so none reached individual field comparison.
Their 0/168 score therefore reflects format failures, not 168 source facts compared and found wrong.

Astra also found incorrect source interpretations inside the incompatible answers.
Examples included a For vote instead of the disclosed Against vote in 2016.
The 2020 and 2023 answers asserted nonvoting despite explicit cast directions.
The 2013 and 2022 answers treated disclosed management alignment as a management recommendation.
These findings explain additional failures but do not create unofficial partial scores.

Recorded worker execution was 187.24 seconds, excluding the separate audit and CPU report preparation.
The answers contain 11,361 output tokens, including end tokens.
The largest answer used 1,251 tokens, below the 1,792-token response limit.
All twelve ended normally without truncation or a timeout.
All recorded decode-rate metrics are null.

Private evidence remains in `../proxybench-runs/qwen-base-test-eval-001/`, relative to the repository root.
It includes the raw answers, input records, report, capture records, and runtime configuration.
No review-decision import is required for this run.
Its dataset, prompt, and scorer identities match the first run.

| Evidence | Identity or SHA-256 |
|---|---|
| Evaluator runtime identity | `07e71aa4fc5c500452901b0bf3551cf650dc940898012d2344d8d8fac0bb5f4e` |
| Base model GGUF SHA-256 | `d72a5b5169186b4e889182e97048c674b537c66cefcbc80d65ceda9898a751e5` |
| `inputs.json` SHA-256 | `4bd885d18f7ebcdd209974a910589c515c43e2bb7137193cb06e9bf9faf34fa6` |
| Raw `answers.jsonl` SHA-256 | `dc4d10d30962bd20008a21295ae2736e81fa7948fa4e9e0c74a030de971136ad` |
| Final `report.json` SHA-256 | `fd4b60b05e5bfbce79c9acee0293f8e386748cecbc3b67d2ca770b20896711f6` |

## Luna agent-mediated evaluation

The Luna run finished generation on September 29, 2026.
It requested `gpt-6-luna` with `low` reasoning effort.
Astra reviewed all twelve untouched answers at xhigh effort against the original accepted references and supplied source cells.
Five schema-valid answers required new bound decisions, while seven malformed answers failed automatically.
The imported decisions changed the scoring report from `PENDING_REVIEW` to `COMPLETE`.

### Separate generation workspace

Generation used a separate, input-only workspace outside this repository: `../proxybench-runs/luna-low-test-eval-001/`.
The package contained the exact canonical prompt, twelve frozen source contexts, the label contract, and run instructions.
It contained no accepted labels, previous predictions, or previous scores.
Scoring and review used the protected references afterward in the separate `../proxybench-runs/luna-low-test-eval-001-review/` folder.
The original generation files remain unchanged.

The coordinator reports a fresh chat and one new extraction agent per case with `fork_turns="none"`.
It reports one attempt per case, at most two concurrent agents, no corrective messages, and no retries or answer repairs.
No tool calls or protocol violations were reported.
These controls are coordinator-reported, not independently confirmed by complete service traces.
The saved transport files represent spawn responses and final messages, not the full service dispatch or tool history.

Independent file comparisons confirmed all twelve source strings, their order, and the canonical prompt component against the frozen dataset.
Each captured request matches the fixed envelope, and each captured answer matches its saved final-message payload.
All package hashes match, and the capture inventory contains no missing or extra cases.
The envelope prohibits tools and requests only the first final extraction answer.
It carries the canonical prompt as text inside a delegated message, not as a separately controlled system-role message.
Extraction agents also receive platform instructions whose contents are unknown.

Actual model revision, effective effort, tokenizer, sampling, context and output limits, token usage, stop reasons, and cost remain unknown.
The requested model and effort identify the run, not independently verified service internals.
The capture status `COMPLETE` means that a final message arrived without a reported protocol violation.
It does not prove valid JSON, absence of truncation, or successful extraction.
The review run's initialized resource counter does not measure generation time or cost.

This is an agent-mediated evaluation in a separate workspace, not a verified isolated environment or an independently administered blind benchmark.
Workspace placement alone does not establish a technical barrier to other files or hidden context.
The frozen source task and scorer match the local runs, but instruction delivery and generation controls do not.
The scores therefore do not establish an identical-runtime comparison of bare-model capability, speed, or cost.

### Failures and diagnostics

All five schema-valid answers fail at least one primary field comparison.
The seven malformed answers contribute 98 automatic field failures.
The five valid answers contribute ten additional primary field failures, for 60 correct comparisons out of 168.
This does not mean that every fact inside a malformed answer was wrong.
No repaired-output or unofficial partial score replaces the frozen score.

| Filing years | Records | First validation failure |
|---|---|---|
| 2014 | 1 | The JSON ends inside an unterminated property string. |
| 2017 | 1 | An absent vote quantity has a nonnull nested value. |
| 2018, 2019, 2020, 2021 | 4 | Absent `series_identifiers` contain `[]` instead of null. |
| 2024 | 1 | `participation` uses `DERIVED` as an invalid availability value. |

The 2014 malformed JSON is not a proven response-limit cutoff because the service stop reason and output limit are unknown.
The table lists the first failure, not every defect.
For example, the 2017 answer separates all five SEDOLs correctly but still fails the quantity state rules.
The 2020 answer preserves the disclosed Withhold vote but still fails the absent-identifier rule.

| Filing year | Primary failures among schema-valid answers |
|---|---|
| 2013 | Expanded `Mgmt` to `Management`, marked an undisclosed quantity as present, and inferred a recommendation from vote/alignment wording. |
| 2015 | Marked an undisclosed quantity as a present container with unresolved children. |
| 2016 | Marked an undisclosed quantity as a present container with unresolved children. |
| 2022 | Omitted the advisory qualification from the subject, expanded `Mgmt`, and marked an undisclosed quantity as present. |
| 2023 | Omitted `Portfolio` from the fund-name continuation and the requirement action from the subject. |

Astra rejected subject equivalence for the 2022 and 2023 omissions because they remove meaning from the disclosed subjects.
The five valid answers produce eight origin diagnostics, two derivation diagnostics, two literal quotation failures, and one semantic quotation failure.
Origin diagnostics include incorrect tags and added quantity structure, not just incorrect source values.
Both derivation diagnostics reflect quotation differences in `reporting_scope.scope_type`, not an incorrect scope type.
The literal failures concern the 2013 subject quotation and the 2016 proposal-number quotation.
The semantic failure concerns the 2013 recommendation quotation, which belongs to vote/alignment wording instead.

### Retained evidence

Generation evidence remains in `luna-low-test-eval-001/outputs/` under the external runs folder.
It includes twelve case captures, twelve transport representations, and `run-metadata.json`.
Protected scoring evidence remains in `luna-low-test-eval-001-review/evaluation/` under that same parent folder.
It includes rebuilt inputs, raw-answer wrappers, the capture audit, the report, and new bound review decisions.
The dataset and canonical-prompt component identities match the earlier runs.
The model-request identity hashes the requested alias and unknown revision, not model weights.

The capture-inventory binding identifies the audited hashes of all 41 generation files.
The evaluator runtime identity binds the imported capture configuration, not a verified service runtime build.
File hashes below identify retained raw bytes.
Evaluator identities and the inventory binding identify structured representations.

| Evidence | Identity or SHA-256 |
|---|---|
| Input package manifest SHA-256 | `7efc365ee1852742cc1b2075d513ad3ce7b315fb81802c904fb4a615fafbecfa` |
| Coordinator `run-metadata.json` SHA-256 | `91e6fc97dc8e709a43894fdeb38f32ec21e0a9c8d0b8a61b467f0b163039bb4b` |
| Capture-inventory binding | `668e2e3904538e63404ae8f582a3bc33f6558ed526217c2418f94957d435e15f` |
| Evaluator runtime identity | `1668ab44e34c951dbc4a19b71e09536b13896a059ca5a3579d717555c70b1add` |
| Model-request identity | `1a439614145265f9822f60e142680fce2835ab2af7f81837a6d9c12a09a09baf` |
| Scoring `inputs.json` SHA-256 | `b3e924df8428b136853b441aa0b63c766d7c3f9f6674fa0215c8e1d98bc4581b` |
| Raw-answer wrappers `answers.jsonl` SHA-256 | `3ad7e6506e30d3c53bad05e08eb0c415bf6816f177d65e110efcafccc602b513` |
| Final `report.json` SHA-256 | `1c60d28d5eeb0faf59b80312e60489d1bbb1b1b31ce3b1dcf4d19fbeefcbd552` |
| Imported `decisions.json` SHA-256 | `1b8d331fa1d3ce29a00defa67a25e633dea976442765e45e3dc9880e356a847e` |
| Retained `astra-audit.json` SHA-256 | `54312da3e433fa0be10f14f33827743b96d73d477147057958b3d7bab5389a9e` |

## Future model comparisons

Use the [evaluation procedure](training.md#selected-model-evaluation) for each new run.
Use a new external run folder for each model evaluation.
Keep the frozen source content, case order, system instructions, reference labels, and scoring rules unchanged.
Record each model revision, tokenizer, weight format, runtime identity, and effective generation configuration.
Disclose model-specific input formatting and any differences in limits or sampling.
Do not silently truncate source context to fit another model.

Retain raw answers and terminal failure details before parsing.
Review each new model's answers against their exact source context and reference labels.
Do not reuse decisions from another model's answers.
Append a results row only after scoring and required review finish.
Include missing answers, pending reviews, invalid references, failure counts, run conditions, and evidence hashes with each new entry.
If review remains pending, record its status without presenting partial scores as final results.

Keep the twelve-record and 168-field denominators for comparable runs.
Count malformed answers, terminal failures, and extra records as failures.
Do not remove difficult records or substitute a successful retry without disclosure.
If timing comparisons use different hardware, runtimes, cache policies, or limits, disclose those differences.
Token totals also depend on the model's tokenizer.
Preserve earlier rows and evidence rather than replacing them with later results.

## Interpretation limits

This deliberately varied twelve-record batch is not a random sample of N-PX filings.
One record changes the record score by about 8.3 percentage points.
The test does not establish exclusion from base-model pretraining.
Its independence is limited to audited project exposure.
Astra's review is AI-assisted semantic review, not a blind external benchmark.
The exposed 90-record development set remains separate from these results.

Do not use these results to select prompts, models, thresholds, or replacement records.
If test findings guide development, identify subsequent runs as regression evaluations rather than untouched tests.
Regression evaluations measure behavior after changes on an already inspected set.
This log contains reviewed results for ProxyType-4B, its Qwen3.5-4B base, and the separately described Luna agent run.
The source task and scoring rules remain frozen, while the Luna execution conditions differ.
Public release remains a separate gate under the [release guide](release.md).
