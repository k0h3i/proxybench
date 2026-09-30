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

## Scoring rules

The scorer compares answers with accepted labels and checks quotations against individual source cells.
[`source-cells-v1`](../src/proxybench/evaluation/answers.py) identifies version 1 of these scoring rules.

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

`Exact matches` counts answers that match the whole accepted reference answer after the evaluator reads the JSON.
All fourteen fields and their nested fields must match.
This includes values, availability states, origin tags, quotations, and list order.
Capitalization and spaces inside text values must also match.
JSON formatting and object key order do not affect the result.
An answer can contain correct facts but fail exact matching because a quotation or origin tag differs.

Semantic review means that a reviewer compares meaning with the supplied source.
Review can accept equivalent wording for `separate_subject` under the frozen rules.
It does not relax `raw_description` or remove disclosed qualifications.
Literal quotation support and semantic support remain separate requirements.
All twelve records remain in record denominators, and all 168 comparisons remain in the field denominator.
A malformed answer fails the record and all fourteen fields.

## Reviewed results

An Astra agent with `xhigh` reasoning effort reviewed the source context, accepted labels, and raw model answers.
Review decisions apply only to the exact reviewed inputs and answers.
Malformed answers fail automatically.

GGUF is a model file format for llama.cpp.
BF16 and F32 describe weight precision.

| Model | Setup | Valid answers | Correct records | Correct fields | Exact matches |
|---|---|---|---|---|---|
| Qwen3.5-4B | BF16/F32 GGUF | 0/12 (0%) | 0/12 (0%) | 0/168 (0%) | 0/12 (0%) |
| ProxyType-4B | BF16/F32 GGUF | 11/12 (91.7%) | 8/12 (66.7%) | 151/168 (89.9%) | 2/12 (16.7%) |
| ProxyType-9B | BF16/F32 GGUF | 11/12 (91.7%) | 9/12 (75%) | 152/168 (90.5%) | 0/12 (0%) |
| GPT-6 Luna | API-low | 5/12 (41.7%) | 0/12 (0%) | 60/168 (35.7%) | 0/12 (0%) |
| GPT-6 Sol | API-low | 11/12 (91.7%) | 0/12 (0%) | 140/168 (83.3%) | 0/12 (0%) |

All five evaluations have all expected answers, completed reviews, and valid reference labels.
Every local GGUF answer ended without truncation or a generation timeout.
Luna and Sol returned all expected answers, but their reasons for stopping are unknown.
These results do not establish accuracy across all N-PX filings or independence from model pretraining.
The agent runs delivered instructions differently and used different runtime controls from the local runs, as described below.

### Field results

Each row counts correct primary comparisons out of twelve.
Each malformed answer contributes one failure to every row.
Qwen3.5-4B produced twelve malformed answers, and GPT-6 Luna produced seven.
ProxyType-4B, ProxyType-9B, and GPT-6 Sol each produced one malformed answer.
The exclusions for derived values above still apply.

| Field | Qwen3.5-4B | ProxyType-4B | ProxyType-9B | GPT-6 Luna | GPT-6 Sol |
|---|---|---|---|---|---|
| `reporting_scope` | 0/12 | 10/12 | 10/12 | 4/12 | 11/12 |
| `series_identifiers` | 0/12 | 11/12 | 11/12 | 5/12 | 10/12 |
| `issuer_name` | 0/12 | 11/12 | 11/12 | 5/12 | 11/12 |
| `security_identifiers` | 0/12 | 9/12 | 10/12 | 5/12 | 11/12 |
| `ticker` | 0/12 | 11/12 | 11/12 | 5/12 | 11/12 |
| `meeting_date` | 0/12 | 11/12 | 11/12 | 5/12 | 11/12 |
| `meeting_type` | 0/12 | 11/12 | 11/12 | 5/12 | 11/12 |
| `proposal_number` | 0/12 | 11/12 | 11/12 | 5/12 | 11/12 |
| `raw_description` | 0/12 | 11/12 | 11/12 | 5/12 | 11/12 |
| `separate_subject` | 0/12 | 11/12 | 11/12 | 3/12 | 9/12 |
| `proposal_source` | 0/12 | 11/12 | 11/12 | 3/12 | 11/12 |
| `participation` | 0/12 | 11/12 | 11/12 | 5/12 | 11/12 |
| `vote_components` | 0/12 | 11/12 | 11/12 | 1/12 | 0/12 |
| `management_recommendation` | 0/12 | 11/12 | 11/12 | 4/12 | 11/12 |

### Findings for ProxyType-4B

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

## Qwen3.5-4B comparison

Qwen3.5-4B ran without a trained adapter.
It uses the same pinned base as ProxyType-4B.
An agent reviewed all twelve original answers with `xhigh` reasoning effort.

The source cells, case order, references, system instructions, rendered prompts, and prompt token IDs match the ProxyType-4B comparison.
Both comparisons use the same llama.cpp runtime, NVIDIA RTX 3090, generation configuration, token limits, and cache policy.
Generation selected the most likely next token, with thinking disabled.
These results describe this extraction task under those conditions.

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

All twelve answers failed the output format, so none required semantic review before scoring.
The separate review leaves the original answers and automatic failures unchanged.
Origin, derivation, and quotation diagnostics require valid answers.
Their zero counts therefore do not establish correctness.

An answer can contain the correct issuer name but omit `availability` in another field.
The scorer rejects the entire answer and marks all fourteen fields as failures before comparing their individual values.
All twelve base answers failed the required format, so none reached individual field comparison.
Their 0/168 score therefore reflects format failures, not 168 source facts compared and found wrong.

Astra also found incorrect source interpretations inside the incompatible answers.
Examples included a For vote instead of the disclosed Against vote in 2016.
The 2020 and 2023 answers asserted nonvoting despite explicit cast directions.
The 2013 and 2022 answers treated disclosed management alignment as a management recommendation.
These findings explain additional failures but do not create unofficial partial scores.

## ProxyType-9B comparison

An agent reviewed all twelve raw answers against the source context and accepted labels with `xhigh` reasoning effort.
Eleven answers passed format checks, and one failed automatically.
Review decisions completed scoring.
No subject or `OTHER` equivalences applied.
The original predictions and labels remain unchanged.

ProxyType-9B uses Qwen3.5-9B with an adapter trained for 660 updates over two epochs.
The [base model configuration](../configs/base-model-9b.json) identifies the pinned model revision.

The dataset, case order, source cells, references, system instructions, rendered prompts, and prompt token IDs match the ProxyType-4B comparison.
Both comparisons use the same llama.cpp runtime, NVIDIA RTX 3090, generation controls, token limits, request timeout, and cache policy.
The configuration differs only in the model and tokenizer.

### Failures and diagnostics

Three records fail the primary record score.
The malformed answer contributes fourteen automatic field failures, and the two valid failing answers contribute one field failure each.
This leaves 152 correct comparisons out of 168.

| Filing year | Failure |
|---|---|
| 2017 | Omitted `Security` identifier `G021A5106` and combined five separate SEDOLs into one value. |
| 2023 | Omitted `Portfolio` from the continued fund name. |
| 2024 | Absent ISIN and FIGI values contain `-` and use `EXTRACTED` origin instead of null value and origin. |

The 2024 answer is valid JSON but fails the label schema.
Its first validator error is `security_identifiers[1].value: unresolved value and origin must be null`.
It ended normally after 1,305 output tokens, below the 1,792-token limit, without truncation.
Astra also found that this answer misplaces the series identifier among security identifiers and uses the voting frequency as a quantity unit.
These additional findings remain diagnostic observations, not unofficial partial scores.

All nine source-value-correct answers differ from the accepted references only in `raw_text`.
Their values, availability states, origin tags, and list order match.
Supported quotation differences and null quotations therefore leave zero exact records without making those nine records factually wrong.

The report counts one origin diagnostic, three derivation diagnostics, no literal quotation failures, and two semantic quotation failures.
The origin diagnostic reflects changed security-identifier membership, not an incorrect origin tag.
All three derivation diagnostics reflect quotation differences in `reporting_scope.scope_type`, whose values and origins remain correct.
The semantic failures concern the combined SEDOL quotation and the incomplete fund-name quotation.
Astra also inspected the malformed answer's quotations, but that answer receives no imported quotation decision or partial field credit.

## GPT-6 Luna comparison

The run requested `gpt-6-luna` with `low` reasoning effort.
An agent reviewed all twelve raw answers against the source context and accepted labels with `xhigh` reasoning effort.
Five answers passed format checks, and seven failed automatically.
Review decisions completed scoring.

### Generation method

Generation used the canonical prompt, frozen source contexts, label contract, and run instructions.
The input excluded accepted labels, previous predictions, and scores.
An independent comparison confirmed that the source strings, case order, and canonical prompt matched the frozen dataset.

The coordinator reported a fresh chat, one new extraction agent per case, and one attempt per case.
It reported at most two concurrent agents and no retries, repairs, corrective messages, tool calls, or protocol violations.
The review lacked complete service traces to confirm those reports.

The agents received the canonical prompt as text inside a delegated message.
They also received platform instructions whose contents were unknown.
The actual model revision, effective effort, tokenizer, sampling, token limits, usage, stop reasons, and cost remain unknown.
The source task and scorer match the local comparisons, but instruction delivery and generation controls differ.
These scores do not establish equal conditions for comparing model capability, speed, or cost.

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

## GPT-6 Sol comparison

The run requested `gpt-6-sol` with `low` reasoning effort.
An agent reviewed all twelve raw answers against the source context and accepted labels with `xhigh` reasoning effort.
Eleven answers passed format checks, and one failed automatically.
Review decisions completed scoring.

### Generation method

Generation used the canonical prompt, frozen source contexts, label contract, and run instructions.
The input excluded accepted labels, previous predictions, and scores.
An independent comparison confirmed that the source strings, case order, and canonical prompt matched the frozen dataset.

The coordinator reported a fresh chat, one new extraction agent per case, and one attempt per case.
It reported a concurrency limit of three, with at most two agents active.
It reported no retries, repairs, corrective messages, tool calls, or protocol violations.
The review lacked complete service traces to confirm those reports.

The agents received the canonical prompt as text inside a delegated message.
They also received platform instructions whose contents were unknown.
The actual model revision, effective effort, tokenizer, sampling, token limits, usage, stop reasons, and cost remain unknown.
The source task and scorer match the local comparisons, but instruction delivery and generation controls differ.
These scores do not establish equal conditions for comparing model capability, speed, or cost.

### Failures and diagnostics

All twelve records fail the primary record score, but 140 of 168 primary field comparisons pass.
A record needs every primary field to pass, so one quantity-state mistake can fail an otherwise correct record.
Zero correct records therefore does not mean that every extracted fact was wrong.
The malformed 2021 answer contributes fourteen automatic field failures.
The eleven schema-valid answers contribute fourteen further primary failures.

| Filing years | Primary failure |
|---|---|
| 2013–2020, 2022, 2023 | An undisclosed quantity is marked `PRESENT` with unresolved amount and unit children. |
| 2021 | An absent quantity contains a nonnull nested object, which violates the required structure. |
| 2022 | The subject omits the disclosed advisory-vote qualification. |
| 2024 | The subject omits the current approval's nonbinding advisory qualification. |
| 2024 | The series identifier uses a different disclosed source label, and the quantity unit uses `SHARES VOTED` instead of `SHARES`. |

The ten schema-valid answers with undisclosed quantities require a null quantity value and `ABSENT_IN_CONTEXT` availability.
Sol instead marks the enclosing quantity as present, although its amount and unit remain absent.
The 2024 identifier value and numeric quantity are correct.
Its alternative identifier label and unit heading occur in the source, but they fail the frozen field comparisons.
These failures are not unsupported quotations, and review does not replace the original answers or references.

Astra accepted five subject paraphrases for 2014, 2015, 2017, 2019, and 2023 under the existing equivalence rule.
The 2014 paraphrase retains the board powers, additional US$75 million cap, and identified repurchase program.
The omitted historical approval date identifies that same program but changes no authorization condition.
Its original `raw_description` also retains the prior approval date.
The 2019 paraphrase names the same allocation-of-income subject.

The 2022 and 2024 omissions fail equivalence because they remove qualifications from the current voted subject.
No `OTHER` equivalences applied.

The report counts ten origin diagnostics, four derivation diagnostics, and no literal or semantic quotation failures among schema-valid answers.
The origin diagnostics reflect the added quantity structure.
All four derivation diagnostics reflect quotation differences in `reporting_scope.scope_type`, not incorrect scope types.
Astra also inspected the malformed answer's quotations for diagnosis only.
That answer receives no imported quotation decision or partial field credit.

## Future model comparisons

Use the [evaluation procedure](training.md#selected-model-evaluation) for each new run.
Keep the frozen source content, case order, system instructions, reference labels, and scoring rules unchanged.
Record each model revision, tokenizer, weight format, runtime, and generation configuration.
Disclose model-specific input formatting and any differences in limits or sampling.
Do not silently truncate source context to fit another model.

Retain raw answers and terminal failure details before parsing.
Review each new model's answers against their exact source context and reference labels.
Do not reuse decisions from another model's answers.
Append a results row only after scoring and required review finish.
Include missing answers, pending reviews, invalid references, failure counts, and relevant comparison conditions with each new entry.
If review remains pending, record its status without presenting partial scores as final results.

Keep the twelve-record and 168-field denominators for comparable runs.
Count malformed answers, terminal failures, and extra records as failures.
Do not remove difficult records or substitute a successful retry without disclosure.
If timing comparisons use different hardware, runtimes, cache policies, or limits, disclose those differences.
Token totals also depend on the model's tokenizer.
When you add a comparison, preserve earlier results.

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
This log contains reviewed results for Qwen3.5-4B, ProxyType-4B, ProxyType-9B, GPT-6 Luna, and GPT-6 Sol.
The source task and scoring rules remain frozen, while the agent execution conditions differ from the local runs.
Data and model publication require separate approval under the [data notice](../DATA_NOTICE.md#source-redistribution) and [model card](../MODEL_CARD.md#model-publication).
