# Benchmark implementation

The `benchmark-v1` helpers enforce record rules and score preserved responses.
A reference is a reviewed answer used for scoring.
The [development runner](extractor-runner.md) captures extraction outputs before calling these helpers.
Comparative results require a separate execution decision.

The package requires Python 3.11 or later and adds no runtime dependencies.
Synthetic fixtures exercise the accepted failure rules.
Local labels, source bundles, decisions, and generated reports remain outside Git.

## Record validation

`proxybench.schemas.records.normalize_record(record)` returns a new, normalized record or raises `ValidationError`.
The input dictionary remains unchanged.
`strict_json(raw)` rejects duplicate keys, incomplete JSON, nonstandard numbers, and surrounding prose.
Preserve the exact response bytes before calling these helpers.
Malformed decimal metadata uses `{ "json_type": "number", "value": "0.5" }` in diagnostic reports.
This representation keeps reports compatible with the standard JSON writer and preserves the distinction between numbers and strings.
It does not replace the exact raw response bytes.

Every record requires `record_id`, `source_anchor`, `fields`, and `enrichments`.
`FIELD_TYPES` defines the 14 fields and their nested types.
Each field requires `value`, `raw_text`, `availability`, `origin`, `evidence`, `reason`, and `rule_id`.
Unavailable values use nulls instead of empty strings.
Uncertain members can remain unresolved inside a readable container.

Core fields permit extracted facts and explicit uncertainty.
D2 permits derived `VOTED` participation and the empty component field for explicit `DID_NOT_VOTE`.
Other derived or inferred values belong in `enrichments`.
Zero quantities cannot create cast-vote components.
Withhold, Abstain, and Against remain distinct directions.

Normalization preserves identifier string types, leading zeros, punctuation, and internal whitespace.
Names receive Unicode casefold after whitespace normalization.
Casefold makes text comparable without letter-case differences.
Descriptions retain case and wording.
Date parsing accepts only the finite calendar grammar in `normalization.values`.
Guide `benchmark-v1-date-order-v2` treats slash dates as month/day/year and rejects invalid dates without swapping components.
Guide `benchmark-v1` retains the earlier ambiguity rule for reproduction.
`normalize_record(record, guide_version=...)` selects the date policy explicitly.
The default remains `benchmark-v1` for compatibility.
Admitted references carry their guide version into the scorer.
Under the active guide, `6/10/2014` matches `2014-06-10`, while `15/05/2007` fails.

## Reference admission and source evidence

`annotation.bindings.validate_input_binding` compares the hashes of the manifest, review view, and exact model input.
A hash identifies a file by its contents.
Scored references require a recorded `SAME_EVIDENCE` decision.
That decision concerns source facts and associations, without requiring identical view and model-input bytes.

`evaluation.references.admit_reference` rejects provisional labels, pending conversions, stale artifacts, and incorrect scoring masks.
A scoring mask lists the reference fields included in comparison.
The function derives that list from the reference instead of trusting predictions.
Admission examines citations in core fields and all enrichment fields, including nested values.
Core scoring diagnostics still exclude enrichment citations.
The caller supplies the trusted workspace, input identity, guide version, model-input hash, and source context.

`evaluation.evidence.SourceContext` holds original bytes, permitted source ranges, declared encodings, and prepared-text mappings.
Its `views` dictionary uses `(view_id, block_id)` keys.
Each entry contains `span` and `text`.
A prepared quotation must appear in that exact mapped text and cite that mapping's byte range.

Mechanical citation validity does not establish semantic support.
Semantic support means that the cited source supports the asserted fact.
The diagnostic report starts each scalar assertion as `NOT_ASSESSED`.
`apply_semantic_reviews` attaches recorded source-review decisions to a new report version.
`summarize_evidence` reports assessed fractions and responses with unknown assertion counts.

## Fragment scoring

`evaluation.scoring.score_fragment(raw, reference, context=...)` scores one scheduled target.
Use an admitted `Reference` for reviewed data.
`synthetic_reference` explicitly marks software fixtures as synthetic.
It cannot establish acceptance of historical labels.

Exactly one complete record must match every included field.
Missing, malformed, duplicate, extra, failed, and abstained outputs fail the primary score.
Array comparison ignores order and preserves duplicates.
The reference controls the exclusion of derived participation.
Anchor and evidence findings remain separate from field accuracy.

Answered-slot coverage counts present answers to recoverable scalar fields.
It does not require correct values.
Array assignment maximizes answered slots without assigning extra credit to duplicate output.
`summarize_fragments` retains failed responses in whole-record and field denominators.

## Complete-filing scoring

`FilingReference` holds admitted records, reviewed anchor alternatives, a source context, and a filing coverage review.
A source anchor identifies a subject and its reporting scope by original byte ranges.
The coverage review lists every document, its disposition, reviewed ranges, hashes, and unresolved coverage.
Partial references cannot supply complete-filing recall.
Empty reference filings need a recorded source-review basis.

`score_filing(raw, filing, processing_coverage=...)` matches records through frozen source zones.
A zone is a reviewed range for one logical subject.
Ambiguous anchors receive no match.
The earliest uniquely anchored prediction claims a reference, regardless of its field values.
Other copies remain false positives.

The caller supplies trusted processing coverage separately from model output.
Its `documents` dictionary maps each document ID to completed `[start_byte, end_byte]` ranges.
Its `unrecovered_truncation` value must be false for complete execution.
Missing document coverage prevents exact filing success.
Valid partial records in a failed response can still receive recovery credit.

`summarize_filings` pools recall across all eligible references.
It reports precision on countable responses with the response-coverage fraction.
Unknown prediction counts and zero denominators produce undefined rates.
These measures do not establish complete historical coverage from fragment fixtures.

## Review and retrieval changes

The development review page now requires exact manifest, view, and model-input bindings.
Suggestions and resumed drafts must match those bindings.
Value edits preserve separate source wording and append correction history.
The page refuses test packets because test review requires its separate initial-answer procedure.
Existing pages and legacy labels retain their bytes.

Source and draft text no longer act as template substitution instructions.
The page preserves literal template markers and escapes closing-script text.
The [source tools guide](source-calibration-tools.md) explains the new review arguments and cache recovery procedure.

`annotation.audit.audit_inventory` compares legacy labels with pinned decision and export artifacts.
It tests acceptance states, paths, source ranges, hashes, and export consistency.
Its checks remain active under optimized Python.
It establishes legacy integrity only, without accepting typed conversions.

## Remaining work

The six local accepted conversions passed source and representation review.
Each run still requires admission against their exact bindings.
Future test labeling needs a separate time allocation and the frozen source-only initial-answer procedure.
Future selection needs a versioned candidate ledger and reviewed split groups.
The current development assignments do not create an untouched test set.

The bounded HTML parser, `RecordedPredictionAdapter`, and development runner now have implementations and synthetic tests.
Complete-filing extraction and model execution remain unimplemented.
System selection, prompts, observable configuration, and numeric resource limits remain unfrozen.
A controlled run requires those artifacts and explicit execution review.
Training retains its separate gate.
