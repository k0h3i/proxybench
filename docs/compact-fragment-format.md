# Compact fragment format

The compact format reduces training tokens while retaining source text and record evidence.
It uses short source coordinates, shared citations, and explicit defaults for field wrappers.
A field wrapper holds a value with its availability and evidence.
The [model contract](compact-fragment-contract.md) defines the new JSON representation.
The existing benchmark contract remains unchanged.

## Input and output

The input contains the marked target and ordered source blocks.
Each block retains its exact original text and prepared text.
The `gap_before` flag records whether omitted bytes separate it from the preceding block.
The format removes document names, hashes, and absolute byte positions from the model prompt.
A separately stored bundle retains that metadata for deterministic reconstruction.

The output uses the existing fourteen field names and nested structures.
It retains every availability state, origin, raw value, reason, rule, and citation.
The following synthetic excerpt illustrates a present field and an absent field:

```json
{
  "issuer_name": {"v": "Example Corp", "raw": "Example Corp", "e": [0]},
  "ticker": null
}
```

The citation at index zero supplies the source location and quotation once.
Other fields can refer to that same citation without repeating it.
The decoder restores field defaults and metadata from the bound source bundle.
It does not consult accepted answers or infer field values.
It assigns an administrative record ID independently of the source facts.

Null means packet absence only when no other field information exists.
Ambiguous, conflicting, unreadable, and inapplicable fields retain explicit states and their evidence.
An explicit recommendation of `NONE` remains distinct from null.
The special nonvoting component list remains empty with its derivation and supporting evidence.
The initial version rejects enrichment records, which contain additional calculations or semantic inferences.

## Reversibility and limits

Reversibility means that the decoder reconstructs the original record information.
The preparation audit compares every restored field and citation with its original value.
It also compares the complete record after restoring its original administrative ID for that audit only.
This comparison establishes serialization fidelity, not source accuracy or model learnability.

The decoder rejects unknown keys, duplicate JSON keys, missing fields, invalid citation indices, and source ranges outside the bundle.
It also rejects invalid quotation matches and extra records in a complete fragment response.
It retains raw predictions before decoding when integrated with an execution runner.
No execution runner uses the compact decoder yet.
Transport truncation still needs the runner’s completion flag because valid JSON alone cannot establish complete capture.

The shorter contract changes the model-facing instructions.
The source content and reconstructed record requirements stay the same for admitted codec inputs.
Review the format before freezing a new comparison or training configuration.
Keep old prompts, references, and results under their original versions.
Do not score compact output directly as malformed benchmark JSON.

## Preparation interface

The package module `proxybench.training.compact` implements source packaging and record conversion.
The module `proxybench.training.compact_prepare` writes development previews and token measurements.
It requires the optional tokenizer environment and never loads model weights.
Every output directory must be new.

```bash
PYTHONPATH=src GPU_PYTHON -m proxybench.training.compact_prepare \
  --schedule HISTORICAL_SCHEDULE.json \
  --modern MODERN_PREVIEWS_DIRECTORY \
  --tokenizer PINNED_TOKENIZER_DIRECTORY \
  --historical-lengths HISTORICAL_LENGTHS.json \
  --contract docs/compact-fragment-contract.md \
  --old-contract docs/model-extraction-contract.md \
  --output NEW_OUTPUT_DIRECTORY
```

The prior historical lengths must use the same tokenizer as the new measurements.
Use complete prompts and responses, including message formatting and the termination token.
The previews remain development data and receive no automatic training admission.
Retokenize with a new model’s tokenizer before changing its context cap.

The initial measurements justify considering a 12,288-token cap for the inspected examples.
They do not establish GPU memory fit or a safe cap for future source samples.
Keep a failure path for longer fragments and never truncate evidence silently.
Further reductions to HTML require a separate source-preservation review.
