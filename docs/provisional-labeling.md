# Provisional labeling

The first wave prepares up to eight source targets for human review.
A target is one proposed voting subject within a disclosed reporting scope.
Generated responses remain development drafts until the user reviews their sources.
Training admission requires a separate decision for each target.

## XML source packets

`proxybench.annotation.xml_packets.build` creates a derived packet from two preserved XML attachments.
The packet includes the complete primary attachment, one selected voting entry, and the voting attachment's root context.
The primary attachment retains series identities and explanatory information.
Unknown voting-table context causes rejection instead of silent omission.

Each block copies exact original bytes.
Generated separators remain outside source blocks and cannot support citations.
The prepared view contains literal XML, including escaped entities.
For example, `&amp;` remains `&amp;` in a literal quotation.
The packet is not a complete original XML document.

`project` maps a packet span back to one preserved attachment.
A span identifies a document and its start and end byte positions.
The function rejects changed source bytes, invalid bounds, unknown identities, and citations across separators.
The packet hash identifies the derived packet, while each attachment retains its original hash.

Select context from source structure before generating labels.
Keep historical bundles unchanged.
Retain unresolved duplicate relationships and review the logical record count before admission.

## Isolated generation

`proxybench.execution.labeling.freeze` accepts source bundles without reference labels.
It freezes the contract, prompts, model cache, executable hashes, and Python source hashes.
The separate comparison controller retains its existing medium-effort behavior.
The new controller requests `gpt-5.6-sol` at low effort through subscription authentication.
The frozen task explicitly prohibits tools, including tools for citation arithmetic.
Disabled tools can still receive attempted calls from the model.
The controller rejects those attempts even when the tool host denies execution.

Execution starts with a filesystem isolation test and one synthetic configuration probe.
The probe must record the requested model, effort, exact prompt, and native final response.
The controller compares subsequent application instructions with this observed context.
A missing observation, tool call, changed binding, or failed cleanup stops generation.

Each attempt uses a fresh filesystem namespace and private state directory.
A namespace limits the files that a process can see.
The labeler cannot see the repository, reference labels, notes, or previous sessions.
Authentication stays in temporary private state and never enters exported artifacts.

Run a frozen package with the existing subscription authentication:

```bash
PYTHONPATH=src python3 -m proxybench.execution.labeling artifacts/runs/RUN/generation \
  --auth /path/to/private/auth.json
```

The controller permits one attempt per target, with no automatic retries.
Each attempt has a 300-second limit.
The 2,700-second total includes the synthetic probe and capture work.
Before submission, the controller reserves 20 seconds for preparation, 10 for finalization, and 15 for cleanup.
Cancellation tests must establish termination within that cleanup allowance before freezing a package.
These limits are execution caps, not expected durations.

## Review results

Raw native response bytes, event streams, and original session records remain available before parsing.
Late responses remain separate and cannot replace an incomplete primary response.
Mechanical decoding does not establish semantic support or acceptance.
The report retains unstarted targets and invalid outputs in the scheduled denominator.

`proxybench.annotation.label_review.render` creates a source-first review page and an admission ledger.
The ledger separates generation, mechanical validity, source review, unresolved findings, rejection, and training admission.
The page records active review time and exports notes without changing existing labels.
The user must read sources before revealing suggestions.

Review record count, scope, values, uncertainty, and evidence meaning.
Reconcile missing discussion time before assigning review work within the existing allowance.
Human corrections require a new label revision and must preserve the original draft.
New training workloads and dataset expansion retain separate review gates.
