# Release procedure

Cleanup does not authorize public release.
The intended publication is noncommercial, from Canada, through GitHub, with a later adapter release on Hugging Face.
Keep the existing private history unchanged.
A separate public repository with a reviewed initial commit is the proposed default.

## Public content and identity

Prepare the exact file list before asking for publication approval.
Keep training labels, review exports, contact configuration, environments, and experiment output private.
The user's name can appear in intended attribution.
Exclude personal email, local usernames, home paths, IP addresses, and machine identifiers.
Hosting account handles remain visible.

Use a GitHub noreply address for public author and committer metadata.
Inspect every selected commit and annotated tag, including trailers and signing metadata.
Compare the proposed tree and archives against a private list of known owner identifiers.
Do not write that list or matched values into shared reports.
Changing future commit identity does not remove old metadata.

Run a maintained secret scanner over the exact public tree and selected history.
Inspect archives, LFS objects, logs, screenshots, hosted assets, and model metadata where applicable.
Record unavailable surfaces and skipped files.
If a live secret appears, revoke or rotate it before addressing copied files.
Do not claim that a pattern scan proves the absence of secrets.

Choose the code license before publication.
Review the pinned base model license and required attribution separately.
Do not apply a blanket license to third-party filings or accepted labels.
Screen ProxyType-4B on GitHub, Hugging Face, package registries, and relevant trademark sources before publication.
A search does not establish legal availability.

## Raw source review

The private source manifest is not the public upload list.
Create an allowlist, an explicit list of approved files.
For each file, record rights and privacy status, reason, reviewer, date, and supporting source.
Use `approved`, `held`, or `excluded`.
Publish only approved entries.
A shared SEC rationale can cover multiple files without conflicting notices.

The SEC permits use and distribution of its public EDGAR filings.
Review specific third-party notices rather than treating this policy as a blanket ownership claim.
Do not imply SEC endorsement or use its seal as project branding.
See the [SEC dissemination policy](https://www.sec.gov/about/privacy-information#dissemination).

Two retained HTML sources contain generator notices from Summit Financial Printing and Thomson Reuters Accelus.
Their original identities are `0001398344-15-005568/fp0015663_npx.htm` and `0001398344-14-004541/fp0011455_npx.htm`.
Hold these files until their notice scope and redistribution rationale receive review.
Neither source belongs to the 36 final-dataset source files.
Preserve the notices and original bytes.

CUSIP Global Services describes permitted self-collected noncommercial uses and separate distribution rights.
Record the rationale for identifiers in unchanged filings.
Do not add vendor identifier tables or infer a universal paid-license requirement.
See the [CGS licensing statement](https://www.cusip.com/services/license-fees.html).
Canadian research fair dealing does not automatically settle redistribution of a complete archive.
See [Copyright Act section 29](https://laws-lois.justice.gc.ca/eng/acts/C-42/section-29.html).
For a concrete unresolved right, obtain Canadian legal advice or omit the affected file.

Review contacts, signatures, personal addresses, accounts, and embedded attachments in context.
Public availability alone does not settle privacy requirements.
Do not classify ordinary business contacts or issuer identifiers as secrets without evidence.
If redaction is necessary, withhold the original or name and document a separate derivative.
Never reuse the original hash for modified bytes.

## Archive and approval

The public manifest records kind, original name, friendly name, SEC URL, retrieval date, size, and SHA-256 hash.
SHA-256 is a digest used to identify exact file bytes.
Use `filing` or `index` as the kind.
Keep accession identities and amendments separate.
Identify indexes by year and quarter, with a null accession.
Record unknown retrieval dates as unknown.

Use a versioned GitHub release asset for the raw archive.
Keep its small public manifest and notice in the public repository.
Inspect the exact archive locally before approval.
Make sure that its names and hashes match the allowlist.
Exclude unsafe paths, links, owner metadata, hidden files, private labels, and local configuration.
List omissions and their effect on reproducibility in [DATA_NOTICE.md](../DATA_NOTICE.md).

Present the archive, manifest, notice, code license, and selected history to the user.
Upload only after separate authorization.
After upload, download the asset and compare its digest with the approved archive.
Publish a correction route without promising removal from third-party copies.

## Adapter and code gates

For code release, test retrieval, parsing, source display, and all supported public input boundaries.
Review applicable dependency findings before release.
For adapter release, test loading with the pinned base revision.
Stage only Safetensors weights, portable configuration, necessary tokenizer files, notices, and the model card.
Exclude optimizer state, run output, labels, tokens, hidden folders, and `.env` files.
Compare exact training messages against the private owner-identifier list.
If private identifiers entered training, hold the adapter for assessment.

Use multifactor authentication and repository-scoped publishing credentials.
Keep tokens outside files, transcripts, notebooks, and archives.
Do not add publishing automation without a concrete need.
Model-runtime findings do not block a raw-only release that contains no affected executable.
