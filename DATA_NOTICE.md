# Data notice

Original filings come from [SEC EDGAR](https://www.sec.gov/edgar/search/).
The [acquisition guide](docs/dataset.md#acquisition-and-source-storage) explains how to download and import filings for this repository.

Third-party rights remain separate from the project's code license.
Preserve notices in the original filings.

The SEC does not endorse this project.

## Source redistribution

A manifest records file identities and metadata.
The private source manifest is not the public upload list.
Create an allowlist, an explicit list of approved files.
For each file, record rights and privacy status, reason, reviewer, date, and supporting source.
Use `approved`, `held`, or `excluded`.
Publish only approved entries.

Do not apply a blanket license to third-party filings or accepted labels.
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

## Source archives

The public manifest records kind, original name, friendly name, SEC URL, retrieval date, size, and SHA-256 hash.
SHA-256 is a digest used to identify exact file bytes.
Use `filing` or `index` as the kind.
Keep accession identities and amendments separate.
Identify indexes by year and quarter, with a null accession.
Record unknown retrieval dates as unknown.

Use a versioned GitHub release asset for the source archive.
Keep its small public manifest and notice in the public repository.
Inspect the exact archive locally before approval.
Make sure that its names and hashes match the allowlist.
Exclude unsafe paths, links, owner metadata, hidden files, private labels, and local configuration.
List omissions and their effect on reproducibility in this notice.

Before requesting publication approval, follow the [security policy](SECURITY.md#reporting-and-release-review).
Present the archive, manifest, notice, code license, and selected history to the user.
Upload only after separate authorization.
After upload, download the asset and compare its digest with the approved archive.
Publish a correction route without promising removal from third-party copies.
