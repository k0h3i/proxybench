# Modern N-PX mapping

`npx-mapping-v2` supports a conservative subset of registered-fund voting reports.
The mapper preserves original byte locations and rejects interpretations that lack a supported rule.
Mechanical mapping success produces provisional examples, never automatic training admission.
The [preparation guide](training-preparation.md) describes execution and review.

## Source specification

The source is the [official N-PX XML specification, version 3.1](https://www.sec.gov/files/edgar/filer-information/specifications/edgar-form-n-px-xml-technical-specification-31.zip).
The specification page dates that version February 3, 2025.
Each private run preserves the downloaded archive, retrieval date, SHA-256 hash, schemas, and inspected stylesheet.
The archive version and any source-file schema marker remain separate observations.

The vote namespace is `http://www.sec.gov/edgar/document/npxproxy/informationtable`.
The primary namespace is `http://www.sec.gov/edgar/npx`.
In the table below, `R` means `/proxyVoteTable/proxyTable` and `V` means `R/vote/voteRecord`.
`S` means `/edgarSubmission/formData/seriesPage/seriesDetails/seriesReports` in the related primary attachment.

The mapper accepts `registrantType=RMIC`, `reportType=FUND VOTING REPORT`, and submission types `N-PX` or `N-PX/A`.
It preserves amendments without consolidation.
Manager-only reports, notices, unknown namespaces, and unresolved series joins fail automatic mapping.
Missing or unsupported filing structures produce a filing rejection, not a zero-vote result.

## Field rules

Every row below applies to one selected voting subject and its disclosed reporting scope.
Source evidence starts with the named XML element and retains its exact byte range.
Rendered evidence points to the exact visible cell or line, with a link back to that XML element.
An unsupported interpretation causes rejection rather than an invented absence label.

| Benchmark field | Source path and multiplicity | Mapping, absence, and unsupported cases |
|---|---|---|
| `reporting_scope` | One `R/voteSeries`, joined to one `S/idOfSeries` | Direct association with `S/nameOfSeries`. Produces an individual fund. Missing, duplicate, conflicting, or group joins are rejected. |
| `series_identifiers` | One `R/voteSeries` | Direct string disclosure. The renderer prints the natural heading `Series ID`. No conversion to numbers. |
| `issuer_name` | One `R/issuerName` | Direct full wording. Empty or structured scalar values are rejected. |
| `security_identifiers` | Zero or one each of `R/cusip`, `R/isin`, and `R/figi` | Direct labeled strings. Preserve all supplied identifiers and leading zeros. Missing identifiers remain absent. `N/A` needs a separate rule and is rejected. |
| `ticker` | No supported vote-table path | Absent in current renderings. Do not copy the reporting fund's class ticker as the voted security's ticker. |
| `meeting_date` | One `R/meetingDate` | Direct disclosure with deterministic calendar formatting under `benchmark-v1-date-order-v2`. Preserve original date text. Invalid dates are rejected. |
| `meeting_type` | No supported vote-table path | Absent in current renderings. Do not infer a meeting type from description wording. |
| `proposal_number` | No supported vote-table path | Absent in current renderings. XML entry indices remain administrative locations and never become proposal numbers. |
| `raw_description` | One `R/voteDescription` | Direct full wording. Preserve punctuation, escaped characters, and source instructions as text. Supplemental voting text triggers review instead of silent omission. |
| `separate_subject` | No general deterministic name path | Absent for the supported ordinary proposals. Director-election categories are rejected until a reviewed subject rule exists. Individual auditor or similar subject cases also need audit. |
| `proposal_source` | Zero or one `R/voteSource` | Direct `ISSUER` or `SECURITY HOLDER` wording. Missing elements produce packet absence. Categories never supply a proponent. |
| `participation` | Positive cast components in `V` | Permitted D2 derivation of `VOTED`. Zero totals, missing vote containers, and source instructions do not establish explicit nonvoting. |
| `vote_components` | One to 999 `V` elements within the selected entry | Preserve each direction, positive quantity, and disclosed alignment. Keep split votes together. Reject zero components and conflicting totals. |
| `management_recommendation` | No supported recommendation path | Absent in current renderings. `V/managementRecommendation` supplies alignment, not a recommended vote. Never reverse alignment to infer a recommendation. |

The current renderer can deliberately omit supported cells for synthetic visibility tests.
When it omits a cell, its target becomes absent within that packet.
The broader XML finding stays in the separate source audit.
Partial omission of a split direction is rejected because it changes disclosed component membership.

## Nested values and evidence

Nested wrappers retain their own values, availability, origin, and evidence.
A present container establishes structure without inventing unsupported children.
The mapper currently rejects difficult source interpretations before rendering, so unresolved real-source branches remain coverage gaps.

| Nested field | Source and rule | Evidence and absence behavior |
|---|---|---|
| `reporting_scope.name` | `S/nameOfSeries` after the exact series join | Cite the original series-name element and its rendered fund heading. |
| `reporting_scope.scope_type` | The uniquely joined registered-fund series | The visible `Reporting fund` heading supports `INDIVIDUAL_FUND`. A group join is unsupported. |
| `reporting_scope.members` | `reporting-members-v1` | Use `NOT_APPLICABLE`, null value and origin, scope evidence, and an individual-fund explanation. Never invent a one-member group. |
| Identifier `source_label` | The supported XML element's defined role | Print and cite `Series ID`, `CUSIP`, `ISIN`, or `FIGI` as a natural heading. Never infer an identifier type from length. |
| Identifier `value` | The same XML element | Preserve the string exactly, apart from permitted outer whitespace. |
| Component `direction` | `V/howVoted` | Direct direction. Known directions retain their enum. Other readable wording uses `OTHER` with original text. |
| Component `quantity.amount` | `V/sharesVoted` | Positive nonnegative-decimal syntax, with at most 24 digits. Zero is rejected as a cast component. |
| Component `quantity.unit` | The schema-defined shares field | Print and cite `Shares voted`. Do not invent another unit or convert quantities. |
| Component `disclosed_management_alignment` | `V/managementRecommendation` | Under `For or against management`, map `FOR` to `WITH_MANAGEMENT` and `AGAINST` to `AGAINST_MANAGEMENT`. |

The official `NPX-INFO-TABLE_X01.xsl` stylesheet labels `managementRecommendation` as `FOR OR AGAINST MANAGEMENT`.
The element name alone is misleading for the benchmark's recommendation field.
The unreviewed `NONE` alignment branch retains `OTHER` and raw `NONE`, without asserting a management recommendation.
This branch has synthetic status until an original disclosure and its meaning receive review.

`R/sharesVoted` and `R/sharesOnLoan` remain separate source-audit values.
The mapper compares component quantities with the disclosed total and rejects a disagreement.
That comparison is a consistency test, not an extracted sum field.
Loaned shares never become cast-vote components.

## Rejection and coverage limits

Keep every selected rejection with its source pointer and reason.
Reconcile physical entries discovered, parsed, selected, unselected, provisionally mapped, rejected, and admitted.
Parsing an attachment does not mean that every entry received semantic mapping or review.
The selection manifest caps semantic work at 24 entries across four filings.

The current rules reject director subject extraction, unjoined fund groups, supplemental voting text, ambiguous identifiers, and nonvoting interpretations.
They also reject unknown row structures, duplicate singleton elements, malformed quantities, and unsupported nested structures.
These rejections preserve uncertainty but limit the usable modern sample.
Do not describe rejected branches as supported coverage.

Some ordinary descriptions can still require a separate-subject decision or disclose overlapping share classes.
Human review must resolve those cases before rule-level admission.
The first bounded sample and both renderers remain development material.
Matching citation bytes establishes a valid location, not semantic support.
