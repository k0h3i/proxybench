# Historical pilot scoring examples

The scorer uses the fourteen-field `training-label-v1` contract.
Its fixed version is `historical-pilot-v1`.
It preserves each raw answer before parsing or normalization, which standardizes permitted formatting differences.
The primary result requires every directly disclosed fact in one target to be correct.
Reference rules choose comparison fields, so a predicted `DERIVED` tag cannot hide an extracted fact.

Each example below changes only the named field in an otherwise valid answer.
Formatting whitespace can differ across fields.
Only issuer and fund names ignore case.
Lists ignore order but retain duplicates.
Source review is required for subject equivalence and quotation meaning.

| Field | Reference and prediction example | Primary outcome |
|---|---|---|
| `reporting_scope` | Fund `Alpha Fund` becomes `ALPHA  FUND` | Pass. A different fund or omitted member fails. Derived scope type receives a separate diagnostic. |
| `series_identifiers` | `Series ID: S0000123` becomes `S123` | Fail. Preserve the label, string, and leading zeros. |
| `issuer_name` | `Example Corp.` becomes `EXAMPLE  CORP.` | Pass. An inferred legal alias fails. |
| `security_identifiers` | One disclosed identifier appears twice | Fail. Duplicate identifiers change membership. |
| `ticker` | `ABC.A` becomes `ABC` | Fail. Preserve punctuation and case. |
| `meeting_date` | `2007-05-25` becomes `2007-05-26` | Fail. Non-ISO or invalid dates also fail format validation. |
| `meeting_type` | `Annual Meeting` becomes `Annual meeting` | Fail. Preserve source wording and case. |
| `proposal_number` | `001.` becomes `1` | Fail. Preserve the string, punctuation, and leading zeros. |
| `raw_description` | `Elect Director Jane Doe` becomes `Jane Doe` | Fail. This field does not permit paraphrases. |
| `separate_subject` | `Elect Director Jane Doe` becomes `Jane Doe` | Pass only after source review confirms the same single nominee. `Elect directors` omits the named subject and fails. |
| `proposal_source` | `ISSUER` becomes `Management` | Fail. Do not replace disclosed wording with a guessed category. |
| `participation` | Directly disclosed `DID_NOT_VOTE` becomes `VOTED` | Fail. A wrong reference-derived participation value receives a separate diagnostic. |
| `vote_components` | A `FOR` component becomes `AGAINST`, or appears twice | Fail. Component order can differ. Multiplicity and explicit nonvoting structure cannot differ. |
| `management_recommendation` | Missing information becomes explicit `NONE` | Fail. A recommendation also remains separate from agreement with management. |

A legal but incorrect origin value causes an origin diagnostic.
It does not change an otherwise correct primary source value.
An unknown origin enum fails format validation.
A missing field, extra record, malformed object, duplicate JSON key, or output-limit ending fails whole-record accuracy.
The exact-answer count compares parsed objects without normalization, including origins and literal quotations.

For `OTHER`, the original wording carries the category meaning.
Changing `OTHER` wording from `No action` to `Declined` fails automatic primary comparison.
A source excerpt can be literal but unrelated to its associated claim.
The automatic quotation diagnostic finds excerpts absent from the supplied source.
The source reviewer separately records quotation errors and decides whether a subject paraphrase preserves the complete voted subject.

Export admission compares each engine answer with its Python answer.
It requires equal normalized values, availability, origins, derivations, and component multiplicity.
Only subject equivalence and supported quotation differences can enter source review.
Two malformed answers pass this gate only with identical decoded output and terminal status.
Output-limit endings require identical token IDs and terminal status, and still fail extraction scoring.
Different validity states or unexplained differences stop the export comparison.

The [behavior tests](../tests/test_historical_pilot.py) exercise these distinctions.
The final report keeps format validity, exact matches, primary source values, origins, derivations, and quotations separate.
It reports all 24 targets and the three development families.
One target changes the aggregate rate by about 4.17 percentage points.
Related targets share filings and templates, so the report does not treat them as independent statistical observations.
