# Security policy

Linux and WSL are the supported platforms.
The CPU package uses the Python standard library.
Optional training and inference dependencies have separate version records under `configs/`.
GPU validation remains a separate user-launched gate.

## Supported boundaries

Treat filings, model output, and imported review files as untrusted data.
Keep source display escaped and sandboxed without scripts, forms, or external requests.
Serve only a dedicated display directory over loopback.
Never serve the repository or home directory.

Keep inference servers bound to `127.0.0.1`.
Keep download size, parser, process lifetime, and memory limits enabled.
Reject redirected requests outside approved SEC HTTPS hosts before forwarding contact headers.
Reject embedded URL credentials, unsafe ports, path traversal, and symlink escape.
Do not bypass SEC access blocks.

Load published adapter tensors through Safetensors.
Future resumable checkpoints include optimizer objects and require trusted local inputs.
A hash next to an untrusted checkpoint does not authenticate its producer.
Do not distribute optimizer checkpoints as inference artifacts.

## Reporting and release review

Use the hosting platform's private vulnerability-reporting route for sensitive findings when it is enabled.
Before public release, enable that route and test access.
If it is unavailable, request a private channel without disclosing the sensitive details in a public issue.
Do not add a personal email address to this document.

Gitleaks 8.30.1 found no findings in the cleaned tracked-file candidate and all reachable local Git history on 2026-09-25.
This scan does not prove the absence of secrets or inspect remote-only history, hosted assets, or every secret format.

Before publication, repeat a maintained secret scanner on the exact files and selected Git history.
Inspect archives, Git LFS (storage for large files) objects, logs, screenshots, hosted assets, and model metadata where applicable.
Record unavailable inspection surfaces and skipped files.
If a live secret appears, revoke or rotate it before addressing copied files.
Do not claim that a pattern scan proves the absence of secrets.

Use a GitHub noreply address for public author and committer metadata.
Inspect every selected commit and annotated tag, including trailers and signing metadata.
Compare the proposed tree and archives against a private list of known owner identifiers.
Do not write that list or matched values into shared reports.
Changing future commit identity does not remove old metadata.

Keep training labels, review exports, contact configuration, environments, and experiment output private.
Exclude personal email, local usernames, home paths, IP addresses, and machine identifiers.
The user's name can appear in intended attribution.
Hosting account handles remain visible.

For code release, test retrieval, parsing, source display, and all supported public input boundaries.
Review Python dependencies before a code release and native libraries before a supported model-runtime release.
Test required security updates in a separate environment.
Record unresolved findings and unavailable inspection surfaces before requesting release approval.
Platform malware scanning adds a check but does not certify all files as safe.
Model runtime findings do not block a source-only release that contains no affected executable.

Use multifactor authentication and repository-scoped publishing credentials.
Keep tokens outside files, transcripts, notebooks, and archives.
Do not add publishing automation without a concrete need.
