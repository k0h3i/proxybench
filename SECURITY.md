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
Repeat the scanner and private-identifier inspection on the exact approved release under [docs/release.md](docs/release.md).

Review Python dependencies before a code release and native libraries before a supported model-runtime release.
Test required security updates in a separate environment.
Record unresolved findings and unavailable inspection surfaces before requesting release approval.
Platform malware scanning adds a check but does not certify all files as safe.
