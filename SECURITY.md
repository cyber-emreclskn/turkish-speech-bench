# Security Policy

## Supported versions

Only the latest release of PortPilot receives security fixes.

## Reporting a vulnerability

Please **do not** open a public issue. Instead, use GitHub's
[private vulnerability reporting](https://github.com/cyber-emreclskn/turkish-speech-bench/security/advisories/new)
for this repository.

You can expect an initial response within 7 days. Once a fix is available we
will publish a GitHub Security Advisory and credit you unless you prefer
otherwise.

## Scope notes

turkish-speech-bench downloads models and datasets from the Hugging Face Hub and
the Vosk model server, and runs third-party model code (e.g. `transformers`).
Only benchmark models you trust.
