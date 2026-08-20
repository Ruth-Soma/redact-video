# Security Policy

Video Redactor processes highly sensitive visual data. Please report security
problems privately and never attach real payment-card media or recognized card
values to a report.

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x public beta | Yes |
| Unreleased development snapshots | Best effort |

Until a stable release exists, fixes are applied to the latest `0.1.x` release
and the default development branch.

## Reporting a vulnerability

Use the repository's **Security → Report a vulnerability** flow to open a
private GitHub security advisory. If private reporting is unavailable, contact
the repository owner through the contact method on their GitHub profile. Do not
open a public issue for a suspected vulnerability.

Include only sanitized information:

- affected version or commit;
- operating system, Python version, and installation method;
- the component and expected versus observed behavior;
- minimal reproduction steps using generated or value-free fixtures;
- impact and any known mitigation.

Do not include source videos, frame captures, OCR transcripts, cardholder data,
credentials, filesystem paths, or cloud identifiers. The maintainer will
acknowledge a report as soon as practical, validate it privately, coordinate a
fix and disclosure, and credit reporters who want attribution.

## Deployment boundary

The bundled API is a local service boundary, not an internet-ready multi-tenant
service. It binds to loopback by default and does not provide authentication or
TLS. Public deployments must add a trusted reverse proxy, authentication,
transport encryption, request controls, isolated storage, monitoring, and a
retention policy.

The detailed application threat model and data-handling guarantees are in
[docs/SECURITY.md](docs/SECURITY.md).
