---
description: How to report a security vulnerability in Voicera.
---

# Security policy

## Reporting a vulnerability

**Do not open a public issue.** Report privately, so a fix can ship before the problem is widely known.

Email the maintainers at the address in [`SECURITY.md`](https://github.com/COSS-India/voicera/blob/main/SECURITY.md) at the repository root, or use GitHub's private security advisory flow on the repository.

Please include:

* What the vulnerability is, and which component it affects.
* Steps to reproduce, or a proof of concept.
* The impact you believe it has.
* The version, branch, or commit you tested.
* Any suggested fix.

You will get an acknowledgement, an assessment, and notice when a fix ships. Please give maintainers reasonable time to respond before disclosing publicly.

## Scope

In scope: the Voicera source in this repository — the API, the runtime, the providers and telephony packages, the model server, and the deployment scripts.

Out of scope:

| Not in scope | Report to |
| --- | --- |
| Cloud AI provider vulnerabilities | That vendor |
| Telephony provider vulnerabilities | That vendor |
| Third-party dependencies | Upstream, though tell us if Voicera is affected |
| Your own deployment's configuration | See [Security hardening](../deployment/security-hardening.md) |
| Model weights and their licences | The model publisher |

## Known design characteristics

These are documented properties, not undisclosed vulnerabilities. Reporting them is not needed; deploying without accounting for them is a risk.

| Characteristic | Detail |
| --- | --- |
| The runtime's `/answer` and `/agent/{org_id}/{agent_id}` are **unauthenticated** | Required for telephony webhooks. Rate limit and allowlist at the proxy. |
| `INTERNAL_API_KEY` is a shared credential with organisation-wide reach | It mints org-scoped tokens for any organisation. Treat it as root. |
| CORS defaults to `allow_origins=["*"]` with credentials | Restrict it before exposing the API. |
| `GET /users/check/{email}` is public and confirms account existence | An enumeration surface. Rate limit it. |
| The reference Compose stack ships default passwords | Change all of them. |
| An unset `SECRET_KEY` generates a temporary key rather than failing | Set it explicitly. |
| `GET /health` returns 200 even when degraded | Probes must parse the body. |

Each is covered in [Security hardening](../deployment/security-hardening.md).

## Secrets and your data

Voicera is self-hosted, so securing a deployment is your responsibility:

* **`PROVIDER_AUTH_ENCRYPTION_KEY`** encrypts stored provider credentials. It cannot be rotated — losing it makes every stored credential permanently undecryptable. Back it up.
* **`SECRET_KEY`** signs tokens and must be identical across API replicas.
* **Call recordings and transcripts** are regulated data in most jurisdictions. Encrypt volumes at rest and set a retention policy — nothing expires automatically.

## Supported versions

The project has not cut a tagged release. Security fixes land on `dev` and flow to `main`. Track the repository for updates.

## Related

* [Security hardening](../deployment/security-hardening.md)
* [Provider credentials](../concepts/provider-auth.md)
* [Code of conduct](code-of-conduct.md)
