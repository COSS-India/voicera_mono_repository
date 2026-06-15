# Security Policy

## Supported Versions

VoicERA is under active development. Security fixes are applied to the latest
`main` branch. We recommend running the most recent commit or Docker image built
from `main`.

| Version   | Supported          |
| --------- | ------------------ |
| `main`    | :white_check_mark: |
| `< 0.1.0` | :x:                |

There is no formal release cadence yet; see [CHANGELOG.md](CHANGELOG.md) for
notable changes.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

If you discover a security issue, report it privately so we can investigate and
patch before public disclosure:

**Email:** [security@voicera.ai](mailto:security@voicera.ai)

Include as much detail as possible:

- Description of the vulnerability and potential impact
- Steps to reproduce (proof of concept if available)
- Affected component(s) (backend, voice server, frontend, etc.)
- VoicERA version or commit hash, if known
- Your suggested fix or mitigation, if you have one

### What to expect

- **Acknowledgment** within 3 business days
- **Status update** as the investigation progresses
- **Coordinated disclosure** — we will work with you on a reasonable timeline
  before any public announcement

We appreciate responsible disclosure and will credit reporters in the
[CHANGELOG.md](CHANGELOG.md) when fixes ship (unless you prefer to remain
anonymous).

## Scope

In scope:

- Authentication and authorization flaws in the backend API
- Cross-tenant data access (org isolation bypasses)
- Remote code execution or server-side injection in any service
- Secrets or credentials exposed in logs, responses, or client bundles
- WebSocket / telephony endpoint abuse that compromises other tenants

Out of scope (please still report if severe, but may be handled as regular bugs):

- Denial-of-service against self-hosted deployments without a practical fix
- Issues requiring physical access to the host
- Social engineering of maintainers or users
- Missing security headers or best-practice hardening with no demonstrated exploit

## Security Best Practices for Self-Hosters

- Keep `INTERNAL_API_KEY` identical between backend and voice server; rotate it
  if compromised.
- Store provider API keys in per-organization Integrations or `.env` — never in
  agent configs or version control.
- Do not expose MongoDB (27017) or MinIO (9000) to the public internet.
- Use HTTPS and a reverse proxy (see [docs/deployment/](docs/deployment/docker.md))
  in production.
- Review [README → Environment Configuration](README.md#environment-configuration)
  before going live.
