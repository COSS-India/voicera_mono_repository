# Documentation Source Briefs

One file per item from the VoicEra documentation review. Each brief gives the documentation writer (or AI) factual context from the codebase.

## How to use

1. Open the brief for the review item you are writing.
2. Expand into user-facing documentation (plain language for B*, technical detail for A*).
3. **A7:** Document MIT license (COSS India, 2026) — see `LICENSE` file.

## Index

| File | Review item |
|------|-------------|
| [A1-submodule-readmes.md](./A1-submodule-readmes.md) | Submodule READMEs |
| [A2-api-reference.md](./A2-api-reference.md) | API reference |
| [A3-vobiz-telephony.md](./A3-vobiz-telephony.md) | Vobiz / telephony |
| [A4-integration-pipeline-languages.md](./A4-integration-pipeline-languages.md) | Pipeline, languages, fallback |
| [A5-ai4bharat-servers.md](./A5-ai4bharat-servers.md) | AI4Bharat STT/TTS servers |
| [A6-security-production-hardening.md](./A6-security-production-hardening.md) | Security & production |
| [A7-license.md](./A7-license.md) | License contradiction |
| [A8-johnaic-public-urls.md](./A8-johnaic-public-urls.md) | JOHNAIC / public URLs |
| [B1-plain-language-overview.md](./B1-plain-language-overview.md) | Plain-language overview |
| [B2-before-you-start-checklist.md](./B2-before-you-start-checklist.md) | Prerequisites checklist |
| [B3-dashboard-walkthrough.md](./B3-dashboard-walkthrough.md) | Dashboard walkthrough |
| [B4-glossary.md](./B4-glossary.md) | Glossary |
| [B5-deployment-walkthrough.md](./B5-deployment-walkthrough.md) | Deployment steps |
| [B6-how-to-know-its-working.md](./B6-how-to-know-its-working.md) | Success verification |
| [B7-operations-guide.md](./B7-operations-guide.md) | Operations guide |
| [B8-faq.md](./B8-faq.md) | FAQ |

## Important correction (all telephony docs)

**Vobiz Auth ID and Auth Token** are stored in **Dashboard → Integrations** (MongoDB), not in `.env` for normal operation.
