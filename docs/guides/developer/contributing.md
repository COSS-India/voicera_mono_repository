---
description: How to fork, branch, code, test, and submit a pull request for VoicEra.
---

# Contributing

The end-to-end contribution flow for VoicEra — from fork to merged PR. Audience: external contributors and new team members.

## Code of conduct

Be respectful, constructive, and inclusive. Help others learn. Harassment and discrimination are not tolerated.

## Getting started

1. Fork the repository at `https://github.com/COSS-India/voicera_mono_repository`.
2. Clone your fork:

   ```bash
   git clone https://github.com/<your-username>/voicera_mono_repository.git
   cd voicera_mono_repository
   ```

3. Set up the dev environment — see [local setup](local-setup.md).
4. Create a branch:

   ```bash
   git checkout -b feat/your-feature
   ```

## Before you start coding

- Check existing [issues](https://github.com/COSS-India/voicera_mono_repository/issues) for related work.
- Open a new issue for non-trivial features before writing code.
- Discuss API or schema changes in the issue first.

## Code style

### Python

Follow PEP 8 and use type hints. Comments and docstrings are optional — add them only when the intent is non-obvious from the code itself.

```python
def calculate_statistics(
    call_logs: list[CallLog],
    filter_date: datetime,
) -> dict[str, float]:
    return {"total": len(call_logs)}
```

Format before committing:

```bash
pip install black isort
black .
isort .
```

### TypeScript

Use TypeScript everywhere, prefer `const`/`let`, and write arrow functions for handlers.

```typescript
interface Agent {
  id: string;
  name: string;
  config: AgentConfig;
}

/** Fetch agent details by id. */
export const getAgent = async (agentId: string): Promise<Agent> => {
  // ...
};
```

Format and lint:

```bash
npm run format
npm run lint
```

## Commit messages

Use Conventional Commits:

```
type(scope): subject

optional body

optional footer
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

Examples:

```
feat(agent): add agent cloning
fix(api): return correct status code for 404
docs(readme): update install steps
test(backend): cover auth service edge cases
chore(deps): bump fastapi to 0.110
```

## Testing

Every change ships with tests. See the [testing guide](testing.md) for command reference.

```bash
# Backend
cd voicera_backend && pytest

# Frontend
cd voicera_frontend && npm test
```

Cover unit logic, API contracts, and any critical user flow you touched.

## Documentation

Update docs in the same PR as the code change. New features get a dedicated page under the right section. Each docs page should answer:

- What does it do?
- How do I use it?
- What configuration applies?
- What can go wrong?

## Pull request

1. Push the branch:

   ```bash
   git push origin feat/your-feature
   ```

2. Open the PR on GitHub.
3. Fill in the template:

   ```markdown
   ## Description
   Brief summary of the change.

   ## Type of change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation

   ## Changes
   - ...

   ## Testing
   - [ ] Tested locally
   - [ ] Added or updated tests
   - [ ] CI passes

   ## Checklist
   - [ ] Follows the style guide
   - [ ] Docs updated
   - [ ] No new warnings
   ```

### Review expectations

- At least one approval is required.
- CI must be green.
- Coverage should not drop.
- Docs must be updated for user-visible changes.

## Reviewing PRs

When you review, aim to:

- **Approve** when the code is correct, tested, documented, and follows the style guide.
- **Request changes** for logic bugs, missing tests, security concerns, or unclear naming.
- **Comment** with non-blocking suggestions and questions.

## Architecture decision records

For significant architectural changes, add an ADR under `docs/internal/adr/`:

```markdown
# ADR 001: Use Pipecat for voice processing

## Status
Accepted

## Context
Need real-time voice processing for agent calls.

## Decision
Adopt the Pipecat framework for the voice pipeline.

## Consequences
- Benefits: real-time, modular, extensible.
- Drawbacks: learning curve.
- Mitigations: docs, examples, pairing.
```

## Repository layout

### Backend

```
voicera_backend/
├── app/
│   ├── main.py            # FastAPI app
│   ├── config.py
│   ├── models/            # data structures
│   ├── routers/           # HTTP endpoints
│   ├── services/          # business logic
│   ├── schemas/           # request/response
│   └── utils/
├── tests/
└── requirements.txt
```

### Frontend

```
voicera_frontend/
├── app/                   # Next.js routes
├── components/            # React components
├── hooks/                 # custom hooks
├── lib/                   # API clients, helpers
├── public/                # static assets
├── styles/
└── __tests__/
```

### Voice server

```
voice_2_voice_server/
├── api/
│   ├── server.py
│   ├── bot.py
│   └── services.py
├── config/
├── services/              # provider integrations
│   ├── ai4bharat/
│   ├── bhashini/
│   └── kenpath_llm/
└── main.py
```

See [architecture](../../concepts/architecture.md) for the runtime picture.

## Security

### Before opening a PR

- No hardcoded secrets or API keys.
- Validate every external input.
- Sanitise rendered values to avoid XSS.
- Use parameterised queries (no string concatenation for Mongo filters).
- Confirm error messages do not leak internals.
- Run `safety check` and `npm audit`.

### Reporting a vulnerability

Do not open a public issue. Email the maintainers privately with:

- Description of the vulnerability.
- Steps to reproduce.
- Potential impact.
- Suggested fix, if any.

## Performance guidelines

### Backend

```python
# Use async I/O end-to-end.
async def get_agent(agent_id: str) -> Agent:
    return await Agent.find_by_id(agent_id)

# Cache hot reads.
@cache(expire=300)
async def get_agent_config(agent_id: str):
    return await Agent.find_by_id(agent_id)
```

### Frontend

```tsx
// Memoise expensive components.
const ExpensiveComponent = React.memo(({ data }) => <div>{data}</div>);

// Memoise expensive computations.
const value = useMemo(() => compute(a, b), [a, b]);

// Code-split heavy modules.
const Heavy = dynamic(() => import('./Heavy'));
```

## Release process

VoicEra follows semantic versioning: `MAJOR.MINOR.PATCH`.

| Bump | When |
|------|------|
| MAJOR | Breaking changes |
| MINOR | New features, backward-compatible |
| PATCH | Bug fixes |

To cut a release:

```bash
# Update version in package.json / pyproject.toml
# Update CHANGELOG.md
git tag -a v1.2.3 -m "Release v1.2.3"
git push origin v1.2.3
```

## Getting help

- Questions: GitHub Discussions.
- Bugs: GitHub Issues.
- Security: private email to maintainers.

## Next steps

- [Local setup](local-setup.md)
- [Testing](testing.md)
- [Architecture](../../concepts/architecture.md)
- [REST API reference](../../reference/rest-api.md)
