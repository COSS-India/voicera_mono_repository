# Contributing to VoicERA

Thanks for your interest in contributing! This guide covers how to set up the project, make
changes, and submit them. By participating you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

> New to the codebase? Read [AGENTS.md](AGENTS.md) first — it explains the service layout,
> the config-driven agent model, and where everything lives.

---

## Ways to contribute

- 🐛 **Report bugs** — open an issue with the bug template.
- ✨ **Request features** — open an issue with the feature template (please discuss large
  changes before building them).
- 📝 **Improve docs** — the `docs/` site (MkDocs) and these top-level files.
- 🔌 **Add a provider** — LLM/STT/TTS or telephony (see [AGENTS.md](AGENTS.md#extending-voicera)).
- 🧹 **Fix/refactor** — bug fixes, tests, and cleanups are always welcome.

---

## Development setup

### Prerequisites
- Docker & Docker Compose
- Node.js 18+
- Python 3.10+
- (Optional) CUDA-capable GPU for local AI4Bharat STT/TTS servers

### 1. Fork & clone

```bash
git clone https://github.com/<your-username>/voicera_mono_repository.git
cd voicera_mono_repository
git checkout -b feat/your-change
```

### 2. Configure environment

```bash
cp voicera_backend/env.example       voicera_backend/.env
cp voicera_frontend/.env.example     voicera_frontend/.env.local
cp voice_2_voice_server/.env.example voice_2_voice_server/.env
```

See the [README → Environment Configuration](README.md#environment-configuration) for
variable details. **Never commit `.env` files or secrets.**

### 3. Run the stack

```bash
# Full stack in Docker
make build-all-services && make start-all-services   # dashboard: http://localhost:3000

# OR: infra in Docker, apps locally
make start-backend-services
cd voicera_frontend && npm install && npm run dev
cd voice_2_voice_server && python -m venv venv && source venv/bin/activate \
  && pip install -r requirements.txt && python main.py
```

`make stop-all-ports` frees ports 3000/8000/8001/8002/7860/27017 if something is stuck.

---

## Making changes

### Code style
- **Python:** follow PEP 8, use type hints and docstrings. Format with:
  ```bash
  pip install black isort && black . && isort .
  ```
- **TypeScript/React** (in `voicera_frontend/`):
  ```bash
  npm run lint
  npm run format   # if configured
  ```

### Tests
Run the tests for the service(s) you touched:

```bash
cd voicera_backend && pytest          # backend
cd voicera_frontend && npm test       # frontend
```

> There is no repo-wide test/lint command — run them per service.

### Verify end-to-end
Bring up the stack, create an agent in the dashboard, and use **Test in Browser** to confirm
the STT → LLM → TTS path still works. No telephony credentials are required for this.

---

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): short summary

Optional body explaining what and why.
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`.

Examples:
```
feat(voice-server): add Cartesia voice cloning option
fix(backend): return 404 when agent_id is unknown
docs(readme): clarify browser testing flow
```

---

## Pull requests

1. Keep PRs focused and reasonably small.
2. Update docs and the [CHANGELOG.md](CHANGELOG.md) (`Unreleased` section) when behavior
   changes.
3. Ensure relevant tests pass and the app still runs.
4. Fill out the PR template; link the issue it resolves.
5. PRs target the `main` branch and require at least one maintainer approval.

---

## Reporting security issues

**Do not** open public issues for vulnerabilities. Follow [SECURITY.md](SECURITY.md).

---

## Questions

Open a GitHub Discussion or issue. Thanks for helping make VoicERA better! 🎙️
