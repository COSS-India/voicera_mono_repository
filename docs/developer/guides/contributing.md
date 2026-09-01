---
description: How to propose changes to Voicera.
---

# Contributing

Contributions are welcome. This page covers the workflow; [Local setup](local-setup.md) covers getting the code running.

## Before you start

* Open an issue first for anything substantial. A design discussion is cheaper than a rejected pull request.
* Read [Repository layout](repository-layout.md) so your change lands in the right package.
* Adding a provider or a telephony vendor? Follow [Adding an AI provider](adding-a-provider.md) or [Adding a telephony provider](adding-a-telephony-provider.md) — both are designed as extension points, so you should not need to touch shared code.

## Setting up

```bash
git clone https://github.com/COSS-India/voicera.git
cd voicera
./scripts/start_docker.sh
```

There is **no** `make` target and **no** `pip install -e .` — `Makefile` and `pyproject.toml` are empty placeholders. Install per app:

```bash
pip install -r apps/api/requirements.txt
pip install -r apps/runtime/requirements.txt
```

## Branching

Branch from `dev`, not `main`:

```bash
git checkout dev
git pull
git checkout -b your-change
```

Use a short descriptive name. Existing branches follow patterns like `dev-<feature>` and `chore/<task>`.

## Commit messages

Match the existing history: a sentence-case imperative summary, no type prefix, no trailing period.

```
Add campaign CSV upload functionality and validation
Enhance call log patching functionality and introduce new endpoint
Standardise both STT models on /v1/realtime; detect MPS rather than assume it
```

Add a body when the reason is not obvious from the diff. Keep the summary under about 72 characters.

## Code style

| Rule | Value |
| --- | --- |
| Line length | 100 |
| Target Python | 3.12 (`model-server`); the apps run on 3.11+ |
| Type hints | Expected on new code; the codebase uses `from __future__ import annotations` |
| Docstrings | One line on modules and public functions |

`model-server/ruff.toml` is the lint configuration for that tree. It deliberately excludes vendored upstream model code — restyling somebody else's project turns every upstream sync into a merge conflict — but everything authored here is linted.

```bash
cd model-server && ruff check .
```

Match the surrounding code. The repository favours explicit registries over if/elif chains, Pydantic models at boundaries, and small modules over large ones.

## Run the checks

There is **no CI**, so nothing runs these for you. Run them before opening a pull request:

```bash
python -m pytest apps/api/tests
python -m pytest apps/runtime/tests
python -m pytest apps/telephony/tests
python -m pytest apps/providers/tests
cd model-server && python -m pytest tests
```

The model-server suite needs no GPU. See [Testing](testing.md) for what each suite protects.

## Opening a pull request

Target `dev`. In the description:

* What changed and why.
* Which suites you ran, and their result.
* Anything you deliberately left out.
* For behaviour changes, how a reviewer can reproduce it.

Keep pull requests focused — one concern each. A refactor bundled with a feature is hard to review and harder to revert.

## Documentation

Documentation lives in `docs/` and is published with GitBook. If your change alters behaviour, update the page that describes it — a feature with stale docs is a feature people cannot use.

`.docs-meta/STYLE.md` is the style contract: front matter, hint and tab usage, mermaid conventions, naming, and the rule that every value must come from the source rather than from memory.

New page? Add it to `docs/SUMMARY.md`, or GitBook will not show it.

## Reporting security issues

Do **not** open a public issue for a vulnerability. Follow the [Security policy](../../guides/legal/security.md).

## Code of conduct

Participation is governed by the [Code of conduct](../../guides/legal/code-of-conduct.md). Be respectful, assume good faith, and keep criticism on the work.

## Licence

Contributions are accepted under the MIT licence. By opening a pull request you agree your contribution may be distributed under it. See [License](../../guides/legal/license.md).

## Related

* [Local setup](local-setup.md)
* [Testing](testing.md)
* [Repository layout](repository-layout.md)
