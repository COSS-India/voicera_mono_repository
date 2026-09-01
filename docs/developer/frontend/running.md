---
description: Run the Beta dashboard against a local Voicera stack.
---

# Running the dashboard

The dashboard is a standalone Next.js app. You fetch its branch, install its dependencies, point it at an API, and run `next dev`. It never runs inside the Compose stack.

{% hint style="warning" %}
The dashboard is **Beta**. It lives on the `dev-frontend` branch, is not merged into `dev`, and is not part of the Docker Compose stack. You run it separately against a running API.
{% endhint %}

## Before you start

Bring the core stack up first, following [Install and run](../../guides/quickstart/install-and-run.md). The dashboard is useless without an API to talk to — its first action after sign-in is `GET /users/me`, and it redirects to the sign-in page if that fails.

## Get the branch

The frontend is not on `dev`. Fetch and check out `dev-frontend` in a separate working copy so your stack checkout stays put:

```bash
git fetch origin dev-frontend
git worktree add ../voicera-frontend dev-frontend
cd ../voicera-frontend/frontend
```

If you would rather switch branches in place, `git checkout dev-frontend` works too — but you then cannot run the Compose stack from the same checkout, because `docker-compose.yaml` and the `apps/` services differ between branches.

## Install

```bash
npm install
```

`frontend/package.json` declares no `engines` field, so no Node version is pinned there. Next.js 16 is the binding constraint in practice — use an actively supported Node LTS release. If `npm install` or `next dev` fails on a version complaint, the error names the version Next requires.

## Configure

Two environment variables control where the dashboard sends traffic. Both are read directly in the source, and both have hard-coded fallbacks.

| Variable | Read in | Default | What it is |
| --- | --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `frontend/src/lib/api/http.ts` | `http://localhost:8000/api/v1` | Base URL for every REST call. Include the `/api/v1` path segment — the API modules append paths like `/agents` directly to it. |
| `NEXT_PUBLIC_RUNTIME_WS_URL` | `frontend/src/hooks/usePipecatAudio.ts` | `ws://localhost:7860` | Base URL for the browser test-call WebSocket. No path segment — the hook appends `/agent/{org_id}/{agent_id}`. |

There are no other environment variables anywhere under `frontend/`, and no `.env.example` is committed (`.env*` is gitignored). `frontend/next.config.ts` is an empty config object — no rewrites, no proxy, no custom headers.

If your API and runtime are on the default ports on `localhost`, you can skip this step entirely and rely on the fallbacks. Otherwise create `frontend/.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
NEXT_PUBLIC_RUNTIME_WS_URL=ws://localhost:7860
```

The `NEXT_PUBLIC_` prefix means both values are inlined into the browser bundle at build time. They are not secrets, and changing them requires a restart of `next dev` (or a rebuild for `next start`).

## Run

```bash
npm run dev
```

The dashboard comes up on `http://localhost:3000`. The available scripts are exactly the `create-next-app` defaults:

| Script | Command |
| --- | --- |
| `npm run dev` | `next dev` |
| `npm run build` | `next build` |
| `npm run start` | `next start` — serves a prior `build` |
| `npm run lint` | `eslint` |

Open `http://localhost:3000`. You land on the sign-in page. If you have not created an account yet, use the signup link — signup creates the organisation and its first `super_admin`, the same as `POST /users/signup`.

## Pointing at a local stack

Ports must match what the Compose stack actually published. Check them against [Ports and defaults](../reference/ports-and-defaults.md), and against your `.env` if you overrode `RUNTIME_HOST_PORT` or the API port.

```bash
# API reachable?
curl -s http://localhost:8000/api/v1/languages | head -c 200

# Runtime reachable?
curl -s http://localhost:7860/health
```

If both respond and the dashboard still shows an error banner, open the browser devtools network tab: a failing request shows the exact URL the dashboard built, which tells you immediately whether `NEXT_PUBLIC_API_URL` is wrong or the token expired.

On a `401` from any request, `apiFetch` clears the stored session and hard-redirects to `/`. A sudden bounce back to the sign-in screen means your token was rejected, not that the page crashed.

### CORS

You do not need a proxy or a rewrite. The API sets `allow_origins=["*"]` in `apps/api/app/main.py`, so a browser on `http://localhost:3000` can call `http://localhost:8000` directly.

{% hint style="warning" %}
That wildcard is convenient for local development and wrong for production. If you expose the dashboard beyond your machine, narrow `allow_origins` to your real origins first. See [Security hardening](../../guides/deployment/security-hardening.md).
{% endhint %}

## Why it is not in Compose

Three reasons, all verifiable:

* **It is on a different branch.** `docker-compose.yaml` on `dev` describes the API, runtime, FerretDB, MinIO, and Redis. The `frontend/` directory does not exist on that branch, so there is nothing for a service definition to build.
* **There is no Dockerfile for it.** `frontend/` contains no container build at all.
* **The stack does not need it.** The API's only reference to the dashboard is `FRONTEND_URL` (default `http://localhost:3000`), used to build password-reset links in `apps/api/app/services/user_service.py`. Nothing else calls it, and every dashboard feature is reachable over the API.

Running it outside Compose is therefore the intended arrangement, not a workaround. Point it at whichever API you want and start it with `npm run dev`.

## Related

* [Overview](overview.md)
* [Dashboard tour](dashboard-tour.md)
* [Install and run](../../guides/quickstart/install-and-run.md)
* [Environment variables](../reference/environment-variables.md)
