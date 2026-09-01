---
description: A tour of the dashboard's pages.
---

# Dashboard tour

Every screen in the dashboard, what it shows, and which API endpoints back it. Several screens are not wired to the API yet — this page says which, so you do not mistake sample data for your own.

{% hint style="warning" %}
The dashboard is **Beta**. It lives on the `dev-frontend` branch, is not merged into `dev`, and is not part of the Docker Compose stack. You run it separately against a running API.
{% endhint %}

## The layout shell

Every signed-in route lives under the `(app)` route group and shares one layout, `frontend/src/app/(app)/layout.tsx`. It wraps the page in two things:

* **`AuthProvider.tsx`** — the auth gate. On mount it reads the session from `localStorage` and calls `GET /users/me`. No stored session, or a failed call, and it clears storage and redirects to `/`. Children never render without a session, so no page needs its own auth check. It also merges the organisation name into the stored session so the sidebar can show it.
* **`AppSidebar.tsx`** — a collapsible nav rail, its expanded state persisted in `localStorage` under `voicera_sidebar_expanded`. It groups routes into **Build** (Agents, Numbers, Knowledge Base, Batches, History, Analytics, Telemetry) and **Workspace** (Members, Integrations), with a Walkthrough link and a profile menu in the footer.

Note that the sidebar does not link to `/agent-creation`, `/library`, `/components`, or `/account` — those are reached from within pages or from the profile menu.

`PageHeader.tsx` is a shared eyebrow-plus-title-plus-description header. Only the UI-kit pages use it; the main dashboard screens write their own headers.

Sign-in and signup sit outside the shell at `/` and `/signup`, backed by `POST /users/login` and `POST /users/signup`.

## At a glance

| Route | Status | Backed by |
| --- | --- | --- |
| `/dashboard` | Live | `GET /agents`, `GET /calls/org/{org_id}` |
| `/agent-creation` | Live | `GET /configuration/*`, `GET /auth/configured`, `POST /agents` |
| `/agents/{agentId}/edit` | Live | `GET /agents/{id}`, `PATCH /agents/{id}` |
| `/numbers` | Live | `GET /phone-numbers`, attach/detach, provider inventory |
| `/history` | Live | `GET /calls/org/{org_id}`, recording and transcript blobs |
| `/members` | Live | `GET /members/{org_id}`, invite, assign-admin, remove |
| `/integrations` | Live | `GET /auth/catalog`, `GET /auth/configured`, `POST /auth` |
| `/account` | Live | `GET /users/me`, `GET /users/organisations`, switch-organisation |
| `/batches` | **Sample data** | Nothing — static `dashboard-data.ts` |
| `/knowledge-base` | **Sample data** | Nothing — static `dashboard-data.ts` |
| `/analytics` | **Placeholder** | Nothing |
| `/telemetry` | **Placeholder** | Nothing |
| `/library` | Internal | Nothing — static prompt modules |
| `/components` | Internal | Nothing — UI kit |
| `/walkthrough` | **Placeholder** | Nothing |

## dashboard

The agents home, rendered by `AgentsHome.tsx`. It lists every agent in your active organisation from `GET /agents`, and pulls up to 500 recent calls with `GET /calls/org/{org_id}` to show per-agent activity alongside each card.

Each card shows the agent's name, its purpose (derived from the first sentence of its system prompt by `agentPurposeFromApi()`), and either its linked phone number or a WebSocket badge. The distinction is read from `agent_category` directly, never guessed from whether a number field is populated.

The card's test action branches on that same field, and this is the most useful thing to understand about the page:

* A **`telephony`** agent opens `TestCallSheet.tsx` — you type an E.164 number and it fires `POST /calls/outbound`, placing a real phone call from the agent's linked number.
* A **`websocket`** agent opens `AgentTestModal.tsx` — a live browser microphone call over the runtime WebSocket. See [Browser test calls](test-calls.md).

The page can also duplicate an agent (`GET /agents/{id}` then `POST /agents` with the copied config) and delete one (`DELETE /agents/{id}`).

## agents/[agentId]/edit

Loads one agent with `GET /agents/{agent_id}`, runs it through `agentToForm()` to rebuild the wizard's flat form shape, and renders the same `AgentStackFields` and `AgentFieldsStep` components the wizard uses. Saving maps the form back with `formToAgentCreatePayload()` and sends `PATCH /agents/{agent_id}`.

Because it shares the wizard's mapper, it shares the wizard's gaps: the token, temperature, and audio-buffer controls are displayed but not sent. See [Agent creation wizard](agent-wizard.md).

## agent-creation

The six-step wizard. It has its own page: [Agent creation wizard](agent-wizard.md).

## numbers

Phone number inventory, rendered by `PhoneNumbers.tsx` over the `usePhoneNumbers` hook.

| Action | Endpoint |
| --- | --- |
| List the organisation's numbers | `GET /phone-numbers` |
| List numbers on a provider account not yet imported | `GET /phone-numbers/providers/{provider}/inventory` |
| Import a number, optionally linking it to an agent | `POST /phone-numbers/attach` |
| Unlink from its agent and from the provider | `DELETE /phone-numbers/detach` |

Each row carries an audit line built from `last_link_action`, `last_link_by_email`, and `last_link_at` — who attached, detached, or imported the number and when. Detaching keeps the inventory row; it only breaks the agent link and the provider-side binding. Background in [Telephony model](../../guides/concepts/telephony-model.md).

## batches

{% hint style="warning" %}
**Batches and Knowledge Base show fixed sample data, not yours.** Both import their contents from `frontend/src/lib/dashboard-data.ts` and make no API calls at all. Their filters, counts, progress bars, and action buttons operate on a static array, so nothing you do on either screen reaches the API and your real campaigns and documents never appear.
{% endhint %}

`Campaigns.tsx` reads `CAMPAIGNS` and `BROADCAST_TEMPLATES` from that file. The pause, resume and finish-setup buttons only re-filter the static rows.

Run campaigns over the API instead — see [Running a campaign](../../guides/operator/running-a-campaign.md) and [Campaigns](../../guides/concepts/campaigns.md).

## history

The one screen where the dashboard clearly beats the API, rendered by `History.tsx`.

It pages through `GET /calls/org/{org_id}` twenty rows at a time, joins each call to its agent (`GET /agents`), and lets you filter by call type and by status — `completed`, `failed`, `in_progress`, `ringing`, `initiated`. Phone numbers are masked in the list by `maskPhoneNumber()`. A bulk export uses `listAllOrgCalls()`, which pages the whole organisation a hundred rows at a time, and can pull every transcript with `GET /calls/{call_id}/transcript`.

Clicking a row opens `CallDetailSheet.tsx`, which fetches two things:

* `GET /calls/{call_id}/recording` — an authenticated proxy in front of MinIO. It is fetched as a blob because `<audio src>` cannot send an `Authorization` header; the sheet builds an object URL for playback and decodes the same blob for the waveform.
* `GET /calls/{call_id}/transcript` — parsed by `frontend/src/lib/transcript.ts` into `[timestamp] role: content` lines.

Transcript lines are anchored so the first line is t=0 and later lines are offset by their deltas. There is no shared clock between the transcript's timestamps and the recording's start, so the alignment between text and audio is a close estimate rather than an exact sync.

Browser test calls appear here alongside telephony calls: the runtime registers a `call_type: web` CallLog on connect, with a transcript and recording.

## knowledge-base

`KnowledgeBase.tsx` imports `DOCS` from the same static `dashboard-data.ts` and calls no API. The upload dialog animates a progress bar and uploads nothing; the Indexed / Indexing / Failed counts are properties of the static array.

Upload and index documents over the API — see [Managing knowledge documents](../../guides/operator/managing-knowledge.md) and [Knowledge base (RAG)](../../guides/concepts/knowledge-base-rag.md). The wizard's knowledge-base document picker draws from the same static `KB_DOCS` list, so document ids selected there will not match real documents.

## members

Fully wired, over the `useMembers` hook and `frontend/src/lib/api/members.ts`.

| Action | Endpoint | Who can |
| --- | --- | --- |
| List members | `GET /members/{org_id}` | Any member |
| Add a member | `POST /members/invite` | `admin`, `super_admin` |
| Promote to admin | `POST /members/assign-admin` | `super_admin` |
| Remove from the organisation | `POST /members/remove` | `super_admin` |

Cards sort highest-rank-first: `super_admin`, then `admin`, then `member`. There is no accept-invite step — `POST /members/invite` creates the account directly in your active organisation with a password you set, so you hand the credentials over yourself. The "add member" link the modal generates points at `/add-member/{uid}`, a shareable page identifier; the organisation context travels in query parameters. Roles are explained in [Multi-tenancy and roles](../../guides/concepts/multi-tenancy.md).

## integrations

The credential manager, rendered by `Integrations.tsx`. It is the screen you need before the wizard is usable, because the wizard hides every provider you have not configured here.

`GET /auth/catalog` returns each provider's field schema — types, descriptions, examples, and which fields are secret. `groupProvidersByKind()` buckets them into STT, TTS, LLM, and telephony sections, with providers that serve several kinds appearing in each. `GET /auth/configured` marks which are already connected.

Saving a provider sends `POST /auth` with `{ provider, auth }`. `GET /auth/{provider}` reads a stored entry back and `DELETE /auth/{provider}` removes it. Secret fields render behind a show/hide toggle. Credentials are encrypted at rest by the API — see [Provider credentials (ProviderAuth)](../../guides/concepts/provider-auth.md).

Because the form is generated from the catalog rather than hand-written, it always matches what the API accepts. That makes this screen genuinely more reliable than reading credential field names out of documentation.

## analytics

A stub. `Analytics.tsx` renders a heading, a paragraph describing volume, latency, drop-off and cost per agent as intended future content, and one button that routes to `/dashboard`. It makes no API calls and computes nothing.

The language map it refers to lives in `Languages.tsx`, which draws a d3 and topojson choropleth from `LANGS_GEO` in `dashboard-data.ts` — also static sample figures, not your call volumes.

## telemetry

A placeholder card reading "Real-time agent telemetry coming soon." The file is nine lines long and has no logic.

## library

Internal. A standalone browser for the nine static prompt modules in `frontend/src/lib/prompt-modules.ts` — the same content as the wizard's prompt library dialog, minus the insert action. Its own header calls it part of the "VoicEra UI kit". Useful for reading the modules; it is not an operational screen.

## account

Your profile and organisation switcher, rendered by `Account.tsx`. It loads `GET /users/me` and `GET /users/organisations` in parallel, and switching organisation calls `POST /users/switch-organisation` and writes the returned token back to `localStorage`, so the whole dashboard re-scopes to the new organisation.

`frontend/src/lib/api/organisations.ts` also exposes `DELETE /organisations/{org_id}`, restricted by the API to a `super_admin` acting on their own active organisation.

## components

Internal. A UI-kit gallery of every shared primitive — buttons, inputs, cards, badges, switches, spinners, progress bars, tooltips, toasts, modals, the stepper, the nav rail — each shown with its variants. It is a developer reference for the design system, not a product feature. Nothing here reads or writes data.

## walkthrough

A placeholder card reading "Interactive product tour coming soon." It is linked from the sidebar footer, which makes it more visible than its content warrants.

## Real features versus scaffolding

Being blunt about it, because the sidebar gives all of these equal weight:

* **Genuine, API-backed features:** Agents home, agent creation, agent edit, Numbers, History, Members, Integrations, Account.
* **Screens that look complete but are sample data:** Batches, Knowledge Base, and the language map. They render convincing tables and charts from a hardcoded array. Do not read them as your data.
* **Placeholders:** Analytics, Telemetry, Walkthrough — each a single card admitting the feature is not built.
* **Developer scaffolding:** `/components` and `/library` are UI-kit pages, self-described as such. They are unlinked from the sidebar and safe to ignore.

One more piece of scaffolding worth knowing about: `frontend/src/app/api/` contains three Next.js route handlers (`/api/agents`, `/api/auth/login`, `/api/auth/signup`) whose own comments call them "a UI-kit demo, not wired to a real database" — one keeps agents in a module-level array that resets on restart, the others return a mock user and a fake token. The live dashboard does not call them; real auth goes to the API's `/users/login` and `/users/signup`. Do not mistake them for a backend.

## Related

* [Overview](overview.md)
* [Running the dashboard](running.md)
* [Agent creation wizard](agent-wizard.md)
* [Browser test calls](test-calls.md)
* [Calls and call artifacts](../../guides/concepts/calls.md)
* [REST API](../../api-reference/overview.md)
