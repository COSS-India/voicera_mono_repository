---
description: What teams build with VoicEra — inbound helplines, outbound campaigns, IVR replacement, and more.
---

# Use cases

VoicEra is built for voice workloads that need **Indian languages, real phone numbers, and self-hosting**. The patterns below cover most deployments.

## Inbound helplines

A citizen dials a published number; an AI agent answers, understands the question, and either resolves it or routes to a human.

**Typical stack**

* One Vobiz inbound number per language.
* One agent per language, each grounded with a [knowledge base](../concepts/knowledge-base-rag.md).
* Call recording enabled for QA.

**Examples** — government scheme enquiries, exam result lookup, district-level grievance intake.

## Outbound campaigns

The system places calls from a list — for surveys, reminders, or notifications — and logs each outcome.

**Typical stack**

* Single Vobiz outbound trunk.
* One agent with a tightly scripted prompt.
* Contact list uploaded via [Dashboard → Campaigns](../guides/operator/dashboard-tour.md).
* Webhooks notify upstream systems of completed calls.

**Examples** — health-check reminders, vaccination follow-ups, scheme awareness drives.

## IVR replacement

Replace fixed-menu IVR ("press 1 for English") with natural-language understanding. The caller just speaks the request.

**Typical stack**

* Single inbound number.
* One multilingual agent (auto-detect or first-utterance routing).
* Knowledge base for FAQs.

## Appointment scheduling and reminders

Agents read a calendar, propose slots, confirm, and write the booking back via webhook.

**Typical stack**

* Inbound agent for "I want to book…" flows.
* Outbound agent for "Your appointment is tomorrow at 10 AM" reminders.
* Custom LLM tool/webhook to your scheduling system.

## Internal voice assistant

A staff-only number that answers operational queries — policy lookups, status of internal tickets, document search.

**Typical stack**

* Restricted number with PIN-based auth.
* Knowledge base loaded with internal SOPs.
* No call recording, or recording with restricted retention.

## Next steps

* [Prerequisites](../quickstart/prerequisites.md) — what you need before installing.
* [Install and run](../quickstart/install-and-run.md) — get a local stack up.
* [Telephony model](../concepts/telephony-model.md) — how Vobiz fits in.
