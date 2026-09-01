---
description: What people build with Voicera, and what each one takes to configure.
---

# Use cases

Voicera covers two shapes of work: **answering** calls that come to you, and **placing** calls to a list. Everything below is a variation on one of those.

## Inbound helpline

A public number that answers immediately, in the caller's language, at any hour.

**Typical of:** citizen services, grievance intake, health and welfare-scheme information lines.

| What you configure | Where |
| --- | --- |
| A `telephony` agent with your prompts and greeting | [Create your first agent](../quickstart/first-agent.md) |
| A phone number attached to it | [Agents](../concepts/agents.md) |
| Documents it may answer from | [Knowledge base](../concepts/knowledge-base-rag.md) |
| A public HTTPS and WSS address | [Public voice URLs](../deployment/public-voice-urls.md) |

Callers reach an agent that answers on the first ring and never queues. Every call is transcribed and recorded, so escalations have a record.

## Outbound campaign

Dial a list of numbers, hold the same short conversation with each, and record what happened.

**Typical of:** appointment and payment reminders, scheme-enrolment drives, surveys, follow-ups after a service visit.

| What you configure | Where |
| --- | --- |
| A contact CSV | [Running a campaign](../operator/running-a-campaign.md) |
| Retry policy — how often, how long apart | [Campaigns](../concepts/campaigns.md) |
| Calling hours, so you do not dial at night | [Campaigns](../concepts/campaigns.md) |
| A concurrency ceiling | [Call concurrency](../concepts/call-concurrency.md) |

The orchestrator paces the run, retries the numbers worth retrying, and **stops the campaign on its own** if the failure rate crosses your threshold — so a misconfigured agent burns a handful of calls, not the whole list.

## IVR replacement

Replace a menu tree with a question.

**Typical of:** any line currently starting "press 1 for…".

Instead of navigating options, the caller says what they want and the agent routes or answers directly. Configuration is the same as an inbound helpline; the difference is in the prompt — describe what the agent can help with rather than enumerating branches.

{% hint style="info" %}
Callers interrupt menus constantly. Barge-in is on by default, and `interruption_min_words` tunes how eagerly the agent yields. See [Agent configuration](../../developer/reference/agent-configuration.md).
{% endhint %}

## Document-grounded support

An agent that answers from your PDFs rather than from what the model happens to know.

**Typical of:** scheme eligibility rules, policy documents, product manuals, admissions criteria.

Upload the documents, attach them to the agent, and pick a retrieval mode: `context` prepends relevant passages to every turn, `tool` lets the model decide when to look something up. See [Knowledge base](../concepts/knowledge-base-rag.md).

## Multilingual service line

One number, several languages.

Set the agent's primary language and list the secondary ones, then choose providers whose models cover them. The provider catalog reports supported languages per model, so you can check coverage before committing.

{% hint style="warning" %}
An agent declares its languages, but Voicera does **not** switch language mid-call. Choose a provider whose model handles the languages you expect, or run separate numbers per language.
{% endhint %}

## Data-sensitive deployment

Everything above, with nothing leaving your network.

**Typical of:** government deployments and anyone whose call content cannot go to a third-party API.

Run the [model server](../../developer/model-server/overview.md) on your own GPUs so speech and language processing stay in-house. Your telephony provider still carries the audio — that is unavoidable for phone calls — but no model vendor sees it. See [Self-hosted models](../deployment/self-hosted-models.md).

## Testing and development

Before spending call minutes, use a `websocket` agent: it talks to a browser instead of a phone, needs no telephony account, and exercises the same pipeline.

{% hint style="info" %}
Browser sessions are recorded too — the runtime registers a `call_type: web` call log on connect, so you get a transcript and a recording without spending call minutes.
{% endhint %}

## Where next

* [Prerequisites](../quickstart/prerequisites.md)
* [Create your first agent](../quickstart/first-agent.md)
* [Running a campaign](../operator/running-a-campaign.md)
