---
description: The Next.js dashboard — Beta, on a separate branch.
---

# Dashboard (Beta)

A web dashboard for Voicera exists and is usable, with two caveats that apply to every page in this section.

{% hint style="warning" %}
The dashboard lives on the **`dev-frontend`** branch, is **not merged** into `dev`, and is **not** part of `docker-compose.yaml`. You run it yourself, against a running API. Treat it as a preview, not a supported surface.
{% endhint %}

## The pages

| Page | Covers |
| --- | --- |
| [Overview](overview.md) | What it is, its stack, and which API surfaces it consumes. |
| [Running the dashboard](running.md) | Cloning the branch, installing, and pointing it at your stack. |
| [Agent creation wizard](agent-wizard.md) | The guided flow from provider keys to a working agent. |
| [Dashboard tour](dashboard-tour.md) | Every route: agents, numbers, batches, history, knowledge, members, analytics. |
| [Browser test calls](test-calls.md) | Talking to an agent from the browser, and how the audio path works. |

## Why it is documented at all

Everything the dashboard does goes through the public REST API. It is the most complete worked example of a Voicera client — useful reading even if you build your own console.

## Related

* [Connecting a client](../clients/README.md) — the surfaces the dashboard uses
* [REST API](../../api-reference/overview.md)
