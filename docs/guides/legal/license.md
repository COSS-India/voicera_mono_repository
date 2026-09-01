---
description: Voicera is distributed under the MIT License.
---

# License

Voicera is open source under the **MIT License**, Copyright (c) 2026 COSS India. The authoritative text is the [`LICENSE`](https://github.com/COSS-India/voicera/blob/main/LICENSE) file at the repository root.

## What the MIT License permits

You may use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software — including commercially and in closed-source products — provided that:

* The copyright notice and permission notice appear in all copies or substantial portions of the software.
* You accept that the software is provided "as is", without warranty of any kind.

There is no copyleft obligation. You do not have to open source work that builds on Voicera.

{% hint style="info" %}
Do not describe Voicera as proprietary or "all rights reserved" anywhere. The repository licence is MIT only.
{% endhint %}

## What the licence does not cover

The MIT grant applies to **this repository's source code**. It does not extend to:

| Thing | Governed by |
| --- | --- |
| Models you download and run | Each model's own licence — several are gated and carry their own terms. See [Model server](../../developer/model-server/overview.md). |
| Cloud AI provider usage | Your contract with that vendor. |
| Telephony provider usage | Your contract with Vobiz, Plivo, or another carrier. |
| Container base images and Python dependencies | Their own licences. |
| Call recordings, transcripts, and contact data | You. Voicera is self-hosted, so this data is yours and your obligation. |

{% hint style="warning" %}
Model weights are the most common surprise. A permissive platform licence does not make a restrictively licensed model redistributable. Check each model's terms before deploying it — the per-model pages under [Model server](../../developer/model-server/overview.md) note where weights are gated.
{% endhint %}

## Contributions

Contributions are accepted under the same MIT terms. By opening a pull request you agree your contribution may be distributed under the project licence. See [Contributing](../../developer/guides/contributing.md).

## Related

* [Code of conduct](code-of-conduct.md)
* [Security policy](security.md)
* [Contributing](../../developer/guides/contributing.md)
