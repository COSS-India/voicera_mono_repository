---
description: Pointing Voicera at your own model server instead of cloud providers.
---

# Self-hosted models

Run speech and language models on your own GPUs so audio and text never leave your network. This page covers deploying the model server and wiring it to the runtime.

{% hint style="warning" %}
**This path is not verified end to end.** `model-server/README.md` states plainly that a real call through the voice server pointed at the gateway has **not been tested**, and that the LLM slot "has never been built or started". STT and TTS have been verified standalone on an H200; the integration has not. Budget time for debugging, and do not put this in front of callers without testing it yourself.
{% endhint %}

## When to self-host

| Reason | Detail |
| --- | --- |
| **Data residency** | Call audio and transcripts never reach a third-party API |
| **Cost at volume** | Fixed hardware cost instead of per-minute billing |
| **Language coverage** | Indic models that cloud vendors serve poorly or not at all |
| **Offline operation** | No dependency on external availability |

Against: you need GPUs, the images are large, and cold starts are slow. Mixing is common — self-host speech, use a cloud LLM, or the reverse.

## Deploy the model server

```bash
cd model-server
STT_MODEL=indic-conformer TTS_MODEL=indic-parler ./setup.sh
```

The gateway comes up on `:8100`; the model slots stay internal on `8001`, `8002`, `8003`. Check it:

```bash
curl -s localhost:8100/health
curl -s localhost:8100/v1/models
```

Full detail in [Model server overview](../../developer/model-server/overview.md) and [Slots and models](../../developer/model-server/slots-and-models.md).

{% hint style="warning" %}
Weights are **not** in the repository. `stt/indic-conformer/models/IndicConformer.nemo` and `tts/indic-parler/checkpoints/` are gitignored, and `ai4bharat/indic-parler-tts` is a **gated** HuggingFace repo — you need a token with access, or a pre-populated cache. Build one image at a time on a tight disk; parallel builds double peak usage at the export stage, which is where they fail.
{% endhint %}

## Network it to the runtime

The model containers publish nothing on the host — only the gateway does — so the stack coexists with others without port conflicts.

{% tabs %}
{% tab title="Same host" %}
Put both stacks on one network, or reach the gateway over the host address:

```bash
MODEL_SERVER_URL=http://host.docker.internal:8100   # Docker Desktop
MODEL_SERVER_URL=http://172.17.0.1:8100             # Linux bridge
```
{% endtab %}

{% tab title="Separate hosts" %}
```bash
MODEL_SERVER_URL=https://models.internal.example.com
```

Keep it on a private network. The gateway has **no authentication** — anything that can reach `:8100` can use your GPUs.
{% endtab %}
{% endtabs %}

## Configure an agent

The gateway is OpenAI-compatible, so agents point at it through providers that accept a `base_url`:

```json
{
  "models": {
    "llm_config": {
      "provider": "openai",
      "model": "qwen3.5-4b",
      "base_url": "http://models.internal:8100/v1"
    }
  }
}
```

The model id must match what `GET /models` reports. Mixing is fine — a self-hosted LLM with cloud STT and TTS is a valid configuration.

{% hint style="info" %}
`apps/providers/local/` is reserved for first-class self-hosted providers but is currently an empty package. Until it is implemented, route through an OpenAI-compatible `base_url` as above.
{% endhint %}

## Verify

Work outward, one layer at a time.

**1. The gateway answers:**

```bash
curl -s localhost:8100/v1/models
```

**2. Each modality works standalone:**

```bash
curl -s -X POST localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Testing one two three", "voice": "default"}' \
  -o out.wav

curl -s -X POST localhost:8100/v1/audio/transcriptions \
  -F file=@out.wav
```

A round trip — TTS speaks a sentence and STT transcribes it back — is the check the maintainers used.

**3. Then a call.** This is the unverified step. Watch the runtime logs closely:

```bash
docker compose logs -f runtime
```

## Known gaps

| Gap | Status |
| --- | --- |
| A real call through the runtime to the gateway | **Not tested** |
| The LLM slot | **Never built or started**; its vLLM flags are unverified |
| `apps/providers/local/` | Reserved, empty |
| Model catalog sharing | `models.yaml` is the model server's own catalogue; the platform's provider catalog is separate |

The per-model pages under [Model server](../../developer/model-server/overview.md) state what has and has not run on hardware. Read them before choosing a model — `ready` in `models.yaml` means "the folder exists with a Dockerfile", not "tested".

## Related

* [Model server overview](../../developer/model-server/overview.md)
* [Gateway API](../../developer/model-server/gateway-api.md)
* [Running on GPUs](../../developer/model-server/gpu-operations.md)
* [Provider registry](../concepts/provider-registry.md)
