---
description: Add an STT, TTS, or LLM vendor in six steps.
---

# Adding an AI provider

How to add a speech-to-text, text-to-speech, or large-language-model vendor to `apps/providers`. Adding a provider means creating one folder and registering one creator function per capability. You never edit a central dispatch table.

{% hint style="info" %}
This page is the how-to. For *why* the registry works this way — discriminated unions, the catalog dump, and where credentials live — read [Provider registry](../../guides/concepts/provider-registry.md).
{% endhint %}

## Before you start

Read `apps/providers/readme.md` and open one existing vendor folder side by side with your editor. `apps/providers/cloud/deepgram/` is the clearest STT example, `cloud/cartesia/` the clearest TTS one, and `cloud/openai/` shows one folder serving all three kinds.

Check two things first:

* **Does Pipecat already support the vendor?** If `pipecat.services.<vendor>` exists, your `service.py` is a handful of lines. If not, you are writing an adapter — see [Cloud, adapter, or local](#cloud-adapter-or-local).
* **What are the credentials?** Everything the vendor authenticates with goes on an `Auth` class and is marked `secret`. Everything else — voices, endpoints, speeds — goes on `Settings`.

## Cloud, adapter, or local

`provider_type` is not something you declare. `apps/providers/schema.py` derives it from the config class's module path, so the directory you choose *is* the decision:

| Directory | `provider_type` | Use when |
| --- | --- | --- |
| `cloud/<vendor>/` | `cloud` | Pipecat already ships a service class for the vendor and you are configuring it. 21 vendors live here. |
| `adapters/<vendor>/` | `adapter` | You are writing the Pipecat `STTService` / `TTSService` subclass yourself. `adapters/bhashini/` is the one example; its `tts.py` holds the NVCF gRPC client. |
| `local/<vendor>/` | `local` | Reserved for self-hosted providers. Currently empty apart from `__init__.py`. |

Put the folder in the wrong place and `_provider_type()` raises with a message telling you exactly that.

## The six steps

These are the steps from `apps/providers/readme.md`, which is the authoritative version.

1. Add `cloud/<name>/`, `adapters/<name>/`, or `local/<name>/` with `catalog.py`, `config.py`, `languages.py`, and `service.py`.
2. Put credentials on Auth, knobs on Settings, and on Config set `name: str = "Display Name"` (the UI label) plus `provider: Literal["…"] = "…"`.
3. On `language` fields, use `json_schema_extra=language_schema_extra(SUPPORTED_LANGUAGES["STT"|"TTS"])`.
4. For selectable fields, set `examples` and optionally `allow_custom_input=True` (the catalog becomes `input_mode` `both` rather than `options`).
5. In `service.py`, implement `@register_stt` / `@register_tts` / `@register_llm` creators that take the typed config and build the Pipecat (or adapter) service. Import Pipecat **inside** the creator so missing extras do not break package import.
6. Do **not** edit a central `if`/`elif` in `factory.py` — registration plus `load_providers()` pick up the new module. The schema dump follows automatically.

Here is what happens to your folder once it exists:

```mermaid
flowchart LR
  SVC["service.py<br/>@register_stt"] --> REG["registry.py<br/>STT_CONFIGS · STT_CREATORS"]
  DISC["load_providers()<br/>walks cloud · adapters · local"] --> SVC
  REG --> FAC["factory.py<br/>discriminated union<br/>+ create_stt_service"]
  REG --> SCH["schema.py<br/>provider_schemas()"]
  SCH --> CFG["API<br/>GET /configuration/stt"]
  FAC --> RT["Runtime<br/>builds the pipeline"]
  CFG --> AG["Agent config<br/>stt_config.provider"]
  AG --> RT
```

`load_providers()` imports every `*.service` module under `cloud`, `adapters`, and `local`, which runs your decorator, which fills the registry maps, which both the factory unions and the catalog dump read from. One registration reaches all of it.

## The four module files

| File | Holds | Imported by |
| --- | --- | --- |
| `catalog.py` | Plain tuples and defaults: `STT_MODELS`, `TTS_VOICES`, `DEFAULT_STT_MODEL`. No Pydantic, no vendor SDK. | `config.py` |
| `languages.py` | `SUPPORTED_LANGUAGES`, a `{"STT"\|"TTS": {model_id: {vendor_code: canonical_id}}}` map. | `config.py` |
| `config.py` | The Auth, Settings, and Config classes. Pure Pydantic — importable without the vendor SDK or Pipecat. | `service.py`, `registry.py` |
| `service.py` | The `@register_*` creator functions. The only file that touches Pipecat. | `load_providers()` |

Keeping `catalog.py` and `languages.py` as flat data means the model list and the language matrix stay reviewable in a diff, and the API's `/configuration/*` catalog is generated from them rather than maintained twice.

## Auth versus Settings

Each vendor `config.py` stacks three layers, in this inheritance order:

```python
class DeepgramAuth(BaseModel):
    api_key: str | list[str] = Field(
        description="Deepgram API key (or a list for rotation).",
        json_schema_extra={"secret": True},
    )


class DeepgramSTTSettings(BaseModel):
    base_url: str | None = Field(
        default=None,
        description="Override the Deepgram API base URL.",
    )


class DeepgramSTTConfig(DeepgramAuth, DeepgramSTTSettings, BaseSTTConfig):
    name: str = "Deepgram"
    provider: Literal["deepgram"] = "deepgram"
```

The split is load-bearing, not cosmetic:

| Layer | Contains | Where it ends up |
| --- | --- | --- |
| **Auth** | Credentials and account identity. Every field carries `json_schema_extra={"secret": True}`. | Encrypted into `ProviderAuth` with `PROVIDER_AUTH_ENCRYPTION_KEY`. Never on the agent document. |
| **Settings** | Vendor knobs — `voice`, `speed`, `volume`, `base_url`, `grpc_url`. | Stored in plain text on the agent's `config.models`. |
| **Config** | Auth + Settings + a `Base*Config` supplying `kind`, `name`, `provider`, `model`, and `language`. | The class the creator is typed against. |

Two rules follow from it, and the tests enforce both:

* **Credentials never live on the bases.** `BaseSTTConfig` and friends in `base.py` carry no `api_key`.
* **Endpoints and hosts belong on Settings, not Auth.** `base_url` is configuration; a token is a secret. `test_provider_auth_secrets_only_plus_auth_mro` asserts the auth dump contains only the secret fields.

Multiple secret fields are fine. `aws_bedrock` declares both `aws_access_key` and `aws_secret_key`, and `test_aws_bedrock_lists_credential_secrets` pins that pair.

Credentials are **provider-level**, not per-kind. One OpenAI key covers OpenAI STT, TTS, and LLM, because `provider_level_auth("openai")` merges the auth fields across the kinds sharing the provider id.

## Languages

Canonical language ids live once in `apps/providers/languages.py` — `hi`, `en`, `en-US`, `multi`, and the rest. Your `languages.py` maps *vendor* codes to those canonical ids, per model:

```python
SUPPORTED_LANGUAGES = {
    "STT": {
        "nova-3-general": {"multi": "multi", "en": "en", "hi": "hi", "ta": "ta"},
        "flux-general-en": {"en": "en"},
    },
}
```

Then wire the map into the `language` field:

```python
language: str = Field(
    default="multi",
    json_schema_extra=language_schema_extra(SUPPORTED_LANGUAGES.get("STT", {})),
)
```

`language_schema_extra()` inverts and flattens the map into three keys the API serves:

| Key | Shape | Used for |
| --- | --- | --- |
| `examples` | Flat sorted list of canonical ids | The full language list for this provider. |
| `model_options` | `model → [canonical ids]` | Filtering the language picker once a model is chosen. |
| `language_codes` | `model → {canonical: vendor_code}` | Translating back to the vendor's own code on the wire. |

The inversion keeps the first vendor code that maps to a canonical id. ElevenLabs sends `or` for Odia and `auto` for auto-detect, so `language_codes["scribe_v2_realtime"]["od"] == "or"` and `["multi"] == "auto"` — the storage layer never sees the vendor spelling.

Vendor **defaults may differ on purpose**. Deepgram STT defaults to `multi`, Deepgram TTS to `en`, Bhashini to `hi`. Do not force one default across providers.

{% hint style="warning" %}
Every canonical id you emit must already exist in `LANGUAGES` in `apps/providers/languages.py`. `test_every_stt_tts_schema_with_language_has_structured_extras` walks every provider's `examples` and fails on an id that is not there. If your vendor supports a language Voicera has no canonical id for, add it to `LANGUAGES` in the same change.
{% endhint %}

## Controlling input_mode

`input_mode` tells the API consumer whether a field is a dropdown, a text box, or both. You do not set it. `schema.py` derives it:

| Field has | `allow_custom_input` | `input_mode` |
| --- | --- | --- |
| No `examples` and no `model_options` | — | `input` |
| `examples` or `model_options` | absent or `False` | `options` |
| `examples` or `model_options` | `True` | `both` |

So a free-text override is a field with no `examples`:

```python
base_url: str | None = Field(default=None, description="Override the API base URL.")
# → input_mode "input"
```

A closed list is `examples` alone — Deepgram TTS is English-only, so its `language` uses `allow_custom_input=False` and comes out `options`. An open list with suggestions sets `allow_custom_input=True` and comes out `both`, which is what you want for model ids that the vendor adds to faster than you can ship a release.

Secret fields get **no** `input_mode` at all — `test_secrets_have_no_input_mode` asserts it, and `test_non_secret_fields_have_input_mode` asserts every non-secret field has one. `allow_custom_input` itself is read for the derivation and then dropped; `test_catalog_omits_schema_noise` fails if it leaks into the dump.

## Registering creators

Registration is by type annotation, not by string. `registry._register` reads the creator's first parameter, resolves the config class from it, and takes the provider id from that class's `provider` field default:

```python
from ...registry import register_stt, register_tts, api_key
from .config import DeepgramSTTConfig, DeepgramTTSConfig


@register_stt
def create_stt(cfg: DeepgramSTTConfig):
    from pipecat.services.deepgram.stt import DeepgramSTTService, DeepgramSTTSettings

    kwargs: dict[str, Any] = {}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return DeepgramSTTService(
        api_key=api_key(cfg.api_key),
        settings=DeepgramSTTSettings(model=cfg.model, language=cfg.language),
        **kwargs,
    )
```

Four things in that snippet are conventions worth copying:

* **The Pipecat import is inside the function.** `apps/api` imports `apps.providers` without Pipecat installed. A module-level Pipecat import would break the API for every provider, not just yours.
* **The parameter is annotated with a concrete config class.** An unannotated or non-Pydantic first parameter raises a `TypeError` at import.
* **`api_key(cfg.api_key)` resolves a rotation list.** When a vendor's key field is `str | list[str]`, this helper returns the first entry. `registry.llm_settings(cfg)` does the equivalent for LLM sampling knobs, emitting only the fields that are actually set.
* **Optional overrides go through `kwargs`.** Passing `base_url=None` to a Pipecat service is not the same as omitting it.

Registering the same provider id twice for one kind raises `ValueError` at import — that is the collision check, and it fires before anything can silently shadow an existing vendor.

`adapters/bhashini/service.py` looks identical, except the deferred import points at its own `tts.py` instead of Pipecat.

## Why you never edit factory.py

`factory.py` builds its discriminated unions *from the registry*:

```python
def _union_type(kind: Kind):
    classes = config_classes(kind)
    return Annotated[Union[classes], Field(discriminator="provider")]

load_providers()

STTConfig = _union_type(Kind.STT)
```

`create_stt_service` then dispatches with `get_creator(Kind.STT, cfg.provider)(cfg)`. There is no branch on provider name anywhere in the file. Adding a vendor changes the union's membership and the creator map as a side effect of the decorator running, so:

* `AgentConfig.model_validate({...})` accepts your provider id without a schema change.
* `GET /configuration/stt` lists it without a router change.
* `GET /auth/catalog` exposes its secret fields without a router change.
* The runtime builds it without an `ai_service_factory` change.

The same holds for `schema.py`. If you find yourself adding a provider name to a list outside your own folder, you have gone off the path.

## Testing

`apps/providers/tests/test_provider_schemas.py` is a single module that tests the registry as a whole rather than each vendor, so most of it covers your provider automatically:

```bash
export PYTHONPATH="$PWD"
pytest apps/providers/tests -v
```

These generic checks will start applying to your folder the moment it is discovered:

| Check | What it catches |
| --- | --- |
| `test_every_registered_config_has_creator` | A config class registered without a creator, or the reverse. |
| `test_catalog_omits_schema_noise` | `$defs`, `$ref`, `anyOf`, or a leaked `allow_custom_input` in the dump — usually a nested `BaseModel` where a flat field belonged. |
| `test_provider_type_from_package_path` | A folder placed outside `cloud/`, `adapters/`, or `local/`. |
| `test_secrets_have_no_input_mode` / `test_non_secret_fields_have_input_mode` | A credential missing `secret: True`, or a knob that ended up marked secret. |
| `test_every_stt_tts_schema_with_language_has_structured_extras` | A `language` field wired without `language_schema_extra()`, or a canonical id absent from `LANGUAGES`. |
| `test_no_duplicate_provider_ids_within_kind` | Two folders claiming the same `provider` literal. |

One check needs a manual edit. `test_union_variant_counts_match_registry` asserts exact counts:

```python
assert len(_union_variants(STTConfig)) == 11
assert len(_union_variants(TTSConfig)) == 14
assert len(_union_variants(LLMConfig)) == 9
```

Bump the number for the kinds you added. That failure is the test doing its job — it is how an accidentally unregistered or double-registered provider shows up.

Add a vendor-specific test only where your provider does something the generic checks cannot see, such as a non-obvious vendor code inversion. `test_elevenlabs_stt_odia_vendor_code_in_schema` and `test_sarvam_stt_auto_detect_vendor_code_is_unknown` are the models to follow.

{% hint style="warning" %}
There is no CI. Run the suite yourself before opening a pull request. See [Testing](testing.md).
{% endhint %}

## A worked example

Adding a fictional `acme` STT vendor that Pipecat already supports.

**1. `apps/providers/cloud/acme/catalog.py`**

```python
"""Acme model catalog (STT)."""

STT_MODELS: tuple[str, ...] = ("acme-realtime-v2", "acme-batch-v2")

DEFAULT_STT_MODEL = "acme-realtime-v2"
```

**2. `apps/providers/cloud/acme/languages.py`**

```python
"""Acme: vendor language code -> canonical code (apps/providers/languages.py)."""

_FULL = {
    "en": "en",
    "hi": "hi",
    "ta": "ta",
    "auto": "multi",
}

SUPPORTED_LANGUAGES = {
    "STT": {
        "acme-realtime-v2": dict(_FULL),
        "acme-batch-v2": dict(_FULL),
    },
}
```

Note `"auto": "multi"` — Acme's wire code is `auto`, but Voicera stores `multi`.

**3. `apps/providers/cloud/acme/config.py`**

```python
"""Acme STT configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ...base import BaseSTTConfig
from ...languages import language_schema_extra
from .catalog import STT_MODELS, DEFAULT_STT_MODEL
from .languages import SUPPORTED_LANGUAGES


class AcmeAuth(BaseModel):
    api_key: str = Field(
        description="Acme API key.",
        json_schema_extra={"secret": True},
    )


class AcmeSTTSettings(BaseModel):
    base_url: str | None = Field(
        default=None,
        description="Override the Acme API base URL.",
    )


class AcmeSTTConfig(AcmeAuth, AcmeSTTSettings, BaseSTTConfig):
    """Acme speech-to-text configuration."""

    name: str = "Acme"

    provider: Literal["acme"] = "acme"
    model: str = Field(
        default=DEFAULT_STT_MODEL,
        description="Acme STT model.",
        json_schema_extra={
            "examples": list(STT_MODELS),
            "allow_custom_input": True,
        },
    )
    language: str = Field(
        default="multi",
        description="Canonical language id, or 'multi' for auto-detection.",
        json_schema_extra=language_schema_extra(
            SUPPORTED_LANGUAGES.get("STT", {}),
        ),
    )
```

**4. `apps/providers/cloud/acme/service.py`**

```python
"""Build Pipecat (or adapter) services from this vendor's configs."""

from __future__ import annotations

from typing import Any

from ...registry import register_stt
from .config import AcmeSTTConfig


@register_stt
def create_stt(cfg: AcmeSTTConfig):
    from pipecat.services.acme.stt import AcmeSTTService, AcmeSTTSettings

    kwargs: dict[str, Any] = {}
    if cfg.base_url:
        kwargs["base_url"] = cfg.base_url
    return AcmeSTTService(
        api_key=cfg.api_key,
        settings=AcmeSTTSettings(model=cfg.model, language=cfg.language),
        **kwargs,
    )
```

**5. An empty `apps/providers/cloud/acme/__init__.py`.**

**6. Bump the STT count** in `test_union_variant_counts_match_registry` from `11` to `12`, then run the suite:

```bash
pytest apps/providers/tests -v
```

**7. Confirm the catalog picked it up:**

```bash
python -c "
from apps.providers import Kind, provider_schemas
import json
print(json.dumps(provider_schemas(Kind.STT)['acme'], indent=2))
"
```

You should see `provider_type: "cloud"`, `secrets: ["api_key"]`, `model.input_mode: "both"`, and a `language` entry carrying `examples`, `model_options`, and `language_codes`. Nothing outside `apps/providers/cloud/acme/` changed except one integer in a test.

If your vendor also needs a Pipecat extra, add it to the extras list in `apps/runtime/requirements.txt` — that is the one file outside your folder a cloud provider legitimately touches.

## Related

* [Provider registry](../../guides/concepts/provider-registry.md)
* [Provider credentials (ProviderAuth)](../../guides/concepts/provider-auth.md)
* [Adding a telephony provider](adding-a-telephony-provider.md)
* [Providers service](../services/providers.md)
* [Testing](testing.md)
