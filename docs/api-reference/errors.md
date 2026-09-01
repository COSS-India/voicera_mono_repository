---
description: Status codes, error shapes, and what each one means.
---

# Errors

Every handled error is a FastAPI `HTTPException`, which serialises as:

```json
{ "detail": "Organisation not found: org_abc123" }
```

A validation failure is different. FastAPI returns `422` with a list:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "config", "language", "primary"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

`loc` walks from the request part inwards, so it tells you exactly which nested field failed. Agent config validation failures also arrive as `422`, but with a single string in `detail` rather than a list, because they are raised by hand.

## Status codes

| Code | Meaning | Common cause |
| --- | --- | --- |
| `400` | Bad request | No active organisation in the token; a websocket agent asked for `/answer`. |
| `401` | Unauthenticated | Missing, malformed, or expired token; wrong `X-API-Key`. |
| `403` | Wrong role or wrong organisation | Authenticated, but the role or membership does not permit it. |
| `404` | Not found | The id does not exist, or exists in another organisation. |
| `409` | Conflict | Duplicate signup email; a campaign state transition that is not allowed. |
| `422` | Validation failed | A field is missing, out of range, or the wrong type. |
| `500` | Server error | `INTERNAL_API_KEY` unset; an unhandled provider failure. |

{% hint style="info" %}
A `404` on a resource you know exists usually means your token's active organisation is not the one that owns it. Swap with `POST /api/v1/users/switch-organisation` and retry.
{% endhint %}

## Related

* [Authentication](authentication.md)
* [Troubleshooting](../guides/troubleshooting/README.md)
