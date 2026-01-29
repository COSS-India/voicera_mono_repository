# API Endpoints Reference

Quick reference for all VoiceERA API endpoints.

## Authentication

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/auth/register` | - | Register new user |
| POST | `/auth/login` | - | User login |
| POST | `/auth/refresh-token` | ✓ | Refresh JWT token |
| GET | `/auth/me` | ✓ | Get current user info |
| POST | `/auth/logout` | ✓ | User logout |
| POST | `/auth/forgot-password` | - | Request password reset |
| POST | `/auth/reset-password` | - | Reset password with token |

## Users

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/users` | ✓ | List all users (admin) |
| GET | `/users/{user_id}` | ✓ | Get user details |
| PUT | `/users/{user_id}` | ✓ | Update user |
| DELETE | `/users/{user_id}` | ✓ | Delete user |
| GET | `/users/{user_id}/agents` | ✓ | Get user's agents |
| GET | `/users/{user_id}/campaigns` | ✓ | Get user's campaigns |

## Agents

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/agents` | ✓ | List agents |
| POST | `/agents` | ✓ | Create agent |
| GET | `/agents/{agent_id}` | ✓ | Get agent details |
| PUT | `/agents/{agent_id}` | ✓ | Update agent |
| DELETE | `/agents/{agent_id}` | ✓ | Delete agent |
| GET | `/agents/{agent_id}/config` | ✓ | Get agent configuration |
| POST | `/agents/{agent_id}/clone` | ✓ | Clone agent |
| GET | `/agents/{agent_id}/campaigns` | ✓ | Get agent's campaigns |

## Campaigns

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/campaigns` | ✓ | List campaigns |
| POST | `/campaigns` | ✓ | Create campaign |
| GET | `/campaigns/{campaign_id}` | ✓ | Get campaign details |
| PUT | `/campaigns/{campaign_id}` | ✓ | Update campaign |
| DELETE | `/campaigns/{campaign_id}` | ✓ | Delete campaign |
| POST | `/campaigns/{campaign_id}/launch` | ✓ | Launch campaign |
| POST | `/campaigns/{campaign_id}/pause` | ✓ | Pause campaign |
| POST | `/campaigns/{campaign_id}/resume` | ✓ | Resume campaign |
| POST | `/campaigns/{campaign_id}/stop` | ✓ | Stop campaign |
| GET | `/campaigns/{campaign_id}/stats` | ✓ | Campaign statistics |

## Call Logs

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/call-logs` | ✓ | List call logs |
| GET | `/call-logs/{call_id}` | ✓ | Get call details |
| DELETE | `/call-logs/{call_id}` | ✓ | Delete call log |
| GET | `/call-logs/campaign/{campaign_id}` | ✓ | Get campaign's calls |
| GET | `/call-logs/agent/{agent_id}` | ✓ | Get agent's calls |
| GET | `/call-logs/date/{date}` | ✓ | Get calls by date |
| PUT | `/call-logs/{call_id}/notes` | ✓ | Add notes to call |
| GET | `/call-logs/{call_id}/transcript` | ✓ | Get call transcript |

## Recordings

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/call-recordings` | ✓ | List recordings |
| GET | `/call-recordings/{call_id}` | ✓ | Get recording metadata |
| GET | `/call-recordings/{call_id}/download` | ✓ | Download recording |
| DELETE | `/call-recordings/{call_id}` | ✓ | Delete recording |
| GET | `/call-recordings/{call_id}/transcript` | ✓ | Get transcript |
| POST | `/call-recordings/{call_id}/transcribe` | ✓ | Request transcription |
| GET | `/call-recordings/{call_id}/transcript/download` | ✓ | Download transcript |

## Analytics

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/analytics/calls` | ✓ | Call statistics |
| GET | `/analytics/sentiment` | ✓ | Sentiment analysis |
| GET | `/analytics/emotions` | ✓ | Emotion distribution |
| GET | `/analytics/top-phrases` | ✓ | Most common phrases |
| GET | `/analytics/agent-performance` | ✓ | Agent metrics |
| GET | `/analytics/campaign-stats` | ✓ | Campaign statistics |
| GET | `/analytics/hourly-calls` | ✓ | Calls by hour |
| GET | `/analytics/daily-calls` | ✓ | Calls by day |
| GET | `/analytics/export` | ✓ | Export analytics to CSV |

## Integrations

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/integrations` | ✓ | List integrations |
| POST | `/integrations` | ✓ | Create integration |
| GET | `/integrations/{integration_id}` | ✓ | Get integration details |
| PUT | `/integrations/{integration_id}` | ✓ | Update integration |
| DELETE | `/integrations/{integration_id}` | ✓ | Delete integration |
| POST | `/integrations/{integration_id}/test` | ✓ | Test integration |
| GET | `/integrations/{integration_id}/status` | ✓ | Get status |

## Webhooks

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/webhooks` | ✓ | List webhooks |
| POST | `/webhooks` | ✓ | Create webhook |
| GET | `/webhooks/{webhook_id}` | ✓ | Get webhook |
| PUT | `/webhooks/{webhook_id}` | ✓ | Update webhook |
| DELETE | `/webhooks/{webhook_id}` | ✓ | Delete webhook |
| POST | `/webhooks/{webhook_id}/test` | ✓ | Send test event |
| GET | `/webhooks/{webhook_id}/logs` | ✓ | Get webhook logs |

## Admin

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/admin/users` | ✓ admin | List all users |
| POST | `/admin/users/{user_id}/activate` | ✓ admin | Activate user |
| POST | `/admin/users/{user_id}/deactivate` | ✓ admin | Deactivate user |
| GET | `/admin/system/stats` | ✓ admin | System statistics |
| GET | `/admin/system/health` | ✓ admin | Health status |
| POST | `/admin/system/maintenance` | ✓ admin | Enable maintenance mode |
| GET | `/admin/logs` | ✓ admin | System logs |

## Health & Status

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health` | - | Service health |
| GET | `/readiness` | - | Readiness probe |
| GET | `/liveness` | - | Liveness probe |
| GET | `/version` | - | API version |
| GET | `/docs` | - | Swagger UI |
| GET | `/redoc` | - | ReDoc UI |
| GET | `/openapi.json` | - | OpenAPI spec |

---

## Query Parameters

### Pagination

```
?skip=0        # Number of items to skip (default: 0)
?limit=10      # Number of items to return (default: 10, max: 100)
```

### Filtering

```
?status=active                    # Filter by status
?agent_id=uuid                    # Filter by agent
?campaign_id=uuid                 # Filter by campaign
?date_from=2024-01-01            # Start date
?date_to=2024-01-31              # End date
?phone_number=%2B1234567890      # Filter by phone (URL encoded)
?search=keyword                   # Search in name/description
```

### Sorting

```
?sort=created_at              # Sort field
?order=asc                    # asc or desc
?sort=-created_at             # Descending (shorthand)
?sort=name,created_at         # Multiple sorts
```

---

## Common Responses

### Success (200 OK)

```json
{
  "success": true,
  "data": {...},
  "timestamp": "2024-01-29T10:30:00Z"
}
```

### Created (201 Created)

```json
{
  "success": true,
  "data": {...},
  "message": "Resource created successfully",
  "timestamp": "2024-01-29T10:30:00Z"
}
```

### No Content (204)

```
Empty response
```

### Error (4xx/5xx)

```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": "Additional error details"
  },
  "timestamp": "2024-01-29T10:30:00Z"
}
```

---

## Authentication Headers

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Content-Type: application/json
Accept: application/json
```

---

## Rate Limiting

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1674007200
```

---

## Next Steps

- **[REST API Details](rest-api.md)** - Complete REST API documentation
- **[WebSocket API](websocket-api.md)** - Real-time API
- **[Quick Start](../getting-started/quickstart.md)** - Get started
