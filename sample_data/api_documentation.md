# API Documentation — Nimbus Notify API v1

## Base URL
```
https://api.nimbusnotify.com/v1
```

## Authentication
All requests must include an API key in the header:
```
Authorization: Bearer <YOUR_API_KEY>
```
API keys can be generated from the Developer Dashboard under **Settings → API Keys**.
Keys are scoped per project and can be revoked at any time.

## Rate Limits
- Free tier: 100 requests/minute, 10,000 requests/day.
- Pro tier: 1,000 requests/minute, 500,000 requests/day.
- Exceeding the limit returns HTTP 429 with a `Retry-After` header (in seconds).

## Endpoints

### 1. Send Notification
`POST /notifications`

Request body:
```json
{
  "to": "user_id_or_device_token",
  "title": "string",
  "body": "string",
  "channel": "push | sms | email",
  "priority": "normal | high"
}
```
Response: `202 Accepted` with a `notification_id`.

### 2. Get Notification Status
`GET /notifications/{notification_id}`

Returns delivery status: `queued`, `sent`, `delivered`, `failed`.

### 3. Register Device
`POST /devices`

Request body:
```json
{
  "user_id": "string",
  "device_token": "string",
  "platform": "ios | android | web"
}
```

### 4. List Devices for a User
`GET /devices?user_id={user_id}`

Returns an array of registered devices with their platform and registration date.

## Webhooks
Configure a webhook URL in the dashboard to receive delivery status updates. Webhook payloads
include `notification_id`, `status`, and `timestamp`. Webhook requests are signed with an
`X-Nimbus-Signature` header (HMAC-SHA256) which should be verified before processing.

## Error Codes
| Code | Meaning                          |
|------|-----------------------------------|
| 400  | Invalid request body              |
| 401  | Missing or invalid API key        |
| 404  | Resource not found                |
| 429  | Rate limit exceeded               |
| 500  | Internal server error             |

## SDKs
Official SDKs are available for Python, Node.js, and Java. Community-maintained SDKs exist for
Go and Ruby. All SDKs support automatic retry with exponential backoff on 429 and 500 errors.
