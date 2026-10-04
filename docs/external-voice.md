# External Voice Backend

EchoMuse can send accepted voice turns to one external application over
`WS /api/voice` on the existing controller HTTP API port. ESPHome/Home Assistant
Assist remains the default. This feature changes the voice destination only;
the ESPHome facade remains running, including Home Assistant media and entities.
Explicit HA announce-then-listen/`ask_question` requests continue in their own HA
pipeline; selecting external changes wake/button turns, never reroutes a partially
captured turn. This is not an HA fallback for an unavailable external backend.

EchoMuse owns device connectivity, wake models and sensitivity, audio processing,
microphone routing, local speech gating/endpointing, wake arbitration, wake cues,
barge-in, device state/LEDs, speaker playback, volume/mute/EQ, firmware and
configuration. The external application owns STT, intents, reasoning, tools,
conversation state, response text and TTS generation. Only the winning device's
accepted turn is forwarded; the external application does not arbitrate wakes.

## Enable and authenticate

Use the existing admin API:

```http
PATCH /api/system/config
Authorization: Bearer <EchoMuse session token>
Content-Type: application/json

{"voiceBackend":"external"}
```

`GET /api/system/config` reports `voiceBackend`. Its default is `esphome`;
`external` is the only alternative. The setting is persisted in the existing
SQLite system-config table, applies controller-wide to new turns, and survives
restart. An already accepted turn retains its backend through completion and
cancellation. Set it back to `esphome` to use HA Assist for new turns.

Obtain a normal admin session through `POST /api/auth/login`. Authenticate the
WebSocket upgrade with `Authorization: Bearer <token>`. Existing session-cookie
and `?token=` authentication remain supported; a header avoids credentials in
URLs. Unauthenticated upgrades return HTTP 401; read-only sessions return 403;
a second backend connection returns 409. The first connection is never replaced.
Use HTTPS/WSS when the API is exposed beyond a trusted network. The existing
ingress-only restriction also applies; this endpoint does not bypass it.

For example, a .NET client can authenticate before connecting:

```csharp
using var socket = new System.Net.WebSockets.ClientWebSocket();
socket.Options.SetRequestHeader("Authorization", $"Bearer {token}");
await socket.ConnectAsync(new Uri("ws://controller:8768/api/voice"), cancellationToken);
```

The connection receives:

```json
{"type":"hello","protocolVersion":1,"controllerVersion":"...","voiceBackend":"external"}
```

Protocol version 1 uses JSON text messages only. The backend is ready after
`hello`; no separate negotiation or fleet subscription is needed. It can connect
while `esphome` is selected, but voice turns and external announcements require
external mode. The socket serves the whole fleet.

## Accepted voice turns

Every turn gets a new opaque UUID. Every session message carries both `sessionId`
and `deviceId`; responses must echo both exactly. Device configuration is not
included. Optional wake fields are sent only when EchoMuse actually knows them.

Controller to backend, in order:

```json
{"type":"turn_start","sessionId":"...","deviceId":"...","trigger":"wakeword","wakeWord":"hey_jarvis","wakeScore":0.74,"audio":{"sampleRate":16000,"sampleWidth":2,"channels":1,"encoding":"pcm_s16le"}}
{"type":"audio","sessionId":"...","deviceId":"...","data":"<base64 PCM>"}
{"type":"audio_end","sessionId":"...","deviceId":"...","reason":"speech_end"}
```

`wakeWord` is the detected model's identifier, not a fabricated phrase. When
available, `wakeThreshold` and `noiseFloor` accompany it. `trigger` may also be
`button` or `barge-in`; button turns have no wake metadata. Audio is 16 kHz,
mono, signed 16-bit little-endian PCM, after local routing, speech gating and
optional ASR noise suppression. Frame boundaries carry no semantic meaning.
`audio_end` ends input; no subsequent microphone audio belongs to that session.
`no_speech_timeout` ends a silent turn without waiting for a response.

EchoMuse uses its existing device end sentinel and local fallback endpoint clock;
external mode does not depend on Home Assistant's VAD events. Listening ends
locally, then EchoMuse transitions to thinking. Send a response only after
`audio_end` with `reason: "speech_end"`:

```json
{"type":"turn_response","sessionId":"...","deviceId":"...","audioUrl":"http://backend/response.wav","text":"Optional response text"}
```

The URL must use HTTP(S), have a host, and contain no user-info credentials.
EchoMuse streams it through its existing URL decoder/resampler and voice
playback callback: EQ, bass guard, limiter, speaker buffering, completion and
music restoration stay local. Supported encodings are those already handled by
the existing WAV/ffmpeg path. The backend must keep the URL available for playback.
`text` is optional and is not interpreted or retained as conversation state.

Alternatively, end a turn with either:

```json
{"type":"turn_error","sessionId":"...","deviceId":"...","message":"STT unavailable"}
{"type":"turn_cancel","sessionId":"...","deviceId":"..."}
```

The controller reports `play_started`, `play_finished`, or `play_failed` using
the session's IDs. `play_started` means the playback path was entered; fetching
may still be in progress. `turn_finished` ends a normally processed session and
includes `outcome` (`ok`, `error`, or `no_speech_timeout`). A cancellation/failure
instead ends it with:

```json
{"type":"turn_cancel","sessionId":"...","deviceId":"...","reason":"barge_in"}
```

Other cancellation reasons include `cancelled` (button), `mute`, `disconnect`,
`timeout`, `backend_cancel` and `failure`. A barge cancels the old session before
a fresh accepted turn starts with a different ID. Unknown, completed, cancelled,
duplicate or wrong-device responses are rejected and never played. An external
cancel can affect only its explicitly identified active session.

## Announcements outside a turn

Use the same socket, without adding a management endpoint:

```json
{"type":"play","requestId":"notification-1","deviceId":"...","audioUrl":"http://backend/timer.wav","kind":"announcement"}
{"type":"stop","requestId":"notification-1","deviceId":"..."}
```

`requestId` identifies an active announcement on that device. The controller
returns `play_started`, `play_finished` or `play_failed` with `requestId` and
`deviceId`. Stop cancels only that exact request; completion then includes
`reason: "cancelled"`. Offline devices, busy devices and unsupported kinds are
rejected. Announcements use the existing standalone playback callback and music
preemption/restoration. They are bounded buffered announcements, not a music
library. A button or mute also cancels them. An announcement reserves the device's
voice lock; a wake cannot claim that busy device until its announcement ends.

Invalid commands receive a bounded `protocol_error`. Error responses never echo
the supplied JSON, audio payload or URL. The backend should retain its own request
correlation; a rejected command does not change another session or announcement.

## Failure and limits

External mode never falls back to HA. With no connected ready backend, readiness
fails **before arbitration** and the existing unavailable/dropped-turn cue and
diagnostic path runs. Its historical `no_ha` outcome/cue is reused. A disconnect
cancels all owned turns and announcements, flushes cancelled playback, restores
local turn state, and releases the socket slot after cleanup. Partial audio is
never rerouted. An HTTP/playback failure ends the session cleanly.

Limits in protocol version 1:

| Resource | Limit |
| --- | --- |
| Incoming WebSocket message | 16 KiB |
| Local PCM chunk | 64 KiB, even byte count |
| Active turns plus announcements | 32, at most one per device |
| Session/device/request/type identifier | 128 characters |
| Audio URL | 2048 characters |
| Optional response text | 4096 characters |
| Stored/logged backend error | 512 characters |
| Announcement decoded PCM | 12 MiB |
| Microphone input | 20 seconds |
| Whole turn/announcement | 120 seconds |
| Socket send including lock wait | 2 seconds |

Sends are awaited, serialised and time bounded; there is no external PCM backlog
queue. A slow connection fails its turn rather than stalling the controller loop
or accumulating unlimited audio. Existing bounded device queues remain in use.
The socket has a 30-second heartbeat. Oversized wire messages close the socket.

## Existing management APIs

Continue using `/api/devices`, `/api/devices/{id}`, the existing device and fleet
config endpoints, wake-model endpoints, volume/mute controls, system status,
logs and turn diagnostics. Those APIs remain authoritative. This protocol has
no device inventory, wake-model management, configuration writes, remote LED
control, STT engine, TTS engine or tool protocol.

## Validation

See [external voice validation](external-voice-validation.md) for the recorded
baseline and physical validation checklist.
