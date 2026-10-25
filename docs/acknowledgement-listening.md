# External acknowledgement listening

The voice hello advertises `acknowledgementListening: true`. An authenticated external
backend may send `listen_for_acknowledgement` with `deviceId`, unique `requestId` and
integer `durationMs` from 1 through 30000. Only connected, unmuted, idle devices with a
registered microphone callback can accept it. The external voice backend must be selected.

`acknowledgement_listen_result` returns `started` after microphone-start dispatch, or
`rejected`. A window uses the existing beam-locked turn microphone and external voice
pipeline, with no HA routing or wake-word audio trimming. Its `turn_start` messages carry
`acknowledgementRequestId`, no wake word, and the normal PCM format and session ID.

The backend must accept only acknowledgement words for these tagged turns. It answers
with `turn_acknowledgement`, matching device/session/request IDs, and boolean `acknowledged`.
False consumes unrelated speech without playback and starts another capture within the
original deadline. True finishes the window. Silence also starts another capture; it never
extends the deadline. A normal turn cannot use this terminal response.

`stop_acknowledgement_listen` cancels only a matching window; stale stop IDs are harmless.
Mute, disconnect, backend release and timeout cancel the window and clean up active turns.
Cleanup restores the ordinary wake stream, respecting hardware mute. Window closure emits
`acknowledgement_listen_finished` with the matching device/request IDs and a reason.
Standalone announcements are rejected until the listening window has released the device.

Both the controller and Assister need these changes. Protocol and callback tests simulate
capture and cancellation; physical microphone/speaker behavior still needs live validation.
