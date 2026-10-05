"""One authenticated external voice connection, with explicitly owned turns.

Device capture, endpointing and playback remain local. No HA protocol or run
barrier is involved in these sessions. Sends are serialised and time bounded;
there is no unbounded audio/message queue.
"""

import asyncio
import base64
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import numpy as np
from aiohttp import web

import em_audio_stream
import em_speechgate
import em_turnclock

log = logging.getLogger("echomuse")
MAX_MESSAGE = 16 * 1024
MAX_CHUNK = 64 * 1024
MAX_SESSIONS = 32
MAX_URL = 2048
MAX_TEXT = 4096
MAX_ERROR = 512
MAX_PLAY_BYTES = 12 * 1024 * 1024
SEND_TIMEOUT = 2.0
INPUT_TIMEOUT = 20.0
TURN_TIMEOUT = 120.0


def _string(message, key, limit=128):
    value = message.get(key)
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ValueError(f"invalid {key}")
    return value


def _url(message):
    value = _string(message, "audioUrl", MAX_URL)
    parsed = urlsplit(value)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None):
        raise ValueError("audioUrl must be an HTTP(S) URL without credentials")
    return value


@dataclass
class Turn:
    device: object
    owner: object
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    response: object = None
    phase: str = "listening"
    reason: str = "cancelled"
    outcome: str = "error"
    audio_bytes: int = 0
    playback_ms: float = -1
    tts_bytes: int = 0
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    continue_conversation: bool = False

    def message(self, kind, **values):
        return dict(type=kind, sessionId=self.session_id,
                    deviceId=self.device.device_id, **values)


class ExternalVoiceBackend:
    def __init__(self):
        self.client = None
        self.ready = False
        self._send_lock = asyncio.Lock()
        self.turns = {}
        self.plays = {}
        self._play_callbacks = {}
        self.get_device = lambda device_id: None
        self.persist = None

    def register_device(self, device_id, play):
        self._play_callbacks[device_id] = play

    def device_gone(self, device_id):
        self.cancel_voice_turn(device_id, reason="disconnect")
        self._play_callbacks.pop(device_id, None)

    def claim(self, client):
        if self.client is not None:
            return False
        self.client = client
        log.info("External voice backend connected")
        return True

    async def release(self, client):
        if self.client is not client:
            return
        # Keep the slot reserved through cleanup: a replacement socket must
        # never inherit a previous socket's sessions or playback tasks.
        self.ready = False
        turns = tuple(self.turns.values())
        for turn in turns:
            if turn.owner is client:
                self.cancel_voice_turn(turn.device.device_id, reason="disconnect")
        tasks = tuple(self.plays.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(asyncio.wait_for(t.finished.wait(), 5.0) for t in turns),
                             return_exceptions=True)
        self.client = None
        log.info("External voice backend disconnected")

    def has_turn(self, device_id):
        return any(t.device.device_id == device_id for t in self.turns.values())

    def can_serve_turn(self, device_id):
        return (self.ready and self.client is not None and not self.client.closed
                and not any(target == device_id for _, target in self.plays)
                and len(self.turns) + len(self.plays) < MAX_SESSIONS)

    async def send(self, owner, message):
        async def write():
            async with self._send_lock:
                if self.client is not owner or owner.closed:
                    raise ConnectionError("external backend disconnected")
                await owner.send_json(message)
        try:
            await asyncio.wait_for(write(), SEND_TIMEOUT)
        except (asyncio.TimeoutError, ConnectionError):
            if self.client is owner:
                self.ready = False
                close = getattr(owner, "close", None)
                if close is not None:
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(close(code=1011, message=b"Voice send failed"), SEND_TIMEOUT)
            raise

    def cancel_voice_turn(self, device_id, reason="cancelled", **unused):
        for (_, target), task in tuple(self.plays.items()):
            if target == device_id:
                task.cancel()
        for turn in tuple(self.turns.values()):
            if turn.device.device_id == device_id and not turn.cancelled.is_set():
                turn.reason = {"barged": "barge_in", "muted": "mute"}.get(reason, reason)
                turn.cancelled.set()
                turn.device.cancel_event.set()

    async def trigger_voice_turn(self, device, on_thinking, post_turn_play,
                                 trigger_label="unknown", preroll_discard=0):
        wake = device.last_wake
        device.last_wake = None
        if not self.can_serve_turn(device.device_id):
            log.warning("[%s] External voice backend unavailable", device.device_id)
            return False
        if self.has_turn(device.device_id):
            raise RuntimeError("device already has an external turn")
        turn = Turn(device, self.client)
        turn.response = asyncio.get_running_loop().create_future()
        self.turns[turn.session_id] = turn
        started = time.monotonic()
        work = asyncio.create_task(self._run(turn, wake, trigger_label,
                                            preroll_discard, on_thinking, post_turn_play))
        cancel = asyncio.create_task(device.cancel_event.wait())
        aborted = asyncio.create_task(turn.cancelled.wait())
        try:
            done, _ = await asyncio.wait({work, cancel, aborted}, timeout=TURN_TIMEOUT,
                                         return_when=asyncio.FIRST_COMPLETED)
            if work in done and not device.cancel_event.is_set() and not turn.cancelled.is_set():
                await work
            else:
                if not done:
                    turn.reason = "timeout"
                elif device.barge_detected:
                    turn.reason = "barge_in"
                turn.cancelled.set()
                turn.outcome = "cancelled"
        except asyncio.CancelledError:
            turn.reason = "disconnect"
            turn.cancelled.set()
            raise
        except asyncio.TimeoutError:
            turn.reason = "timeout"
            turn.cancelled.set()
            turn.outcome = "timeout"
        except Exception as error:
            turn.reason = "failure"
            turn.cancelled.set()
            log.warning("[%s] External session %s failed during %s (%s)",
                        device.device_id, turn.session_id, turn.phase, type(error).__name__)
        finally:
            # Invalidate before any await. Late responses can never revive it.
            self.turns.pop(turn.session_id, None)
            for task in (work, cancel, aborted):
                task.cancel()
            await asyncio.gather(work, cancel, aborted, return_exceptions=True)
            if turn.cancelled.is_set():
                turn.outcome = {"barge_in": "barged", "mute": "muted"}.get(turn.reason, turn.outcome)
                with contextlib.suppress(Exception):
                    await device.send_control({"type": "speaker_flush"})
                with contextlib.suppress(Exception):
                    if turn.phase == "speaking" and turn.reason == "failure":
                        await self.send(turn.owner, turn.message("play_failed", message="playback failed"))
                    await self.send(turn.owner, turn.message("turn_cancel", reason=turn.reason))
            else:
                with contextlib.suppress(Exception):
                    await self.send(turn.owner, turn.message("turn_finished", outcome=turn.outcome))
            log.info("[%s] External session %s ended: %s", device.device_id,
                     turn.session_id, turn.reason if turn.cancelled.is_set() else turn.outcome)
            if self.persist is not None:
                wi = wake or {}
                record = dict(ts=time.time(), trigger=trigger_label,
                              wake_model=wi.get("model"), wake_score=wi.get("score"),
                              wake_threshold=wi.get("threshold"), noise_floor=wi.get("noise_floor"),
                              outcome=turn.outcome, total_ms=(time.monotonic() - started) * 1000,
                              audio_ms=turn.audio_bytes / 32, playback_ms=turn.playback_ms,
                              tts_bytes=turn.tts_bytes)
                with contextlib.suppress(Exception):
                    await self.persist(device, record)
                device.turn_history.append(record)
                device.last_turn_outcome = turn.outcome
            turn.finished.set()
        return (turn.continue_conversation and turn.outcome == "ok"
                and not turn.cancelled.is_set() and not device.cancel_event.is_set())

    async def _run(self, turn, wake, trigger, discard, on_thinking, play):
        info = dict(trigger="wakeword" if trigger.startswith("wakeword") else trigger,
                    audio=dict(sampleRate=16000, sampleWidth=2, channels=1, encoding="pcm_s16le"))
        if wake:
            for source, target in (("model", "wakeWord"), ("score", "wakeScore"),
                                   ("threshold", "wakeThreshold"), ("noise_floor", "noiseFloor")):
                if source in wake:
                    info[target] = wake[source]
        await self.send(turn.owner, turn.message("turn_start", **info))
        log.info("[%s] External session %s started: %s", turn.device.device_id, turn.session_id, trigger)
        reason = await asyncio.wait_for(self._input(turn, discard), INPUT_TIMEOUT)
        await self.send(turn.owner, turn.message("audio_end", reason=reason))
        log.info("[%s] External session %s input ended: %s", turn.device.device_id, turn.session_id, reason)
        if reason != "speech_end":
            turn.outcome = reason
            return
        turn.phase = "thinking"
        await on_thinking()
        message = await turn.response
        if message["type"] == "turn_error":
            log.warning("[%s] External session %s error: %r", turn.device.device_id,
                        turn.session_id, message.get("message", "")[:MAX_ERROR])
            turn.outcome = "error"
            return
        turn.phase = "speaking"
        await self.send(turn.owner, turn.message("play_started"))
        log.info("[%s] External session %s playback started", turn.device.device_id, turn.session_id)
        playback_start = time.monotonic()
        async with contextlib.aclosing(em_audio_stream._stream_tts_audio(message["audioUrl"])) as chunks:
            count = await play(chunks)
        turn.playback_ms = (time.monotonic() - playback_start) * 1000
        turn.tts_bytes = count
        if turn.device.cancel_event.is_set() or turn.cancelled.is_set():
            return
        if not count:
            raise RuntimeError("response contained no playable audio")
        turn.outcome = "ok"
        turn.continue_conversation = message.get("continueConversation", False)
        await self.send(turn.owner, turn.message("play_finished"))

    async def _input(self, turn, discard):
        """Reuse EchoMuse's speech gate, local endpoint clock and ASR denoiser."""
        device = turn.device
        vad = em_speechgate.new_turn()
        gate = em_speechgate.SpeechGate() if vad is not None else None
        denoiser = None
        if getattr(device, "ns_asr", False):
            import em_ns
            if em_ns.available():
                try:
                    denoiser = em_ns.StreamingDenoiser()
                except Exception:
                    log.warning("[%s] NS init failed; streaming raw audio", device.device_id)
        started = time.monotonic()
        first_audio = first_speech = last_speech = None
        speech = False
        capture = bytearray() if getattr(device, "save_utterances", False) else None
        device.last_utterance_pcm = None
        try:
            while True:
                now = time.monotonic()
                quiet, _ = em_turnclock.no_speech_verdict(now, started, first_audio, speech)
                if quiet:
                    return "no_speech_timeout"
                ended, _ = em_turnclock.ha_vad_stalled_verdict(
                    now, speech, False, first_speech, last_speech)
                if ended:
                    return "speech_end"
                try:
                    payload = await asyncio.wait_for(device.voice_queue.get(), 0.1)
                except asyncio.TimeoutError:
                    continue
                if payload is None or isinstance(payload, str):
                    return "no_speech_timeout" if payload == "vad_no_speech_timeout" else "speech_end"
                if not isinstance(payload, bytes) or len(payload) > MAX_CHUNK or len(payload) % 2:
                    raise ValueError("invalid microphone chunk")
                if discard:
                    discard -= 1
                    continue
                if first_audio is None:
                    first_audio = time.monotonic()
                frames = [payload]
                if gate is not None:
                    prob = await asyncio.to_thread(vad.prob, payload)
                    frames = gate.push(payload, prob)
                    if gate.open and not speech:
                        speech = True
                        await device.beam_lock()
                for payload in frames:
                    samples = np.frombuffer(payload, dtype=np.int16).astype(np.float64) / 32768
                    rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0
                    if rms >= max(3 * getattr(device, "noise_floor", 0.0), 0.004):
                        last_speech = time.monotonic()
                        if first_speech is None:
                            first_speech = last_speech
                        if not speech:
                            speech = True
                            await device.beam_lock()
                    if denoiser is not None:
                        try:
                            payload = await asyncio.to_thread(denoiser.process, payload)
                        except Exception:
                            denoiser = None
                            log.warning("[%s] NS failed; streaming raw audio", device.device_id)
                    if capture is not None:
                        import em_recordings
                        room = em_recordings.MAX_UTTERANCE_BYTES - len(capture)
                        capture.extend(payload[:max(0, room)])
                    turn.audio_bytes += len(payload)
                    await self.send(turn.owner, turn.message("audio", data=base64.b64encode(payload).decode("ascii")))
        finally:
            if capture:
                device.last_utterance_pcm = bytes(capture)

    async def handle(self, owner, message):
        if not isinstance(message, dict):
            raise ValueError("message must be an object")
        kind = _string(message, "type")
        device_id = _string(message, "deviceId")
        if kind in ("turn_response", "turn_error", "turn_cancel"):
            session_id = _string(message, "sessionId")
            turn = self.turns.get(session_id)
            if (turn is None or turn.owner is not owner or turn.device.device_id != device_id
                    or turn.cancelled.is_set() or turn.device.cancel_event.is_set()):
                raise ValueError("unknown or stale session")
            if kind == "turn_cancel":
                self.cancel_voice_turn(device_id, reason="backend_cancel")
                return
            if turn.phase != "thinking" or turn.response.done():
                raise ValueError("session is not awaiting a response")
            if kind == "turn_response":
                _url(message)
                if not isinstance(message.get("continueConversation", False), bool):
                    raise ValueError("invalid continueConversation")
                if "text" in message and (not isinstance(message["text"], str) or len(message["text"]) > MAX_TEXT):
                    raise ValueError("invalid text")
            elif not isinstance(message.get("message", ""), str):
                raise ValueError("invalid error message")
            # Store only fields the controller needs, bounded independently.
            turn.response.set_result(dict(type=kind, audioUrl=message.get("audioUrl"),
                                          continueConversation=message.get("continueConversation", False),
                                          message=message.get("message", "")[:MAX_ERROR]))
            log.info("[%s] External session %s received %s", device_id, session_id, kind)
            return
        if kind not in ("play", "stop"):
            raise ValueError("unknown message type")
        request_id = _string(message, "requestId")
        key = (request_id, device_id)
        if kind == "stop":
            task = self.plays.get(key)
            if task is None:
                raise ValueError("unknown playback request")
            task.cancel()
            return
        import em_voice_backend
        if em_voice_backend.name() != "external":
            raise ValueError("external voice backend is not selected")
        device = self.get_device(device_id)
        if device is None or device_id not in self._play_callbacks:
            raise ValueError("device is not connected")
        if message.get("kind", "announcement") != "announcement":
            raise ValueError("only announcement playback is supported")
        url = _url(message)
        if (key in self.plays or any(target == device_id for _, target in self.plays)
                or device.voice_lock.locked() or self.has_turn(device_id)
                or len(self.turns) + len(self.plays) >= MAX_SESSIONS):
            raise ValueError("device or backend busy")
        task = asyncio.create_task(self._play(owner, key, device, url))
        self.plays[key] = task
        # Cancellation before the coroutine first runs skips its finally.
        task.add_done_callback(lambda done: self.plays.pop(key, None)
                               if self.plays.get(key) is done else None)

    async def _play(self, owner, key, device, url):
        status = dict(requestId=key[0], deviceId=key[1])
        try:
            async with asyncio.timeout(TURN_TIMEOUT), device.voice_lock:
                pcm = bytearray()
                async with contextlib.aclosing(em_audio_stream._stream_tts_audio(url)) as chunks:
                    async for chunk in chunks:
                        if len(pcm) + len(chunk) > MAX_PLAY_BYTES:
                            raise ValueError("announcement too long")
                        pcm.extend(chunk)
                if not pcm:
                    raise ValueError("announcement contained no audio")
                await self.send(owner, dict(type="play_started", **status))
                if await self._play_callbacks[device.device_id](bytes(pcm)) is False:
                    raise RuntimeError("announcement cancelled")
                await self.send(owner, dict(type="play_finished", **status))
        except asyncio.CancelledError:
            device.cancel_event.set()
            with contextlib.suppress(Exception):
                await device.send_control({"type": "speaker_flush"})
            with contextlib.suppress(Exception):
                await self.send(owner, dict(type="play_finished", reason="cancelled", **status))
        except Exception as error:
            log.warning("[%s] External announcement %s failed (%s)",
                        device.device_id, key[0], type(error).__name__)
            with contextlib.suppress(Exception):
                await device.send_control({"type": "speaker_flush"})
            with contextlib.suppress(Exception):
                await self.send(owner, dict(type="play_failed", message="playback failed", **status))
        finally:
            self.plays.pop(key, None)

    async def websocket(self, request, controller_version, voice_backend):
        ws = web.WebSocketResponse(heartbeat=30, max_msg_size=MAX_MESSAGE)
        # Reserve synchronously before prepare() yields: simultaneous upgrades
        # cannot both win. A failed upgrade also releases its reservation.
        if not self.claim(ws):
            raise web.HTTPConflict(reason="External voice backend already connected")
        try:
            await ws.prepare(request)
            await self.send(ws, dict(type="hello", protocolVersion=1,
                                     controllerVersion=controller_version, voiceBackend=voice_backend))
            self.ready = True
            async for incoming in ws:
                if incoming.type != web.WSMsgType.TEXT:
                    if incoming.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                        break
                    await self.send(ws, dict(type="protocol_error", message="JSON text required"))
                    continue
                try:
                    message = json.loads(incoming.data)
                    await self.handle(ws, message)
                except (ValueError, TypeError, KeyError):
                    # Never echo raw JSON, credentials, audio, or supplied URLs.
                    log.warning("External voice message rejected")
                    await self.send(ws, dict(type="protocol_error", message="invalid message or stale request"))
        finally:
            await self.release(ws)
        return ws


backend = ExternalVoiceBackend()
