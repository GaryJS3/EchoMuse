"""Exercise the backend boundary without wake models, HA, or physical devices."""
import ast
import asyncio
import base64
import contextlib
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web, WSServerHandshakeError
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

import em_auth
import em_external_voice as external
import em_voice_backend as selector

ROOT = Path(__file__).resolve().parents[1]


class Socket:
    closed = False

    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)


class Device:
    def __init__(self, device_id="echo"):
        self.device_id = device_id
        self.last_wake = {"model": "hey_jarvis", "score": .74}
        self.voice_queue = asyncio.Queue(maxsize=8)
        self.voice_lock = asyncio.Lock()
        self.cancel_event = asyncio.Event()
        self.barge_detected = False
        self.controls = []
        self.thinking = False
        self.played = []
        self.beams = 0
        self.turn_history = []

    async def send_control(self, message):
        self.controls.append(message)

    async def beam_lock(self):
        self.beams += 1

    async def think(self):
        self.thinking = True

    async def play(self, chunks):
        async for chunk in chunks:
            self.played.append(chunk)
        return sum(map(len, self.played))


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(selector, "_name", "esphome")
    monkeypatch.setattr(selector, "_turn_backends", {})
    monkeypatch.setattr(external.em_speechgate, "new_turn", lambda: None)

    async def audio(url):
        yield b"response PCM"
    monkeypatch.setattr(external.em_audio_stream, "_stream_tts_audio", audio)


def connected():
    backend = external.ExternalVoiceBackend()
    ws = Socket()
    assert backend.claim(ws)
    backend.ready = True  # hello has been sent
    return backend, ws


async def until(predicate):
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0)


def begin(backend, device, *, speech_end=True, play=None):
    device.voice_queue.put_nowait(b"\x00\x10" * 1280)
    if speech_end:
        device.voice_queue.put_nowait("vad_end")
    return asyncio.create_task(backend.trigger_voice_turn(
        device, device.think, play or device.play, trigger_label="wakeword(0.74)"))


def reply(turn, kind="turn_response", **values):
    return dict(type=kind, sessionId=turn.session_id,
                deviceId=turn.device.device_id, **values)


def test_default_and_both_selector_delegates(monkeypatch):
    calls = []

    async def trigger(**kwargs):
        calls.append(kwargs)
        return True
    ha = SimpleNamespace(can_serve_turn=lambda d: d == "echo", trigger_voice_turn=trigger)
    monkeypatch.setitem(sys.modules, "em_esphome", ha)
    assert selector.name() == "esphome"
    assert selector.can_serve_turn("echo")
    assert not selector.can_serve_turn("offline")
    device = Device()
    assert asyncio.run(selector.trigger_voice_turn(device=device)) is True
    assert calls == [{"device": device}]
    ext = SimpleNamespace(can_serve_turn=lambda d: False, trigger_voice_turn=trigger)
    monkeypatch.setattr(external, "backend", ext)
    selector.configure("external")
    assert not selector.can_serve_turn("echo")
    assert asyncio.run(selector.trigger_voice_turn(device=device)) is True
    assert len(calls) == 2


@pytest.mark.parametrize("value", ["ha", "External", "", None, 1, {}, []])
def test_invalid_backend_rejected(value):
    with pytest.raises(ValueError):
        selector.configure(value)
    assert selector.name() == "esphome"


def test_readiness_and_exclusive_connection_slot():
    async def main():
        backend = external.ExternalVoiceBackend()
        assert not backend.can_serve_turn("echo")
        first, second = Socket(), Socket()
        assert backend.claim(first)
        assert not backend.can_serve_turn("echo")  # no hello yet
        assert not backend.claim(second)
        backend.ready = True
        assert backend.can_serve_turn("echo")
        await backend.release(second)  # a rejected socket cannot release owner
        assert backend.client is first
        await backend.release(first)
        assert not backend.can_serve_turn("echo")
        assert backend.claim(second)
    asyncio.run(main())


def test_turn_order_metadata_audio_and_response_playback():
    async def main():
        backend, ws = connected()
        device = Device()
        task = begin(backend, device)
        await until(lambda: device.thinking)
        turn = next(iter(backend.turns.values()))
        assert [m["type"] for m in ws.messages] == ["turn_start", "audio", "audio_end"]
        start, audio, end = ws.messages
        assert start["wakeWord"] == "hey_jarvis" and start["wakeScore"] == .74
        assert start["audio"] == dict(sampleRate=16000, sampleWidth=2, channels=1, encoding="pcm_s16le")
        assert base64.b64decode(audio["data"]) == b"\x00\x10" * 1280
        assert all(m["sessionId"] == turn.session_id and m["deviceId"] == "echo" for m in ws.messages)
        assert end["reason"] == "speech_end"
        await backend.handle(ws, reply(turn, audioUrl="http://test/response.wav"))
        assert await task is False
        assert device.played == [b"response PCM"]
        assert [m["type"] for m in ws.messages][-3:] == ["play_started", "play_finished", "turn_finished"]
        with pytest.raises(ValueError):
            await backend.handle(ws, reply(turn, audioUrl="http://test/late.wav"))
        assert not backend.turns
    asyncio.run(main())


@pytest.mark.parametrize("stage", ["listening", "thinking", "speaking"])
def test_disconnect_cleans_every_turn_phase(stage):
    async def main():
        backend, ws = connected()
        device = Device()
        playback = asyncio.Event()

        async def blocked_play(chunks):
            playback.set()
            await asyncio.Event().wait()

        task = begin(backend, device, speech_end=stage != "listening", play=blocked_play)
        await until(lambda: bool(backend.turns))
        turn = next(iter(backend.turns.values()))
        if stage != "listening":
            await until(lambda: device.thinking)
        if stage == "speaking":
            await backend.handle(ws, reply(turn, audioUrl="http://test/audio"))
            await playback.wait()
        await backend.release(ws)
        await task
        assert not backend.turns and not backend.can_serve_turn("echo")
        assert device.cancel_event.is_set()
        assert {"type": "speaker_flush"} in device.controls
        assert any(m["type"] == "turn_cancel" and m["reason"] == "disconnect" for m in ws.messages)
        assert backend.claim(Socket())
    asyncio.run(main())


def test_barge_cancels_old_session_and_cannot_play_old_response():
    async def main():
        backend, ws = connected()
        device = Device()
        task = begin(backend, device)
        await until(lambda: device.thinking)
        old = next(iter(backend.turns.values()))
        device.barge_detected = True
        backend.cancel_voice_turn("echo", reason="barged")
        with pytest.raises(ValueError):
            await backend.handle(ws, reply(old, audioUrl="http://test/stale"))
        await task
        assert any(m["type"] == "turn_cancel" and m["reason"] == "barge_in" for m in ws.messages)
        device.cancel_event.clear()
        device.barge_detected = device.thinking = False
        device.last_wake = {"model": "hey_jarvis"}
        task = begin(backend, device)
        await until(lambda: device.thinking)
        new = next(iter(backend.turns.values()))
        assert new.session_id != old.session_id
        with pytest.raises(ValueError):
            await backend.handle(ws, reply(old, audioUrl="http://test/stale"))
        await backend.handle(ws, reply(new, audioUrl="http://test/fresh"))
        await task
        assert device.played == [b"response PCM"]
    asyncio.run(main())


@pytest.mark.parametrize("kind", ["turn_error", "turn_cancel"])
def test_backend_can_end_current_session(kind):
    async def main():
        backend, ws = connected()
        device = Device()
        task = begin(backend, device)
        await until(lambda: device.thinking)
        turn = next(iter(backend.turns.values()))
        await backend.handle(ws, reply(turn, kind, message="failure" * 1000))
        await task
        assert not backend.turns and not device.played
    asyncio.run(main())


def test_decoder_failure_is_reported_and_session_released(monkeypatch):
    async def broken(url):
        raise RuntimeError("private upstream details")
        yield
    monkeypatch.setattr(external.em_audio_stream, "_stream_tts_audio", broken)

    async def main():
        backend, ws = connected()
        device = Device()
        task = begin(backend, device)
        await until(lambda: device.thinking)
        await backend.handle(ws, reply(next(iter(backend.turns.values())), audioUrl="http://test/broken"))
        await task
        assert not backend.turns
        assert any(m["type"] == "play_failed" for m in ws.messages)
        assert "private" not in json.dumps(ws.messages)
    asyncio.run(main())


def test_no_speech_ends_input_without_waiting_for_response():
    async def main():
        backend, ws = connected()
        device = Device()
        device.last_wake = None
        device.voice_queue.put_nowait("vad_no_speech_timeout")
        await backend.trigger_voice_turn(device, device.think, device.play, trigger_label="button")
        assert ws.messages[1]["reason"] == "no_speech_timeout"
        assert "wakeWord" not in ws.messages[0]
        assert not device.thinking and not device.played and not backend.turns
    asyncio.run(main())


@pytest.mark.parametrize("message", [None, [], {}, {"type": "other", "deviceId": "echo"},
    {"type": "turn_response", "deviceId": "echo", "sessionId": "unknown", "audioUrl": "http://test"}])
def test_malformed_and_unknown_messages_rejected(message):
    async def main():
        backend, ws = connected()
        with pytest.raises(ValueError):
            await backend.handle(ws, message)
    asyncio.run(main())


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://test/audio", "http://user:password@test/a", "http://:password@test/a", "http://", "x" * 2049])
def test_invalid_response_urls_do_not_consume_session(url):
    async def main():
        backend, ws = connected()
        device = Device()
        task = begin(backend, device)
        await until(lambda: device.thinking)
        turn = next(iter(backend.turns.values()))
        with pytest.raises(ValueError):
            await backend.handle(ws, reply(turn, audioUrl=url))
        assert not turn.response.done()
        backend.cancel_voice_turn("echo")
        await task
    asyncio.run(main())


def test_slow_client_fails_turn_without_unbounded_queue(monkeypatch):
    monkeypatch.setattr(external, "SEND_TIMEOUT", .01)

    async def main():
        backend, ws = connected()
        async def slow(message):
            await asyncio.Event().wait()
        ws.send_json = slow
        device = Device()
        await asyncio.wait_for(backend.trigger_voice_turn(device, device.think, device.play), .2)
        assert not backend.turns
    asyncio.run(main())


def test_proactive_play_and_stop_target_owned_request():
    async def main():
        selector.configure("external")
        backend, ws = connected()
        device = Device()
        backend.get_device = lambda d: device if d == "echo" else None
        played = []
        async def play(pcm):
            played.append(pcm)
            return True
        backend.register_device("echo", play)
        message = dict(type="play", requestId="p1", deviceId="echo", audioUrl="http://test/audio", kind="announcement")
        await backend.handle(ws, message)
        await until(lambda: not backend.plays)
        assert played == [b"response PCM"]
        assert [m["type"] for m in ws.messages] == ["play_started", "play_finished"]
        with pytest.raises(ValueError):
            await backend.handle(ws, {**message, "deviceId": "unknown"})
        block = asyncio.Event()
        async def long_play(pcm):
            block.set()
            await asyncio.Event().wait()
        backend.register_device("echo", long_play)
        await backend.handle(ws, message)
        await block.wait()
        with pytest.raises(ValueError):
            await backend.handle(ws, dict(type="stop", requestId="stale", deviceId="echo"))
        await backend.handle(ws, dict(type="stop", requestId="p1", deviceId="echo"))
        await until(lambda: not backend.plays)
        assert ws.messages[-1]["reason"] == "cancelled"
        assert not device.voice_lock.locked()
        assert {"type": "speaker_flush"} in device.controls
    asyncio.run(main())


def test_external_unavailable_never_dispatches_to_ha(monkeypatch):
    ha = SimpleNamespace(can_serve_turn=lambda d: pytest.fail("HA fallback"))
    monkeypatch.setitem(sys.modules, "em_esphome", ha)
    monkeypatch.setattr(external, "backend", external.ExternalVoiceBackend())
    selector.configure("external")
    assert not selector.can_serve_turn("echo")
    device = Device()
    assert asyncio.run(selector.trigger_voice_turn(device=device, on_thinking=device.think, post_turn_play=device.play)) is False


def _api_function(name, namespace):
    tree = ast.parse((ROOT / "em_api.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name)
    fn.decorator_list = []
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "em_api.py", "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("transport", ["bearer", "cookie", "query"])
def test_websocket_session_auth_compatibility(monkeypatch, transport):
    monkeypatch.setattr(em_auth.db, "get_session", lambda token: {"user_id": 7} if token == "secret" else None)
    monkeypatch.setattr(em_auth.db, "get_user_by_id", lambda uid: {"id": uid, "username": "admin", "role": "admin"})
    headers = {"Authorization": "Bearer secret"} if transport == "bearer" else {}
    if transport == "cookie":
        headers["Cookie"] = f"{em_auth.AUTH_COOKIE}=secret"
    path = "/api/voice?token=secret" if transport == "query" else "/api/voice"
    user = asyncio.run(em_auth.ws_resolve_session(make_mocked_request("GET", path, headers=headers)))
    assert user["role"] == "admin"


def test_real_websocket_auth_ownership_hello_and_bad_messages(monkeypatch):
    backend = external.ExternalVoiceBackend()
    monkeypatch.setattr(em_auth.db, "get_session", lambda token: {"user_id": token} if token in ("admin", "reader") else None)
    monkeypatch.setattr(em_auth.db, "get_user_by_id", lambda uid: {"id": 1, "username": uid, "role": "admin" if uid == "admin" else "readonly"})
    namespace = dict(auth=em_auth, web=web, em_external_voice=SimpleNamespace(backend=backend),
                     em_voice_backend=selector, CONTROLLER_VERSION="test")
    handler = _api_function("_ws_voice", namespace)

    async def main():
        app = web.Application()
        app.router.add_get("/api/voice", handler)
        async with TestClient(TestServer(app)) as client:
            for headers, status in (({}, 401), ({"Authorization": "Bearer reader"}, 403)):
                with pytest.raises(WSServerHandshakeError) as error:
                    await client.ws_connect("/api/voice", headers=headers)
                assert error.value.status == status
                assert backend.client is None
            ws = await client.ws_connect("/api/voice", headers={"Authorization": "Bearer admin"})
            assert await ws.receive_json() == dict(type="hello", protocolVersion=1, controllerVersion="test", voiceBackend="esphome")
            assert backend.can_serve_turn("echo")
            with pytest.raises(WSServerHandshakeError) as error:
                await client.ws_connect("/api/voice?token=admin")
            assert error.value.status == 409
            await ws.send_str("invalid JSON")
            assert (await ws.receive_json())["type"] == "protocol_error"
            await ws.send_json(["not an object"])
            assert (await ws.receive_json())["type"] == "protocol_error"
            await ws.close()
            await until(lambda: backend.client is None)
            assert not backend.can_serve_turn("echo")
            async with client.ws_connect("/api/voice?token=admin") as replacement:
                await replacement.receive_json()
                await replacement.send_str("x" * (external.MAX_MESSAGE + 1))
                closed = await replacement.receive(timeout=1)
                assert closed.type == web.WSMsgType.CLOSE
                assert closed.data == 1009
            await until(lambda: backend.client is None)
    asyncio.run(main())


def test_controller_preserves_readiness_placement_and_esphome_servers():
    source = (ROOT / "em_controller.py").read_text(encoding="utf-8")
    assert "esphome.can_serve_turn(" not in source
    assert "esphome.trigger_voice_turn(" not in source
    for name in ("_barge_watcher", "_private_wake_turn", "_private_barge", "_stream_listen"):
        tree = ast.parse(source)
        fn = next((n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == name), None)
        assert fn is not None
        text = ast.get_source_segment(source, fn)
        assert text.index("voice_backend.can_serve_turn(") < text.index("await _claim_wake(")
    assert "esphome.start_esphome_servers(" in source
    assert "await esphome.device_connected(" in source
    assert "_run_streaming_post_turn_playback(" in source
    assert "em_external_voice.backend.register_device(device_id, _standalone_play)" in source
    assert "stop_esphome" not in (ROOT / "em_voice_backend.py").read_text()


def test_system_config_defaults_validation_and_persistence():
    stored = {}
    namespace = dict(web=web, asyncio=asyncio,
                     db=SimpleNamespace(get_all_config=lambda: dict(stored),
                                        set_config=lambda k, v: stored.update({k: v})),
                     em_voice_backend=selector, _ok=lambda value: value,
                     _error=lambda code, message, status: {"error": code, "status": status})
    async def body(request):
        return request
    namespace["_json_body"] = body
    get = _api_function("_get_system_config", namespace)
    patch = _api_function("_patch_system_config", namespace)

    async def main():
        assert (await get(None))["voiceBackend"] == "esphome"
        for value in ("unknown", None, 1, [], {}):
            assert (await patch({"voiceBackend": value}))["status"] == 400
            assert not stored
        assert await patch({"voiceBackend": "external"}) == {"voiceBackend": "external"}
        assert stored == {"voiceBackend": "external"}
        assert selector.name() == "external"
        assert (await get(None))["voiceBackend"] == "external"
    asyncio.run(main())


def test_selection_change_keeps_cancellation_with_existing_turn(monkeypatch):
    running = asyncio.Event()
    cancelled = []
    async def trigger(**kwargs):
        running.set()
        await asyncio.Event().wait()
    ha = SimpleNamespace(trigger_voice_turn=trigger,
                         cancel_voice_turn=lambda *a, **kw: cancelled.append(kw))
    monkeypatch.setitem(sys.modules, "em_esphome", ha)

    async def main():
        task = asyncio.create_task(selector.trigger_voice_turn(device=Device()))
        await running.wait()
        selector.configure("external")
        selector.cancel_voice_turn("echo", reason="muted")
        assert cancelled == [{"abort_ha": False, "reason": "muted"}]
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert not selector._turn_backends
    asyncio.run(main())


def test_explicit_ha_conversation_keeps_its_requesting_pipeline(monkeypatch):
    calls = []
    async def trigger(**kwargs):
        calls.append(kwargs)
        return False
    monkeypatch.setitem(sys.modules, "em_esphome", SimpleNamespace(trigger_voice_turn=trigger))
    selector.configure("external")
    device = Device()
    assert asyncio.run(selector.trigger_voice_turn(ha_initiated=True, device=device)) is False
    assert calls == [{"device": device}]
    source = (ROOT / "em_controller.py").read_text(encoding="utf-8")
    start = source.index("async def _start_conversation(")
    assert "ha_initiated=True" in source[start:source.index("await esphome.device_connected(", start)]


def test_session_limits_wrong_device_and_early_response(monkeypatch):
    monkeypatch.setattr(external, "MAX_SESSIONS", 1)
    async def main():
        backend, ws = connected()
        device = Device()
        task = begin(backend, device, speech_end=False)
        await until(lambda: bool(backend.turns))
        turn = next(iter(backend.turns.values()))
        assert not backend.can_serve_turn("other")
        with pytest.raises(ValueError):
            await backend.handle(ws, reply(turn, audioUrl="http://test/early"))
        device.voice_queue.put_nowait("vad_end")
        await until(lambda: device.thinking)
        with pytest.raises(ValueError):
            await backend.handle(ws, {**reply(turn, audioUrl="http://test/a"), "deviceId": "other"})
        assert not turn.response.done()
        backend.cancel_voice_turn("echo")
        await task
        assert backend.can_serve_turn("other")
    asyncio.run(main())


def test_turn_lifetime_is_bounded(monkeypatch):
    monkeypatch.setattr(external, "TURN_TIMEOUT", .01)
    async def main():
        backend, ws = connected()
        device = Device()
        await begin(backend, device)
        assert not backend.turns
        assert any(m["type"] == "turn_cancel" and m["reason"] == "timeout" for m in ws.messages)
    asyncio.run(main())


def test_announcements_have_memory_bound_and_disconnect_cleanup(monkeypatch):
    monkeypatch.setattr(external, "MAX_PLAY_BYTES", 4)
    async def main():
        selector.configure("external")
        backend, ws = connected()
        device = Device()
        backend.get_device = lambda d: device
        async def play(pcm):
            pytest.fail("oversized announcement reached speaker")
        backend.register_device("echo", play)
        message = dict(type="play", requestId="p1", deviceId="echo", audioUrl="http://test/audio")
        await backend.handle(ws, message)
        assert not backend.can_serve_turn("echo")
        await until(lambda: not backend.plays)
        assert ws.messages[-1]["type"] == "play_failed"
        assert not device.voice_lock.locked()
        # Release immediately, before the newly scheduled coroutine runs.
        await backend.handle(ws, message)
        await backend.release(ws)
        await asyncio.sleep(0)
        assert not backend.plays and not device.voice_lock.locked()
    asyncio.run(main())


def test_oversized_microphone_chunk_fails_cleanly():
    async def main():
        backend, ws = connected()
        device = Device()
        device.voice_queue.put_nowait(b"a" * (external.MAX_CHUNK + 2))
        await backend.trigger_voice_turn(device, device.think, device.play)
        assert not backend.turns
        assert not any(m["type"] == "audio" for m in ws.messages)
    asyncio.run(main())


@pytest.mark.parametrize("stage", ["listening", "thinking", "speaking", "playback_failure"])
def test_existing_controller_lifecycle_restores_state_and_music(monkeypatch, stage):
    """Execute the shipped turn orchestrator, including its real callbacks and
    finally blocks, without importing hardware/model startup dependencies."""
    backend, ws = connected()
    monkeypatch.setattr(external, "backend", backend)
    selector.configure("external")
    device = Device()
    device.mic_queue = asyncio.Queue()
    device.oww_paused = asyncio.Event()
    device.oww_paused.set()
    device.oww_paused_since = 1
    device.listen_session = None
    device.private_listening = True
    device.barge_in_enabled = device.led_anim_capable = False
    device.listening = device.speaking = False
    mic_stops, media, released = [], [], []
    playback = asyncio.Event()

    async def stop_mic():
        mic_stops.append(True)
    device.mic_stop = stop_mic
    async def noop(*args):
        pass
    async def spin(device, stopped):
        await stopped.wait()
    async def interrupt(device_id):
        media.append("interrupt")
    async def resume(device_id):
        media.append("resume")
    async def stream(device, chunks):
        device.speaking = True
        playback.set()
        try:
            if stage == "playback_failure":
                raise RuntimeError("speaker failed")
            await asyncio.Event().wait()
        finally:
            device.speaking = False

    namespace = dict(asyncio=asyncio, contextlib=contextlib, Device=Device,
                     log=logging.getLogger("test"), voice_backend=selector,
                     esphome=SimpleNamespace(VOICE_PREROLL_DISCARD=0),
                     em_player=SimpleNamespace(interrupt=interrupt, resume_interrupted=resume),
                     leds_listening=noop, _push_device_state=noop, _leds_turn_end=noop,
                     leds_spin_green=spin, _run_streaming_post_turn_playback=stream,
                     _wake_arbiter=SimpleNamespace(release=lambda d: released.append(d)))
    source = (ROOT / "em_controller.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_run_voice_locked")
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "em_controller.py", "exec"), namespace)

    async def main():
        task = asyncio.create_task(namespace["_run_voice_locked"](device))
        await until(lambda: bool(backend.turns))
        turn = next(iter(backend.turns.values()))
        if stage != "listening":
            device.voice_queue.put_nowait(b"\x00\x10" * 1280)
            device.voice_queue.put_nowait("vad_end")
            await until(lambda: device.thinking)
        if stage in ("speaking", "playback_failure"):
            await backend.handle(ws, reply(turn, audioUrl="http://test/audio"))
            await playback.wait()
        if stage != "playback_failure":
            await backend.release(ws)
        await task
        assert not device.listening and not device.thinking and not device.speaking
        assert not device.oww_paused.is_set() and not device.voice_lock.locked()
        assert mic_stops and released == ["echo"]
        assert media == ["interrupt", "resume"]
        assert not backend.turns
    asyncio.run(main())
