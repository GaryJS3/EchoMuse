"""Bounded microphone windows use external turns without routing normal commands."""
import ast
import asyncio
from types import SimpleNamespace

import pytest

from test_external_voice import Device, ROOT, connected, until, reply
import em_external_voice as external
import em_voice_backend as selector


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(selector, "_name", "external")
    monkeypatch.setattr(external.em_speechgate, "new_turn", lambda: None)


def setup(listen):
    backend, ws = connected()
    device = Device()
    backend.get_device = lambda identity: device if identity == "echo" else None
    backend.register_device("echo", device.play, listen=listen)
    return backend, ws, device


def command(**values):
    return dict(type="listen_for_acknowledgement", deviceId="echo", requestId="window", durationMs=30000, **values)


@pytest.mark.asyncio
async def test_unrelated_speech_reopens_until_acknowledgement():
    calls = 0

    async def listen(opened):
        nonlocal calls
        calls += 1
        await opened()
        device.voice_queue.put_nowait(b"\x00\x10" * 1280)
        device.voice_queue.put_nowait("vad_end")
        await backend.trigger_voice_turn(device, device.think, device.play, trigger_label="acknowledgement")

    backend, ws, device = setup(listen)
    await backend.handle(ws, command())
    window = backend.listens["echo"]
    for acknowledged in (False, True):
        await until(lambda: bool(backend.turns) and next(iter(backend.turns.values())).phase == "thinking")
        turn = next(iter(backend.turns.values()))
        start = next(m for m in ws.messages if m["type"] == "turn_start" and m["sessionId"] == turn.session_id)
        assert start["acknowledgementRequestId"] == "window"
        assert "wakeWord" not in start
        with pytest.raises(ValueError):
            await backend.handle(ws, reply(turn, kind="turn_acknowledgement", requestId="stale", acknowledged=True))
        await backend.handle(ws, reply(turn, kind="turn_acknowledgement", requestId="window", acknowledged=acknowledged))
        await until(lambda: turn.finished.is_set())
    await asyncio.wait_for(window.task, 1)
    assert calls == 2
    assert device.played == []
    assert not device.cancel_event.is_set()
    assert not backend.turns and not backend.listens
    assert ws.messages[-1]["type"] == "acknowledgement_listen_finished"
    assert ws.messages[-1]["reason"] == "acknowledged"
    assert sum(m["type"] == "acknowledgement_listen_result" for m in ws.messages) == 1


@pytest.mark.asyncio
async def test_deadline_closes_microphone_and_does_not_extend_on_silence():
    closed = asyncio.Event()
    calls = 0

    async def listen(opened):
        nonlocal calls
        calls += 1
        await opened()
        if calls == 1:  # one no-speech turn, followed by another listening turn
            return
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    backend, ws, _ = setup(listen)
    await backend.handle(ws, {**command(), "durationMs": 30})
    window = backend.listens["echo"]
    await asyncio.wait_for(window.task, 1)
    assert calls == 2 and closed.is_set()
    assert ws.messages[-1]["reason"] == "timeout"
    assert not backend.listens


@pytest.mark.parametrize("action", ["stop", "disconnect", "mute", "release"])
@pytest.mark.asyncio
async def test_stop_disconnect_mute_and_release_cancel_window(action):
    closed = asyncio.Event()

    async def listen(opened):
        await opened()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    backend, ws, _ = setup(listen)
    await backend.handle(ws, command())
    window = backend.listens["echo"]
    await until(lambda: window.started)
    await backend.handle(ws, dict(type="stop_acknowledgement_listen", deviceId="echo", requestId="stale"))
    assert not window.task.done()
    if action == "stop":
        await backend.handle(ws, dict(type="stop_acknowledgement_listen", deviceId="echo", requestId="window"))
    elif action == "disconnect":
        backend.device_gone("echo")
    elif action == "mute":
        backend.cancel_voice_turn("echo", reason="muted")
    else:
        await backend.release(ws)
    await asyncio.wait_for(window.task, 1)
    assert closed.is_set() and not backend.listens


@pytest.mark.parametrize("duration", [0, -1, 30001, True, "30000"])
@pytest.mark.asyncio
async def test_invalid_duration_never_opens_microphone(duration):
    async def listen(opened):
        pytest.fail("invalid listening request reached microphone")
    backend, ws, _ = setup(listen)
    with pytest.raises(ValueError):
        await backend.handle(ws, {**command(), "durationMs": duration})
    assert not backend.listens


@pytest.mark.parametrize("condition", ["muted", "busy", "offline", "wrong_owner", "wrong_backend"])
@pytest.mark.asyncio
async def test_unavailable_or_unowned_listening_never_opens_microphone(condition, monkeypatch):
    async def listen(opened):
        pytest.fail("unavailable listening request reached microphone")
    backend, ws, device = setup(listen)
    if condition == "muted":
        device.muted = True
    elif condition == "busy":
        await device.voice_lock.acquire()
    elif condition == "offline":
        backend.device_gone("echo")
    elif condition == "wrong_backend":
        monkeypatch.setattr(selector, "_name", "esphome")
    if condition == "wrong_owner":
        with pytest.raises(ValueError):
            await backend.handle(object(), command())
    else:
        await backend.handle(ws, command())
        assert ws.messages[-1]["status"] == "rejected"
    assert not backend.listens


@pytest.mark.asyncio
async def test_actual_controller_callback_starts_turn_stream_and_restores_wake_stream_on_cancel():
    controls = []
    async def control(name):
        controls.append(name)
    async def run(device, **values):
        assert values == dict(trigger_label="acknowledgement", is_wakeword=False)
        assert device.oww_paused.is_set() and device.last_wake is None
        raise asyncio.CancelledError()
    device = SimpleNamespace(muted=False, last_wake={"model": "old"}, cancel_event=asyncio.Event(),
                             oww_paused=asyncio.Event(), mic_stop=lambda: control("stop"),
                             mic_start_turn=lambda: control("turn"), mic_start=lambda: control("wake"))
    tree = ast.parse((ROOT / "em_controller.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_acknowledgement_turn")
    namespace = dict(asyncio=asyncio, _device_ref=device, _run_voice_locked=run)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "em_controller.py", "exec"), namespace)
    with pytest.raises(asyncio.CancelledError):
        await namespace[fn.name](lambda: control("opened"))
    assert controls == ["stop", "turn", "opened", "stop", "wake"]
