import asyncio
import json
from unittest.mock import AsyncMock

import pytest

import em_listen
import em_wake_observation as observation


class Model:
    def __init__(self, second):
        self.second = second
        self.frames = []

    def predict(self, samples):
        self.frames.append(samples.tobytes())
        return {"jarvis": self.second}


def wire(tap, seq, value, listening=True):
    pcm = bytes([value]) * observation.FRAME_BYTES
    frame = em_listen.Frame(pcm, asyncio.get_running_loop().time())
    tap.feed(frame, pcm, seq, listening)
    return frame


@pytest.mark.parametrize("second,verdict", [(0.975, "would_accept"), (0.97499, "would_reject")])
def test_both_decisions_use_the_following_frame_and_the_original_threshold(monkeypatch, second, verdict):
    rows = []
    monkeypatch.setattr(observation.em_dbwriter, "submit", lambda *args: rows.append(args))

    async def run():
        model, tap, reset = Model(second), observation.FrameTap(), AsyncMock()
        first = wire(tap, 65535, 1)
        finished = observation.after_crossing("kitchen", model, "jarvis", 0.975, 0.99,
                                              first, True, reset)
        await asyncio.sleep(0)
        reset.assert_not_awaited()
        assert not finished.done()
        # Routing has already flipped to the voice queue. The observer sees
        # the same next audio, even across a 16-bit sequence rollover.
        command = wire(tap, 0, 2, listening=False)
        await observation.ready(model)
        await finished
        assert model.frames == [bytes(command)]
        reset.assert_awaited_once_with(model)
        assert model not in observation._pending

    asyncio.run(run())
    _, device, level, source, message = rows.pop()
    assert (device, level, source) == ("kitchen", "info", "controller")
    record = json.loads(message.removeprefix("wake_two_frame "))
    assert record["verdict"] == verdict and record["secondScore"] == second
    assert record["threshold"] == 0.975 and record["candidateTsMs"] > 0


@pytest.mark.parametrize("boundary", ["gap", "socket_close", "unaligned", "timeout", "predict_error"])
def test_missing_evidence_is_reported_and_always_releases_model(monkeypatch, boundary):
    rows = []
    monkeypatch.setattr(observation, "_record", lambda _, **fields: rows.append(fields))

    async def run():
        model, tap, reset = Model(float("nan")), observation.FrameTap(), AsyncMock()
        first = wire(tap, 10, 1)
        if boundary == "socket_close":
            tap.end()
        elif boundary == "gap":
            wire(tap, 12, 2)
        elif boundary == "predict_error":
            wire(tap, 11, 2)
        await observation.after_crossing("bedroom", model, "jarvis", 0.975, 0.99,
                                         first, boundary != "unaligned", reset)
        assert rows[0]["verdict"] == "unavailable" and rows[0]["secondScore"] is None
        assert len(model.frames) == (1 if boundary == "predict_error" else 0)
        reset.assert_awaited_once_with(model)

    asyncio.run(run())


def test_observer_does_not_consume_the_voice_frame_or_defer_the_turn(monkeypatch):
    import _oww_stub
    _oww_stub.install()
    try:
        import em_controller as controller
    finally:
        _oww_stub.uninstall()

    async def run():
        device = controller.Device("office", "127.0.0.1", [], AsyncMock())
        device.oww_model = "jarvis"
        device.oww_on_device = "shadow"
        device.oww_threshold = 0.975
        monkeypatch.setitem(controller.em_state.devices, device.device_id, device)
        records = []
        monkeypatch.setattr(observation, "_record", lambda _, **fields: records.append(fields))
        monkeypatch.setattr(controller.em_dbwriter, "submit", lambda *args: None)
        model, tap = Model(0.0), observation.FrameTap()

        def predict(samples):
            model.frames.append(samples.tobytes())
            return {"jarvis": 0.99 if samples[0] == 257 else 0.0}

        model.predict = predict
        model.reset = lambda: None
        monkeypatch.setattr(observation, "ENABLED", True)
        monkeypatch.setattr(controller, "_acquire_wake_model", AsyncMock(return_value=model))
        monkeypatch.setattr(controller.voice_backend, "can_serve_turn", lambda _: True)
        monkeypatch.setattr(controller, "_ringing_devices", lambda: [])
        started, delivered = asyncio.Event(), asyncio.Event()

        async def voice(device, **kwargs):
            assert kwargs["trigger_label"] == "wakeword(0.990)"
            assert device.oww_paused.is_set()
            started.set()
            assert (await device.voice_queue.get())[0] == 2
            delivered.set()
            await asyncio.Future()

        monkeypatch.setattr(controller, "_run_voice_locked", voice)
        for i in range(16):
            device.mic_queue.put_nowait(wire(tap, i, 0))
        device.mic_queue.put_nowait(wire(tap, 16, 1))
        listener = asyncio.create_task(controller._stream_listen(device))
        try:
            await asyncio.wait_for(started.wait(), 2)
            device.voice_queue.put_nowait(wire(tap, 17, 2, listening=False))
            await asyncio.wait_for(delivered.wait(), 2)
            await observation.ready(model)
            assert [r["event"] for r in records] == ["observation_start", "observation"]
            assert records[-1]["verdict"] == "would_reject"
        finally:
            listener.cancel()
            await asyncio.gather(listener, return_exceptions=True)

    asyncio.run(run())
