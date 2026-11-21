"""Shadow the two-frame idle wake rule without holding up the accepted turn."""

import asyncio
import json
import logging
import math
import os
import time

import numpy as np

import em_db as db
import em_dbwriter
import em_oww_models
import em_tasks

log = logging.getLogger("echomuse.controller")
ENABLED = os.environ.get("ECHOMUSE_WAKE_TWO_FRAME_MODE", "observe").lower() == "observe"
FRAME_BYTES = 1280 * 2
# A reconnect can reuse a model while its previous listener is unwinding.
# Keep it exclusively owned until the extra prediction AND reset have finished.
_pending = {}


class FrameTap:
    """Pair consecutive 80 ms wire frames, including frames already queued.

    Only the current eligible wake packet gets a future. Its eventual result
    is plain PCM, so buffered packets cannot form a linked list. Nothing is
    taken from the voice queue, and closing a data socket ends its own tap.
    """

    def __init__(self):
        self.pending = None
        self.sequence = None

    def end(self, pcm=None):
        if self.pending is not None and not self.pending.done():
            self.pending.set_result(pcm)
        self.pending = None

    def feed(self, payload, pcm, sequence, listening):
        valid = len(pcm) == FRAME_BYTES
        adjacent = self.sequence is not None and sequence == (self.sequence + 1) % 65536
        self.end(pcm if valid and adjacent else None)
        self.sequence = sequence
        if valid and listening:
            self.pending = asyncio.get_running_loop().create_future()
            payload.next_wake_frame = self.pending


def _record(device_id, **fields):
    message = "wake_two_frame " + json.dumps(dict(mode="observe", **fields),
                                             separators=(",", ":"), allow_nan=False)
    log.info("[%s] %s", device_id, message)
    em_dbwriter.submit(db.log_device, device_id, "info", "controller", message)


def started(device_id, model, threshold):
    if ENABLED:
        _record(device_id, event="observation_start", model=model, threshold=threshold)


async def ready(model):
    if task := _pending.get(model):
        await asyncio.shield(task)


def after_crossing(device_id, model, model_key, threshold, score, payload, aligned, reset):
    """Start the observation now. The wake listener awaits the reset at its
    normal join point, *after* the voice turn or arbitration loss."""
    successor = getattr(payload, "next_wake_frame", None) if aligned else None
    candidate_ms = time.time_ns() // 1_000_000

    async def observe():
        second, reason = None, None
        try:
            pcm = await asyncio.wait_for(successor, 1.0) if successor is not None else None
            if pcm is None:
                reason = "noncontiguous_audio"
            else:
                prediction = await asyncio.to_thread(model.predict, np.frombuffer(pcm, dtype=np.int16))
                second = em_oww_models.scalar_score(prediction[model_key])
                if not math.isfinite(second):
                    raise ValueError("nonfinite prediction")
        except asyncio.TimeoutError:
            reason = "next_frame_timeout"
        except Exception:
            log.exception("[%s] Two-frame shadow prediction failed", device_id)
            second, reason = None, "prediction_failed"
        try:
            verdict = ("unavailable" if second is None else
                       "would_accept" if second >= threshold else "would_reject")
            _record(device_id, event="observation", model=model_key, threshold=threshold,
                    candidateTsMs=candidate_ms, firstScore=score, secondScore=second,
                    verdict=verdict, **({"reason": reason} if reason else {}))
        finally:
            await reset(model)

    task = em_tasks.spawn(observe(), name=f"wake-observation-{device_id}")
    _pending[model] = task
    task.add_done_callback(lambda done: _pending.pop(model, None))
    return asyncio.shield(task)
