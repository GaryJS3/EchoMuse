"""The narrow selector for accepted voice turns; ESPHome entities stay live."""

_name = "esphome"
_turn_backends = {}


def name() -> str:
    return _name


def configure(value: str) -> None:
    if value not in ("esphome", "external"):
        raise ValueError("voiceBackend must be esphome or external")
    global _name
    _name = value


def _backend():
    if _name == "external":
        import em_external_voice
        return em_external_voice.backend
    import em_esphome
    return em_esphome


def can_serve_turn(device_id: str) -> bool:
    return _backend().can_serve_turn(device_id)


async def trigger_voice_turn(*, ha_initiated=False, **kwargs) -> bool:
    if ha_initiated:
        # HA's explicit announce-then-listen/ask_question belongs to the HA
        # pipeline that requested it, independently of wake/button routing.
        import em_esphome
        backend = em_esphome
    else:
        backend = _backend()
    device_id = kwargs["device"].device_id
    _turn_backends[device_id] = backend
    try:
        return await backend.trigger_voice_turn(**kwargs)
    finally:
        _turn_backends.pop(device_id, None)


def cancel_voice_turn(device_id: str, abort_ha: bool = False,
                      reason: str = "cancelled") -> None:
    # A config change affects new turns. Cancellation still belongs to the
    # backend that owns this device's existing session.
    backend = _turn_backends.get(device_id) or _backend()
    backend.cancel_voice_turn(device_id, abort_ha=abort_ha, reason=reason)


def abort_ha_run(device_id: str) -> None:
    backend = _turn_backends.get(device_id) or _backend()
    if hasattr(backend, "abort_ha_run"):
        backend.abort_ha_run(device_id)
    else:
        backend.cancel_voice_turn(device_id, reason="barge_in")


async def record_dropped_wake(device, trigger_label: str, wake_info) -> None:
    # Reuse the existing unavailable cue and persistent diagnostics.
    import em_esphome
    await em_esphome.record_dropped_wake(device, trigger_label, wake_info)
