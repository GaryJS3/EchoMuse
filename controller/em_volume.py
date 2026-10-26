"""Volume scale conversion between the device's native level and HA's float.

Split out as pure logic for the reason em_linkauth.py was: the three call
sites (em_controller, em_esphome, em_api) each had their OWN copy of
`level / 175`, so the scale lived in three places and none of them were
covered by a test. Changing the ceiling meant finding all three.

The device level is the raw tinymix ctl 61 index — the tlv320aic32x4 DAC
digital volume, 0.5dB per step with 0dB (unity) at index 127. The scale is
therefore dB-linear, which is roughly perceptually linear, so the mapping to
HA's 0.0–1.0 is a plain proportion and needs no extra taper.

DEVICE_VOLUME_MAX is 127 and NOT the control's own maximum of 175. Indexes
above 127 apply positive digital gain to near-full-scale PCM and saturate
inside the DAC — measured at 65% THD by index 153 (see
device/internal/server/volume.go for the full measurement and why stock
FireOS never touches this control).
"""

# Codec unity gain. The mixer control accepts up to 175; everything above
# this clips. See the module docstring.
DEVICE_VOLUME_MAX = 127

# 0.5dB per index step, 0dB at DEVICE_VOLUME_MAX.
DB_PER_STEP = 0.5


def level_to_db(level: int) -> float:
    """Device level as dB relative to unity. Index 127 -> 0.0, index 0 -> -63.5."""
    return (level - DEVICE_VOLUME_MAX) * DB_PER_STEP


def device_level_to_ha(level: int) -> float:
    """Convert a device volume level to an HA float (0.0–1.0)."""
    try:
        return max(0.0, min(1.0, float(level) / DEVICE_VOLUME_MAX))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def ha_volume_to_device(volume: float) -> int:
    """Convert an HA volume float (0.0–1.0) to a device volume level.

    Clamped to the codec's unity gain, so HA asking for full volume can never
    put the DAC into positive digital gain.
    """
    return max(0, min(DEVICE_VOLUME_MAX, round(float(volume) * DEVICE_VOLUME_MAX)))


# ─── The stock-sound profile's volume law (#229) ──────────────────────────────
#
# The plain law (0.5dB per level) lifted by STOCK_CEILING_DB, so full volume
# drives the chain's compressors and limiter. Stock FireOS's own table was
# tried first and dropped: it is front-loaded (most of its range is gone by
# 55%), and its +14.5dB above 70% bought little loudness, measured through
# our chain: past about +10dB, music gains under 0.5dB per dB of input and
# the rest is compression. Keeping 0.5dB per level also keeps every physical
# button press at 4dB and leaves HA, Music Assistant and stored startupVolume
# values meaning what they always have. The ceiling was picked by ear.
STOCK_CEILING_DB = 10.0


def stock_level_db(level: int) -> float | None:
    """Gain in dB for a device level under the stock-sound profile, or None
    for silence (level 0)."""
    if level <= 0:
        return None
    return level_to_db(min(level, DEVICE_VOLUME_MAX)) + STOCK_CEILING_DB
