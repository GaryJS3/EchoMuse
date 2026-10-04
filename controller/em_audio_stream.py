"""Shared URL-to-wire-PCM streaming decoder, extracted unchanged from ESPHome."""

import asyncio
import contextlib
import logging
from typing import AsyncIterator

import em_wav

# Preserve the original decoder's logging category as well as its behavior.
log = logging.getLogger("echomuse.esphome")
WIRE_RATE = 48000


async def _stream_tts_audio(url: str) -> AsyncIterator[bytes]:
    """
    Incrementally fetch and decode HA TTS audio to 48kHz mono S16_LE PCM.

    The HTTP response is piped into one long-lived ffmpeg process while decoded
    PCM is yielded immediately. Neither the encoded response nor decoded speech
    is accumulated in memory. A retry is safe only before PCM has been emitted;
    retrying later would repeat audio the user has already heard.
    """
    emitted = False
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            # aclosing, as below: a barge-in closes this generator, and the
            # inner one must tear ffmpeg down NOW, not when collected.
            async with contextlib.aclosing(_stream_tts_audio_once(url)) as once:
                async for pcm in once:
                    emitted = True
                    yield pcm
            return
        except Exception as err:
            last_exc = err
            if attempt == 0 and not emitted:
                log.warning(
                    f"_stream_tts_audio: stream failed before playback ({err}) "
                    "— retrying once"
                )
                await asyncio.sleep(0.5)
                continue
            raise

    raise last_exc


async def _stream_tts_audio_once(url: str) -> AsyncIterator[bytes]:
    """Run one HTTP streaming attempt for _stream_tts_audio.

    A WAV already at the wire format (what we declare to HA — see
    supported_formats) is passed straight through: no decoder, so nothing is
    held back when HA pauses between sentences (em_wav). Anything else is
    decoded by ffmpeg.
    """
    import aiohttp

    timeout = aiohttp.ClientTimeout(
        total=None,
        connect=10,
        sock_connect=10,
        sock_read=60,
    )
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            resp.raise_for_status()
            chunks = resp.content.iter_chunked(16 * 1024)
            head = bytearray()
            wav = em_wav.WavStream()
            pcm = b""
            try:
                async for chunk in chunks:
                    head += chunk
                    if len(head) >= 4 and head[:4] != b"RIFF":
                        break
                    pcm = wav.push(chunk)
                    if wav.ready:
                        break
            except em_wav.NotWav:
                pass

            if wav.ready and em_wav.is_wire_pcm(wav.format, WIRE_RATE):
                log.debug("TTS: WAV passthrough")
                odd = b""
                async for part in _prepend(pcm, chunks):
                    part = odd + part
                    cut = len(part) & ~1
                    odd = part[cut:]
                    if cut:
                        yield part[:cut]
                return

            log.info(f"TTS: decoding with ffmpeg ({wav.format or 'not WAV'})")
            # aclosing: a barge-in closes THIS generator, and an inner one left
            # to the garbage collector would run its kill-first teardown late,
            # leaving ffmpeg alive past the turn.
            async with contextlib.aclosing(
                    _ffmpeg_decode(_prepend(bytes(head), chunks))) as decoded:
                async for pcm in decoded:
                    yield pcm


async def _prepend(first: bytes, rest: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    if first:
        yield first
    async for chunk in rest:
        yield chunk


async def _ffmpeg_decode(encoded: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Decode any format ffmpeg reads to wire PCM, yielding as it decodes.

    `-threads 1`: frame-threaded decoding holds one frame per thread before it
    emits anything, which on an 8-core host kept ~1.7s of FLAC inside ffmpeg
    (0.9s with one thread) — audio that is then stranded whenever the input
    pauses. See em_wav for why TTS normally avoids this path altogether.
    """
    proc: asyncio.subprocess.Process | None = None
    feeder: asyncio.Task | None = None
    stderr_task: asyncio.Task | None = None
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-threads", "1",
            "-i", "pipe:0",
            "-f", "s16le", "-ar", str(WIRE_RATE), "-ac", "1",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def _feed_encoded_audio() -> None:
            assert proc is not None and proc.stdin is not None
            try:
                async for chunk in encoded:
                    proc.stdin.write(chunk)
                    await proc.stdin.drain()
            finally:
                if not proc.stdin.is_closing():
                    proc.stdin.close()
                    await proc.stdin.wait_closed()

        feeder = asyncio.create_task(_feed_encoded_audio())

        # stderr is drained CONCURRENTLY, not after stdout EOF. A pipe
        # holds ~64KB; if ffmpeg fills it, it blocks writing stderr,
        # stops producing stdout, and the reader below waits forever.
        # -loglevel error keeps that rare, but the case that produces
        # lots of stderr is a malformed or truncated response — exactly
        # the degraded case this path has to survive.
        assert proc.stderr is not None
        stderr_task = asyncio.create_task(proc.stderr.read())

        assert proc.stdout is not None
        while pcm := await proc.stdout.read(16 * 1024):
            yield pcm

        await feeder
        err = await stderr_task
        return_code = await asyncio.wait_for(proc.wait(), timeout=15.0)
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg streaming decode failed: {err.decode()[:200]}"
            )
    finally:
        # Kill ffmpeg FIRST, before cancelling the feeder — the ordering is
        # what makes this teardown terminate at all.
        #
        # On a barge-in the consumer stops iterating this generator, so
        # nobody drains ffmpeg's stdout. Its stdout pipe fills, ffmpeg blocks
        # writing, and a blocked ffmpeg stops reading stdin. The feeder is
        # then stuck in drain(), and cancelling it runs its own finally,
        # which awaits `stdin.wait_closed()` — a flush that needs ffmpeg to
        # read and therefore never completes. The gather() below never
        # returns and the kill never happens: teardown hangs, holding the
        # turn. Killing first breaks both pipes, so those awaits raise
        # instead of waiting.
        #
        # It also fixes #252. The InvalidStateError comes from asyncio
        # resolving `_stdin_closed` twice — once when we close stdin, once
        # when the kill tears the transport down. Killing while stdin is
        # still open collapses that to one resolution, and the feeder's
        # close() then finds a transport already closing and skips.
        #
        # Measured over 20 runs of a standalone repro of this shape, with
        # nobody reading stdout: cancel-then-kill hung 20/20 and logged
        # InvalidStateError 20/20; kill-first, 0/20 and 0/20.
        #
        # On the success path proc.wait() has already returned, so
        # returncode is set and nothing is killed.
        if proc is not None and proc.returncode is None:
            proc.kill()
            await proc.wait()
        for t in (feeder, stderr_task):
            if t is not None and not t.done():
                t.cancel()
        await asyncio.gather(
            *[t for t in (feeder, stderr_task) if t is not None],
            return_exceptions=True,
        )
