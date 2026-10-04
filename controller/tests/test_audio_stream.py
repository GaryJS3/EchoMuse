"""Real HTTP and decoder checks for the behavior-preserving extraction."""
import asyncio
import contextlib
import io
import shutil
import wave

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

import em_audio_stream as audio


def wav_bytes(rate, pcm):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return stream.getvalue()


def test_wire_wav_passes_through_without_decoder(monkeypatch):
    pcm = b"\x00\x10" * 4000
    encoded = wav_bytes(48000, pcm)
    async def forbidden(*args, **kwargs):
        pytest.fail("wire WAV should not launch ffmpeg")
    monkeypatch.setattr(audio.asyncio, "create_subprocess_exec", forbidden)

    async def main():
        async def serve(request):
            response = web.StreamResponse()
            await response.prepare(request)
            for start in range(0, len(encoded), 501):
                await response.write(encoded[start:start + 501])
            await response.write_eof()
            return response
        app = web.Application()
        app.router.add_get("/audio", serve)
        async with TestServer(app) as server:
            async with contextlib.aclosing(audio._stream_tts_audio(str(server.make_url("/audio")))) as chunks:
                result = b"".join([chunk async for chunk in chunks])
        assert result == pcm
    asyncio.run(main())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_non_wire_wav_uses_existing_ffmpeg_resampler():
    encoded = wav_bytes(16000, b"\x00\x10" * 1600)
    async def main():
        async def serve(request):
            return web.Response(body=encoded, content_type="audio/wav")
        app = web.Application()
        app.router.add_get("/audio", serve)
        async with TestServer(app) as server:
            async with contextlib.aclosing(audio._stream_tts_audio(str(server.make_url("/audio")))) as chunks:
                result = b"".join([chunk async for chunk in chunks])
        assert len(result) == 4800 * 2  # 100ms at the speaker wire rate
    asyncio.run(main())


def test_http_error_retries_only_before_any_pcm():
    requests = []
    async def main():
        async def serve(request):
            requests.append(True)
            raise web.HTTPServiceUnavailable()
        app = web.Application()
        app.router.add_get("/audio", serve)
        async with TestServer(app) as server:
            with pytest.raises(Exception):
                async for _ in audio._stream_tts_audio(str(server.make_url("/audio"))):
                    pytest.fail("HTTP error yielded PCM")
        assert len(requests) == 2
    asyncio.run(main())
