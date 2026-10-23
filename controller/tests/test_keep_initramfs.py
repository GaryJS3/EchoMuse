"""em_emos_build.keep_initramfs: radar's kernel must not skip emOS's ramdisk.

The kernel is the user's own and differs per FireOS build (035T carries two
different ones), so these are built synthetically to the same shape: a
decompressor stub, one gzip stream, bytes after it.
"""

import gzip
import random
import struct
import zlib

import pytest

import em_emos_build as b

STUB = bytes(range(256)) * 4
TAIL = b"T" * 57          # 035T has 57 and 59 bytes after its stream


def _kernel(body=b"skip_initramfs\0", level=1):
    rnd = random.Random(7)
    words = [b"init", b"root=", b"console", b"Linux version 3.18.19", b"mtk"]
    filler = b" ".join(rnd.choice(words) for _ in range(30000))
    raw = filler[:200000] + body + filler[:100000]
    return STUB + gzip.compress(raw, compresslevel=level, mtime=0) + TAIL, raw


def _kernel_gunzip(stream):
    """What lib/decompress_inflate.c does: skip 10 bytes, then a NUL-ended
    name if FNAME is set, then raw deflate; the size is the last 4 bytes."""
    p = 10
    if stream[3] & 0x08:
        p = stream.index(b"\0", 10) + 1
    out = zlib.decompressobj(-15).decompress(stream[p:])
    assert struct.unpack("<I", stream[-4:])[0] == len(out)
    return out


def test_renames_the_parameter_in_the_same_space():
    z, raw = _kernel()
    out = b.keep_initramfs(z)
    assert len(out) == len(z)
    assert out[:len(STUB)] == STUB and out.endswith(TAIL)
    stream = out[len(STUB):-len(TAIL)]
    got = _kernel_gunzip(stream)
    assert got == raw.replace(b"skip_initramfs\0", b"keep_initramfs\0")


def test_an_already_patched_kernel_is_left_alone():
    # An emOS image rebuilt in place carries the kernel this already patched.
    z, _ = _kernel()
    once = b.keep_initramfs(z)
    assert b.keep_initramfs(once) == once


@pytest.mark.parametrize("body", [
    b"",                                       # no parameter at all
    b"skip_initramfs\0skip_initramfs\0",        # twice
    b"skip_initramfs\0keep_initramfs\0",        # half-patched
])
def test_refuses_a_kernel_it_does_not_understand(body):
    z, _ = _kernel(body)
    with pytest.raises(b.BuildError):
        b.keep_initramfs(z)


def test_refuses_what_is_not_a_kernel():
    with pytest.raises(b.BuildError):
        b.keep_initramfs(STUB + TAIL)                       # no stream
    with pytest.raises(b.BuildError):
        b.keep_initramfs(STUB + gzip.compress(b"skip_initramfs\0" * 1) + TAIL)
    z, _ = _kernel()
    with pytest.raises(b.BuildError):
        b.keep_initramfs(z[:len(z) // 2])                   # truncated stream


def test_refuses_when_the_recompressed_kernel_does_not_fit():
    # Compressed as hard as we recompress: no slack to pad the name with.
    z, _ = _kernel(level=9)
    with pytest.raises(b.BuildError, match="does not fit"):
        b.keep_initramfs(z)


def test_only_boards_whose_bootloader_skips_are_patched():
    assert b.SKIPS_INITRAMFS == {"radar"}
