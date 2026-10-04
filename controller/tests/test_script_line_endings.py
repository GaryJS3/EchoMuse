"""Prevent Windows checkouts from distributing unbootable Android scripts."""

import ast
import asyncio
import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("function,payload", [
    ("_sync_start_script", "start_server.sh"),
    ("_sync_debloat", "echomuse-debloat.sh"),
])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_device_sync_sends_unix_script_and_verifies_its_digest(tmp_path, function, payload, newline):
    expected = b"#!/system/bin/sh\necho ready\n"
    (tmp_path / payload).write_bytes(expected.replace(b"\n", newline))
    # Exercise the real sync function without starting a controller or device.
    tree = ast.parse((ROOT / "controller/em_api.py").read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == function)
    shell = AsyncMock(side_effect=["SHELL_OK", "SCRIPT_SYNCED DEBLOAT_SYNCED"])
    transfer = AsyncMock(return_value="TRANSFER_OK")
    scope = {
        "PAYLOADS_DIR": tmp_path,
        "_SHELL_OK": "SHELL_OK",
        "DEBLOAT_SCRIPT_PATH": "/service.d/echomuse-debloat.sh",
        "_shell_run": shell,
        "_stream_file_to_device": transfer,
        "_push_log_event": AsyncMock(),
        "_debloat_packages": lambda: [],
        "asyncio": SimpleNamespace(sleep=AsyncMock()),
        "hashlib": hashlib,
        "log": Mock(),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "em_api.py", "exec"), scope)
    asyncio.run(scope[function](object(), "test-device"))
    transfer.assert_awaited_once()
    assert transfer.await_args.args[1] == expected
    # Integrity validation must use the transmitted bytes, not the disk bytes.
    assert hashlib.md5(expected).hexdigest() in shell.await_args.args[1]


def test_git_keeps_every_executable_script_in_unix_format():
    # Check Git's effective rules for every tracked shebang, including em-wifi
    # which has no extension. This also holds when core.autocrlf is enabled.
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    scripts = [name for name in tracked if name and (ROOT / name).is_file()
               and ((ROOT / name).read_bytes().startswith(b"#!") or name.endswith(".sh"))]
    assert scripts
    rules = subprocess.check_output(
        ["git", "-c", "core.autocrlf=true", "check-attr", "-z", "eol", "--", *scripts],
        cwd=ROOT).decode().split("\0")
    for name, _, value in zip(rules[0::3], rules[1::3], rules[2::3]):
        assert value == "lf", f"{name} can be checked out with Windows line endings"
    for name in scripts:
        assert b"\r" not in (ROOT / name).read_bytes(), f"{name} contains CR bytes"
