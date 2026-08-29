"""
OS-level nonblocking process lock for live_slc.

Task Scheduler's own "don't start a new instance" setting only protects
against Task-Scheduler-triggered overlap, not a manually double-launched
process. This acquires a real OS file lock (released automatically if the
process dies, unlike a plain lock file) and fails immediately - never
blocks/waits - if another live_slc process already holds it.
"""
from __future__ import annotations

import msvcrt
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

LOCK_PATH = Path(__file__).parent / ".slc_live.lock"


class LockAlreadyHeld(RuntimeError):
    pass


@contextmanager
def acquire_process_lock(lock_path: Path = LOCK_PATH) -> Generator[None, None, None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure the file has >=1 byte to lock, via a separate open/close that
    # never touches a byte range another handle might already hold locked -
    # reading/writing through a handle that will be used for locking risks
    # a spurious PermissionError from the lock itself, not just from
    # msvcrt.locking().
    if not lock_path.exists() or lock_path.stat().st_size == 0:
        lock_path.write_bytes(b"0")
    handle = open(lock_path, "r+b")
    try:
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise LockAlreadyHeld(
                f"live_slc process lock already held: {lock_path} "
                "(another live_slc invocation is running - exiting, not waiting)"
            ) from exc
        try:
            yield
        finally:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
    finally:
        handle.close()
