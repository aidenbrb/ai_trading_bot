import msvcrt

from live_slc.process_lock import LockAlreadyHeld, acquire_process_lock


def test_acquire_and_release_is_reusable(tmp_path):
    lock_path = tmp_path / "test.lock"
    with acquire_process_lock(lock_path):
        pass
    with acquire_process_lock(lock_path):
        pass  # re-acquiring after release must succeed


def test_contention_is_detected_and_fails_immediately_not_blocking(tmp_path):
    lock_path = tmp_path / "test.lock"
    lock_path.write_bytes(b"0")
    handle = open(lock_path, "r+b")
    handle.seek(0)
    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    try:
        try:
            with acquire_process_lock(lock_path):
                raise AssertionError("acquired the lock while another handle already held it")
        except LockAlreadyHeld:
            pass  # expected - a manual duplicate launch must be refused, not queued
    finally:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        handle.close()


def test_lock_released_on_exception_inside_the_context():
    import tempfile
    from pathlib import Path
    lock_path = Path(tempfile.mkdtemp()) / "test.lock"
    try:
        with acquire_process_lock(lock_path):
            raise ValueError("simulated failure mid-cycle")
    except ValueError:
        pass
    # Must be reacquirable immediately - a crash inside the lock must not
    # leave it permanently held (an OS-level lock, unlike a plain lock
    # file, is released automatically when the holding handle closes).
    with acquire_process_lock(lock_path):
        pass
