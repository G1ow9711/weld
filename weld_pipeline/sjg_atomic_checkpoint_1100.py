"""Publish checkpoints atomically on their destination volume."""
import os
import tempfile
from pathlib import Path


def atomic_binary_write(path, writer):
    """Keep an existing checkpoint intact until serialization and flush finish.

    A hard process stop can leave a uniquely named temporary file, but readers
    continue to see the previous complete checkpoint. This does not provide a
    training watchdog or make the surrounding training loop resumable.
    """
    path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".pending", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        # This path belongs only to this invocation, never to another writer.
        temporary.unlink(missing_ok=True)
