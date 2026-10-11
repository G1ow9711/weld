"""Replace-shared readers and bounded retry for live coordinator snapshots."""
import json
import os
from pathlib import Path
import time
from uuid import uuid4
from .live_json_snapshot_281 import read_json_snapshot, read_json_when_available


def write_json_snapshot(path, value, *, retry_seconds=15):
    path = Path(path)
    temporary = path.with_name(path.name + ".%d.%s.tmp" % (os.getpid(), uuid4().hex))
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    deadline = time.monotonic() + retry_seconds
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.025)
