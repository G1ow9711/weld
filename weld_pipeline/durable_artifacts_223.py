"""Durable local artifacts; telemetry failures never become solver certificates."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

VERSION = 'project_atomic_artifacts_and_phase_journal_223_v1'
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock(path: Path) -> threading.RLock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(path.resolve()), threading.RLock())


class ArtifactWriteError(OSError):
    def __init__(self, path: Path, temporary: Path, attempts: int, error: OSError):
        super().__init__(f'Cannot commit required artifact after {attempts} attempts: {path}; {error}')
        self.path = str(path)
        self.uncommitted_file = str(temporary)
        self.attempts = attempts
        self.original_error = error


class AtomicWriter:
    """Retry only bounded Windows access/sharing failures of os.replace.

    A failed commit retains the unique, explicitly uncommitted temporary file.
    Serialization, directory, write, flush and fsync errors are never suppressed.
    """
    def __init__(self, *, delays=(0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8),
                 replace: Callable | None = None):
        self.delays = tuple(float(delay) for delay in delays)
        if any(delay < 0 or not 0 <= delay <= 2 for delay in self.delays) or sum(self.delays) > 5:
            raise ValueError('Finite retry delays up to five seconds required')
        self.replace = replace if replace is not None else os.replace

    def write_bytes(self, path: str | Path, data: bytes) -> dict:
        path = Path(path)
        if not isinstance(data, bytes):
            raise TypeError('Exact bytes required')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.' + uuid.uuid4().hex + '.uncommitted')
        with _lock(path):
            with temporary.open('xb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            for attempt in range(len(self.delays) + 1):
                try:
                    self.replace(temporary, path)
                    return {'committed': True, 'path': str(path), 'replace_attempts': attempt + 1}
                except OSError as error:
                    retryable = getattr(error, 'winerror', None) in (5, 32, 33)
                    if not retryable or attempt == len(self.delays):
                        raise ArtifactWriteError(path, temporary, attempt + 1, error) from error
                    time.sleep(self.delays[attempt])
        raise AssertionError('Unreachable commit state')

    def write_json(self, path: str | Path, value) -> dict:
        encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
        return self.write_bytes(path, encoded)

    def commit_bytes(self, path: str | Path, data: bytes) -> dict:
        """A unique, required final artifact, read back before its receipt."""
        path = Path(path)
        with _lock(path):
            if path.exists():
                raise FileExistsError('Preserve immutable required artifact: ' + str(path))
            result = self.write_bytes(path, data)
            actual = path.read_bytes()
            expected_hash = hashlib.sha256(data).hexdigest()
            if len(actual) != len(data) or hashlib.sha256(actual).hexdigest() != expected_hash:
                raise ValueError('Committed artifact length/SHA readback differs')
            return {**result, 'immutable': True, 'bytes': len(data), 'sha256': expected_hash,
                    'readback_verified': True}

    def commit_json(self, path: str | Path, value) -> dict:
        encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
        return self.commit_bytes(path, encoded)


class ArtifactRecorder:
    """Required files fail closed; explicitly declared telemetry has a journal.

    Each recorder owns a unique journal namespace. Immutable events are never
    replaced or treated as liveness. A kernel-bound external observation is
    still needed for liveness, including after an application restart.
    """
    def __init__(self, root: str | Path, *, writer: AtomicWriter | None = None,
                 telemetry_names=('real_maxpath_progress.json',),
                 telemetry_suffixes=('_live_native_bounds.json',)):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.writer = writer or AtomicWriter()
        self.telemetry_names = frozenset(telemetry_names)
        self.telemetry_suffixes = tuple(telemetry_suffixes)
        self.namespace = str(os.getpid()) + '-' + uuid.uuid4().hex
        self.sequence = 0
        self.guard = threading.RLock()
        self.telemetry_failures: list[dict] = []

    def _event(self, kind: str, payload: dict) -> Path:
        self.sequence += 1
        event = {'version': VERSION, 'namespace': self.namespace,
                 'process_id': os.getpid(), 'sequence': self.sequence,
                 'UTC': datetime.now(timezone.utc).isoformat(), 'kind': kind, **payload}
        path = self.root / 'artifact_events' / self.namespace / f'{self.sequence:012d}.json'
        if path.exists():
            raise FileExistsError('Immutable event namespace collision')
        self.writer.write_json(path, event)
        return path

    def dump(self, path: str | Path, value) -> dict:
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Recorder must stay within this attempt directory')
        telemetry = path.name in self.telemetry_names or any(path.name.endswith(suffix) for suffix in self.telemetry_suffixes)
        if not telemetry:
            return self.writer.write_json(path, value)
        # Serialize once before any event. NaN/type errors are programming errors.
        encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
        with self.guard:
            event_path = None
            try:
                event_path = self._event('TELEMETRY_OBSERVATION', {'target': str(path), 'value': value})
                result = self.writer.write_bytes(path, encoded)
                return {**result, 'journal': str(event_path), 'telemetry_warning': False}
            except OSError as error:
                failure = {'target': str(path), 'error_type': type(error).__name__,
                           'error': str(error), 'uncommitted_file': getattr(error, 'uncommitted_file', None),
                           'observation_journal': str(event_path) if event_path else None,
                           'required_artifact': False, 'solver_status': 'NOT_INFERRED_FROM_TELEMETRY'}
                self.telemetry_failures.append(failure)
                try:
                    self._event('TELEMETRY_WRITE_WARNING', failure)
                except OSError as journal_error:
                    failure['warning_journal_error'] = str(journal_error)
                print('ARTIFACT_TELEMETRY_WARNING ' + json.dumps(failure, ensure_ascii=False),
                      file=sys.stderr, flush=True)
                return {'committed': False, 'path': str(path), 'telemetry_warning': True,
                        'journal': str(event_path) if event_path else None}

    def receipt(self) -> dict:
        with self.guard:
            return {'version': VERSION, 'journal_namespace': self.namespace,
                    'journal_events_written_or_attempted': self.sequence,
                    'telemetry_warning_count': len(self.telemetry_failures),
                    'telemetry_failures': list(self.telemetry_failures),
                    'liveness_from_journal': False, 'solver_completion_from_exit_code': False}
