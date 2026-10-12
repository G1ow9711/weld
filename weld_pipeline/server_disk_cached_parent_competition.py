"""Reuse exact original relation values within one frozen TRAIN input.

Every requested edge remains in the output. The disk cache stores only values
produced by the unchanged bounded reciprocal prior, has explicit disk and RAM
limits, and is removed when its one-workpiece owner closes. It never contains
references or learned selector scores. It is not a cross-run prediction cache.
"""
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import struct
import tempfile

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from .server_bounded_lazy_parent_competition import BoundedLazyParentCompetition


def parent_fingerprint(parents):
    result = hashlib.sha256()
    for parent in parents:
        value = json.dumps(parent, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        result.update(len(value).to_bytes(8, 'little')); result.update(value)
    return result.hexdigest()


class DiskCachedParentCompetition(BoundedLazyParentCompetition):
    def __init__(self, parents, priors, neighbors=16, max_active_pairs=4_000_000,
                 feature_chunk_size=2048, max_cached_pairs=8192,
                 max_disk_cache_bytes=2 * 1024**3, reserve_disk_bytes=2 * 1024**3):
        if (type(max_disk_cache_bytes) is not int or max_disk_cache_bytes < 65536
                or type(reserve_disk_bytes) is not int or reserve_disk_bytes < 0
                or len(parents)**2 > np.iinfo(np.int64).max):
            raise ValueError('Explicit bounded disk storage and finite pair keys are required')
        frozen = getattr(priors, 'priors', priors)
        model = frozen.duplicate
        if not isinstance(model, ExtraTreesClassifier) or model.n_jobs != 1:
            raise ValueError('Exact reuse requires the original deterministic per-row tree prior')
        super().__init__(parents, priors, neighbors, max_active_pairs,
                         feature_chunk_size, max_cached_pairs)
        self.bound_parents_sha256 = parent_fingerprint(parents)
        self.bound_prior_sha256 = priors.checkpoint_sha256
        self.max_disk_cache_bytes = max_disk_cache_bytes
        self.reserve_disk_bytes = reserve_disk_bytes
        self.storage = tempfile.TemporaryDirectory(prefix='weld-frozen-parent-relations-')
        self.path = Path(self.storage.name) / 'relations.sqlite3'
        self.connection = sqlite3.connect(self.path)
        self.connection.execute('PRAGMA journal_mode=MEMORY')
        self.connection.execute('PRAGMA synchronous=OFF')
        self.connection.execute('PRAGMA cache_size=-16384')
        self.connection.execute('PRAGMA mmap_size=0')
        page_size = self.connection.execute('PRAGMA page_size').fetchone()[0]
        self.connection.execute(f'PRAGMA max_page_count={max_disk_cache_bytes // page_size}')
        self.connection.execute('CREATE TABLE relations (k INTEGER PRIMARY KEY, x BLOB NOT NULL, b BLOB NOT NULL)')
        self.connection.commit()
        self.disk_hits = self.computed_rows = self.stored_rows = self.skipped_store_rows = 0
        self.storage_full = False

    def verify_binding(self):
        if (self.priors.checkpoint_sha256 != self.bound_prior_sha256
                or parent_fingerprint(self.parents) != self.bound_parents_sha256):
            raise ValueError('A frozen input or its original relation prior changed')

    def features(self, pairs):
        pairs = np.asarray(pairs, np.int64).reshape(-1, 2)
        if (len(pairs) > self.max_active_pairs or np.any(pairs < 0)
                or np.any(pairs >= len(self.parents)) or np.any(pairs[:, 0] >= pairs[:, 1])
                or len(np.unique(pairs, axis=0)) != len(pairs)):
            raise ValueError('Unique valid unordered competition pairs within declared resources required')
        self.verify_binding()
        x = np.empty((len(pairs), 32), np.float32); base = np.empty(len(pairs), float)
        for start in range(0, len(pairs), self.feature_chunk_size):
            block = pairs[start:start + self.feature_chunk_size]
            keys = block[:, 0] * len(self.parents) + block[:, 1]
            found = {}
            for first in range(0, len(keys), 256):
                selected = [int(k) for k in keys[first:first + 256]]
                placeholders = ','.join('?' for _ in selected)
                found.update({k:(a,b) for k,a,b in self.connection.execute(
                    f'SELECT k,x,b FROM relations WHERE k IN ({placeholders})', selected)})
            missing = []
            for position, key in enumerate(keys):
                value = found.get(int(key))
                if value is None:
                    missing.append(position)
                else:
                    features, penalty = value
                    if len(features) != 128 or len(penalty) != 8:
                        raise ValueError('An exact cached relation has an invalid representation')
                    x[start + position] = np.frombuffer(features, np.float32)
                    base[start + position] = struct.unpack('<d', penalty)[0]
                    self.disk_hits += 1
            if missing:
                values, penalties = super().features(block[missing])
                x[start + np.asarray(missing)] = values
                base[start + np.asarray(missing)] = penalties
                self.computed_rows += len(missing)
                if self.storage_full or shutil.disk_usage(self.storage.name).free < self.reserve_disk_bytes + len(missing) * 256:
                    self.skipped_store_rows += len(missing)
                    continue
                rows = [(int(keys[position]), values[i].tobytes(), struct.pack('<d', float(penalties[i])))
                        for i,position in enumerate(missing)]
                try:
                    self.connection.executemany('INSERT OR REPLACE INTO relations VALUES (?,?,?)', rows)
                    self.connection.commit()
                    self.stored_rows += len(rows)
                except sqlite3.OperationalError as error:
                    self.connection.rollback()
                    if getattr(error, 'sqlite_errorcode', None) != sqlite3.SQLITE_FULL:
                        raise
                    self.storage_full = True
                    self.skipped_store_rows += len(rows)
        if self.path.stat().st_size > self.max_disk_cache_bytes:
            raise ValueError('The bounded relation storage limit was violated')
        return x, base

    def metrics(self):
        return dict(disk_hits=self.disk_hits, computed_rows=self.computed_rows,
            stored_rows=self.stored_rows, skipped_store_rows=self.skipped_store_rows,
            disk_bytes=self.path.stat().st_size, max_disk_cache_bytes=self.max_disk_cache_bytes,
            original_graph_and_all_output_edges_retained=True)

    def close(self):
        connection = getattr(self, 'connection', None)
        if connection is not None:
            connection.close(); self.connection = None
        storage = getattr(self, 'storage', None)
        if storage is not None:
            storage.cleanup(); self.storage = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
