"""Audited customer curves and reproducible geometric supervision, in world mm."""
import hashlib
import json
from pathlib import Path
import numpy as np
from openpyxl import load_workbook
from scipy.spatial import cKDTree


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read_curves(path):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    assert data['unit'] == 'millimeter'
    assert data['matrixConvention'] == 'row_major_column_vector'
    assert data['coordinateFrame'] == 'reference_workpiece'
    matrix = np.asarray(data['exportToWorld'], dtype=np.float64)
    assert np.allclose(matrix[3], [0, 0, 0, 1])
    assert np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-7)
    starts, ends, instance, samples, sample_segment, welds = [], [], [], [], [], []
    for wi, w in enumerate(data['welds']):
        welds.append({'id': w['id'], 'name': w['name'], 'segments': []})
        for s in w['segments']:
            if s['kind'] != 'line':
                raise ValueError('This pilot accepts audited straight segments only')
            p = np.asarray(s['points'], dtype=np.float64)
            assert len(p) >= 2 and np.isfinite(p).all()
            p = p @ matrix[:3, :3].T + matrix[:3, 3]
            a, b = p[0], p[-1]
            ab = b - a
            length = np.linalg.norm(ab)
            assert length > 1e-5
            t = np.clip((p-a) @ ab / (length**2), 0, 1)
            assert np.linalg.norm(p-a-t[:, None]*ab, axis=1).max() < 0.02
            assert abs(length - s['lengthMm']) < 0.02
            dense = a + np.linspace(0, 1, int(np.ceil(length))+1)[:, None] * ab
            si = len(starts)
            starts.append(a); ends.append(b); instance.append(wi)
            samples.append(dense)
            sample_segment.append(np.full(len(dense), si, dtype=np.int32))
            welds[-1]['segments'].append({'start_world_mm': a.tolist(), 'end_world_mm': b.tolist(), 'length_mm': float(length)})
    return {
        'matrix': matrix, 'starts': np.asarray(starts), 'ends': np.asarray(ends),
        'instance': np.asarray(instance, dtype=np.int32), 'samples': np.concatenate(samples),
        'sample_segment': np.concatenate(sample_segment), 'welds': welds,
    }


def read_excel(path):
    wb = load_workbook(path, read_only=True, data_only=True)
    rows = wb.active.iter_rows(values_only=True)
    heads = next(rows)
    def col(token):
        return next(i for i, h in enumerate(heads) if h and token in str(h))
    indices = [col('Source StartMatrix.Translation.'+a) for a in 'XYZ']
    end_indices = [col('Source EndMatrix.Translation.'+a) for a in 'XYZ']
    idi, ti, ni = col('ObjectId'), col('WeldType'), col('WeldNumber')
    values = {}
    for row in rows:
        if row[idi]:
            values[str(row[idi]).strip()] = {'number': row[ni], 'type': row[ti],
                'start': np.array([row[i] for i in indices], dtype=float),
                'end': np.array([row[i] for i in end_indices], dtype=float)}
    wb.close()
    return values


def point_labels(xyz, curves, positive_mm=4.0, background_mm=6.0):
    """Nearest sampled segments refined analytically; ambiguous boundaries ignored.

    Eight nearest 1 mm curve samples nominate line segments. This candidate
    approximation is checked against all segments on an independent random subset
    during preprocessing; full exact fallback is used for any discrepant cloud.
    """
    tree = cKDTree(curves['samples'])
    n = len(xyz)
    dist = np.empty(n, dtype=np.float32)
    inst = np.full(n, -1, dtype=np.int32)
    conflict = np.zeros(n, dtype=bool)
    for lo in range(0, n, 32768):
        q = xyz[lo:lo+32768]
        _, ix = tree.query(q, k=8, workers=2)
        si = curves['sample_segment'][ix]
        a, b = curves['starts'][si], curves['ends'][si]
        ab = b-a
        t = np.clip(np.sum((q[:, None]-a)*ab, axis=2)/np.sum(ab*ab, axis=2), 0, 1)
        ds = np.linalg.norm(q[:, None]-a-t[:, :, None]*ab, axis=2)
        best = np.argmin(ds, axis=1)
        dd = ds[np.arange(len(q)), best]
        ii = curves['instance'][si[np.arange(len(q)), best]]
        conflict[lo:lo+len(q)] = np.any((ds <= positive_mm) & (curves['instance'][si] != ii[:, None]), axis=1)
        dist[lo:lo+len(q)] = dd
        inst[lo:lo+len(q)] = ii
    # Candidate search is independently compared with brute force, not itself.
    rng = np.random.default_rng(107)
    check = rng.choice(n, size=min(2048, n), replace=False)
    exact = exact_distances(xyz[check], curves)
    max_error = float(np.abs(exact-dist[check]).max())
    fallback = max_error > 0.02
    if fallback:
        dist = exact_distances(xyz, curves).astype(np.float32)
    labels = np.full(n, -1, dtype=np.int8)
    labels[dist <= positive_mm] = 1
    labels[dist >= background_mm] = 0
    labels[conflict] = -1
    inst[labels != 1] = -1
    return labels, dist, inst, {'max_distance_error_checked_mm': max_error,
        'brute_force_checked_points': len(check), 'exact_distance_fallback': fallback,
        'instance_conflict_points': int(conflict.sum())}


def exact_distances(xyz, curves):
    out = np.full(len(xyz), np.inf)
    for a, b in zip(curves['starts'], curves['ends']):
        ab = b-a
        for lo in range(0, len(xyz), 65536):
            q = xyz[lo:lo+65536]
            t = np.clip(np.sum((q-a)*ab, axis=1)/np.dot(ab, ab), 0, 1)
            out[lo:lo+len(q)] = np.minimum(out[lo:lo+len(q)], np.linalg.norm(q-a-t[:, None]*ab, axis=1))
    return out
