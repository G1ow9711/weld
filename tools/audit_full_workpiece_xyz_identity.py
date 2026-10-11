"""Audit exact XYZ point-multiset identities across all qualified versions.

Canonical sorting ignores PLY headers, colors and vertex order, retaining every
point and its multiplicity. It does not normalize pose or certify the absence
of rotated/cropped derivatives or historical development exposure.
"""
from collections import defaultdict
import fcntl
import gc
import hashlib
import os
from pathlib import Path
import time
import traceback

import numpy as np

from tools.qualify_full_dataset_labels import DATA, ROOT, BASE, ply_points
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

OUT = BASE / 'workpiece_xyz_identity_v1'


def canonical_digest(xyz):
    xyz = np.array(xyz, dtype='<f8', copy=True, order='C')
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Finite float64 XYZ points are required')
    xyz[xyz == 0] = 0.0  # Geometrically identical +0/-0 have one representation.
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    ordered = np.ascontiguousarray(xyz[order])
    h = hashlib.sha256(b'EXACT_XYZ_MULTISET_FLOAT64_LE_V1\0')
    h.update(np.asarray(ordered.shape, dtype='<u8').tobytes())
    h.update(memoryview(ordered))
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    rows_dir = OUT / 'rows'
    rows_dir.mkdir(exist_ok=True)
    with (OUT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        qualification_path = BASE / 'qualification_summary.json'
        qualification = read(qualification_path)
        qualification_sha, source_sha = digest(qualification_path), digest(__file__)
        # Admission checks prove order and PLY metadata cannot affect identity;
        # a translated point set remains different under this exact recipe.
        example = np.array([[0., 2., 3.], [-0., 2., 3.], [4., 5., 6.]])
        assert canonical_digest(example) == canonical_digest(example[[2, 0, 1]])
        assert canonical_digest(example) != canonical_digest(example + [1., 0., 0.])
        assert canonical_digest(example) != canonical_digest(example[:2])
        by_file = defaultdict(list)
        for version in qualification['qualified']:
            by_file[version['source_files'][version['workpiece_source']]].append(version)
        rows = []
        started = time.perf_counter()
        for file_sha, versions in sorted(by_file.items()):
            destination = rows_dir / (file_sha + '.json')
            if destination.exists():
                row = read(destination)
                if row['qualification_sha256'] != qualification_sha or row['source_sha256'] != source_sha:
                    raise ValueError('XYZ identity resume binding changed')
            else:
                first = versions[0]
                source = DATA / first['workpiece_source']
                if digest(source) != file_sha:
                    raise ValueError('Qualified raw workpiece changed during the XYZ audit')
                xyz = ply_points(source)
                if any(v['workpiece_points'] != len(xyz) for v in versions):
                    raise ValueError('Byte-identical workpieces have inconsistent point counts')
                row = dict(workpiece_sha256=file_sha, points=len(xyz),
                    xyz_multiset_sha256=canonical_digest(xyz),
                    qualification_sha256=qualification_sha, source_sha256=source_sha,
                    versions=[dict(key=v['key'], family_id=v['family_id'], split=v['split']) for v in versions],
                    pose_normalized=False, point_order_ignored=True, point_multiplicity_preserved=True)
                write(destination, row)
                del xyz
                gc.collect()
            rows.append(row)
            write(OUT / 'progress.json', dict(status='AUDITING_EXACT_XYZ_MULTISET_IDENTITIES',
                controller_pid=os.getpid(), completed_unique_workpieces=len(rows),
                registered_unique_workpieces=len(by_file), qualified_versions=len(qualification['qualified']),
                fit_calls=0, predictions_scored=0, unix_time=time.time()))
            print('XYZ_IDENTITY_AUDITED', len(rows), '/', len(by_file), file_sha[:16], row['points'], flush=True)
        identities = defaultdict(list)
        for row in rows:
            identities[row['xyz_multiset_sha256']].append(row)
        duplicates, conflicts = [], []
        for sha, group in identities.items():
            versions = [v for row in group for v in row['versions']]
            if len(group) > 1:
                duplicates.append(dict(xyz_multiset_sha256=sha,
                    workpiece_sha256=[row['workpiece_sha256'] for row in group], versions=versions))
            if len({v['split'] for v in versions}) > 1:
                conflicts.append(dict(xyz_multiset_sha256=sha, versions=versions))
        result = dict(status='CROSS_SPLIT_EXACT_XYZ_IDENTITIES_FOUND' if conflicts else
            'PASS_NO_CROSS_SPLIT_EXACT_XYZ_IDENTITIES_NOT_INDEPENDENCE_CERTIFICATE',
            qualified_versions=len(qualification['qualified']), unique_file_workpieces=len(rows),
            unique_xyz_multisets=len(identities), different_file_same_xyz=duplicates, cross_split_conflicts=conflicts,
            quarantined_versions_not_used_by_models=len(qualification['quarantined']),
            point_multiplicity_preserved=True, coordinates_transformed=False, header_color_and_order_ignored=True,
            rotated_or_cropped_derivatives_not_certified=True, history_exposure_not_certified=True,
            independent_generalization_certified=False, qualification_sha256=qualification_sha,
            split_manifest_sha256=digest(BASE / 'split_manifest.json'), source_sha256=source_sha,
            fit_calls=0, model_predictions_scored=0, seconds=time.perf_counter() - started)
        destination = BASE / 'full_workpiece_xyz_identity_audit_v1.json'
        if destination.exists():
            raise ValueError('Immutable exact XYZ identity result already exists')
        write(destination, result)
        print(result['status'], 'conflicts', len(conflicts), 'nonbyte_duplicates', len(duplicates), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        OUT.mkdir(exist_ok=True)
        write(OUT / 'failure.json', dict(error=traceback.format_exc(), unix_time=time.time()))
        raise
