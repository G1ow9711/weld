"""Prepare only the preregistered 16 training sources, preserving canonical rows."""
import json
import shutil
import time
from pathlib import Path

import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.canonical_horizontal_patch_heatmaps import canonical_frame
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.joint_relation_heatmaps import LOCAL_CHANNELS
from weld_pipeline.workpiece_relative_context import WorkpieceRelativeContext, VERSION

BASE = Path('weld_pipeline/data/joint_relation_v1')
OUT = Path('weld_pipeline/data/joint_workpiece_context_pilot154')


def main():
    if OUT.exists():
        raise ValueError('Preserve joint workpiece context arrays')
    cfg = json.loads((BASE/'config.json').read_text('utf8'))
    complete = json.loads((BASE/'complete.json').read_text('utf8'))
    names = cfg['folds'][0]['train_real']
    raw_path = Path('weld_pipeline/reports/workpiece_context_raw_153_20261001/complete.json')
    raw = json.loads(raw_path.read_text('utf8'))
    if not raw['all_checks_passed'] or set(names)&set(cfg['folds'][0]['held_out_real']):
        raise ValueError('Pure-PCD feature proof/source partition invalid')
    for source, digest in raw['source_sha256'].items():
        if sha256(source) != digest:
            raise ValueError('Context proof source changed')
    OUT.mkdir(parents=True)
    sources = [__file__, 'weld_pipeline/workpiece_relative_context.py',
        'weld_pipeline/canonical_horizontal_patch_heatmaps.py', 'weld_pipeline/curved_cache.py',
        'weld_pipeline/joint_relation_heatmaps.py', 'training/dataio.py']
    config = dict(version=VERSION, pieces=names, held=cfg['folds'][0]['held_out_real'],
        base_config_sha256=sha256(BASE/'config.json'), base_complete_sha256=sha256(BASE/'complete.json'),
        raw_context_proof_sha256=sha256(raw_path), source_sha256={p:sha256(p) for p in sources},
        added_point_channels=22, added_node_channels=22, combined_point_channels=97, combined_node_channels=255,
        feature_inputs='Raw PCD XYZ + canonical contact geometry only', labels_used_for_context=False,
        original_feature_and_target_arrays_not_modified=True, observed_check_read=False, full_contract_pass=False)
    (OUT/'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf8')
    snapshot = OUT/'source_snapshot'; snapshot.mkdir()
    for source in sources:
        shutil.copy2(source, snapshot/Path(source).name)
    manifests = {}; started = time.perf_counter()
    for name in names:
        manifest = json.loads((BASE/name/'manifest.json').read_text('utf8'))
        if sha256(BASE/name/'manifest.json') != complete['manifests'][name]:
            raise ValueError('Original relation manifest changed')
        for file, digest in manifest['files'].items():
            if sha256(BASE/name/file) != digest:
                raise ValueError('Original relation arrays changed')
        piece = load_piece(manifest['source_cache_root'], verify=True)
        pcd = Path(piece['record']['input'])
        if (piece['cache_manifest_sha256'] != manifest['original_cache_manifest_sha256']
                or sha256(pcd) != manifest['pcd_sha256']):
            raise ValueError('Raw training source changed')
        xyz = read_xyz_pcd(pcd)*1000.; context = WorkpieceRelativeContext(xyz)
        base_features = np.load(BASE/name/'features.npy', mmap_mode='r')
        root = OUT/name; root.mkdir()
        point = np.lib.format.open_memmap(root/'point_context.npy', mode='w+', dtype=np.float32,
            shape=(len(base_features), 22))
        nodes = np.empty((len(manifest['entries']), 22), np.float32)
        probes = []
        for entry in manifest['entries']:
            sequence, _, transform = canonical_frame(piece['sequences'][entry['sequence']])
            a, b = entry['start'], entry['stop']
            if transform != entry['transform'] or not np.array_equal(sequence['features'][:, LOCAL_CHANNELS], base_features[a:b]):
                raise ValueError('Canonical training query/frame order changed')
            point[a:b] = context.describe_sequence(sequence, np.arange(len(sequence['xyz'])))
            mid = (sequence['xyz'][0].astype(float)+sequence['xyz'][-1].astype(float))/2.
            normals = sequence['features'][len(sequence['xyz'])//2, 60:66].reshape(2, 3)
            nodes[entry['node']] = context.describe(mid, normals, sequence['direction'])
            for i in np.unique([0, len(sequence['xyz'])//2, len(sequence['xyz'])-1]):
                # Recompute query-by-query to catch dense versus sparse preparation errors.
                reference = context.describe_sequence(sequence, [i])[0]
                error = float(np.abs(point[a+i]-reference).max())
                if error > 2e-6:
                    raise ValueError('Dense/sparse contact context mismatch')
                probes.append(error)
        point.flush(); del point
        np.save(root/'node_context.npy', nodes)
        record = dict(piece=name, rows=len(base_features), nodes=len(nodes), input=str(pcd), input_sha256=sha256(pcd),
            base_manifest_sha256=sha256(BASE/name/'manifest.json'),
            files={p:sha256(root/p) for p in ('point_context.npy', 'node_context.npy')},
            canonical_query_order_exact=True, canonical_base_features_exact=True,
            dense_sparse_probe_count=len(probes), dense_sparse_max_error=max(probes),
            context_labels_used=False, elapsed_seconds=time.perf_counter()-started)
        (root/'manifest.json').write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf8')
        manifests[name] = sha256(root/'manifest.json')
        print('JOINT_WORKPIECE_CONTEXT_DATA', name, len(base_features), 'NODES', len(nodes), flush=True)
        del context, piece, base_features, xyz, nodes
    (OUT/'complete.json').write_text(json.dumps(dict(config_sha256=sha256(OUT/'config.json'), manifests=manifests,
        elapsed_seconds=time.perf_counter()-started, preparation_scope='16 original fold0 training sources only',
        held_context_not_prepared=True, observed_check_read=False, full_contract_pass=False),
        indent=2, ensure_ascii=False), encoding='utf8')
    print('JOINT_WORKPIECE_CONTEXT_DATA_COMPLETE', len(manifests), flush=True)


if __name__ == '__main__':
    main()
