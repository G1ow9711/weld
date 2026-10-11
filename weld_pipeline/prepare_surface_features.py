"""Extend the frozen pilot with point-derived features, keeping labels/split."""
import argparse
import json
import shutil
import time
from pathlib import Path

from training.dataio import read_xyz_pcd
from training.labels import sha256
from training.scripts.train_pilot import atomic_save
from weld_pipeline.train_typed_pilot import load
from weld_pipeline.surface_features import append_surface_features, FEATURE_VERSION
from weld_pipeline.bidirectional_surface_features import append_bidirectional_surface_features, FEATURE_VERSION as BOTH_VERSION


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='weld_pipeline/data/typed_pilot_v5')
    ap.add_argument('--outdir', default='weld_pipeline/data/surface_pilot_v1')
    ap.add_argument('--groups', nargs='+')
    ap.add_argument('--bidirectional', action='store_true')
    args = ap.parse_args()
    source, out = Path(args.source), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads((source/'split.json').read_text(encoding='utf-8'))
    feature_version = BOTH_VERSION if args.bidirectional else FEATURE_VERSION
    split.update(feature_version=feature_version, parent_split_sha256=sha256(source/'split.json'))
    split_path = out/'split.json'
    if split_path.exists() and json.loads(split_path.read_text(encoding='utf-8')) != split:
        raise ValueError('Existing frozen split differs')
    split_path.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding='utf-8')
    snapshot = out/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    files = [Path('weld_pipeline/surface_features.py'), Path(__file__)]
    if args.bidirectional:files.append(Path('weld_pipeline/bidirectional_surface_features.py'))
    for path in files:
        target = snapshot/path.name
        if target.exists() and sha256(path) != sha256(target):
            raise ValueError('Existing preprocessing differs')
        if not target.exists(): shutil.copy2(path, target)
    for name in args.groups or split['train']+split['validation']:
        if (out/(name+'.pt')).exists():
            print('EXISTING', name, flush=True)
            continue
        start = time.perf_counter()
        piece = load(source/(name+'.pt'))
        input_path = Path(piece['record']['input'])
        if sha256(input_path) != piece['record']['pcd_sha256']:
            raise ValueError('Raw point cloud changed')
        xyz = read_xyz_pcd(input_path)*1000.
        function = append_bidirectional_surface_features if args.bidirectional else append_surface_features
        sequences, stats = function(xyz, piece['sequences'])
        piece['sequences'] = sequences
        record = piece['record']
        record.update(feature_version=feature_version, feature_channels=84,
                      parent_cache_sha256=sha256(source/(name+'.pt')),
                      surface_features=stats, surface_feature_seconds=time.perf_counter()-start,
                      surface_source_hashes={str(p): sha256(p) for p in files},
                      surface_features_use_only_raw_points_and_point_derived_sequences=True)
        atomic_save(piece, out/(name+'.pt'))
        (out/(name+'.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')
        print('SURFACE_FEATURES', name, round(record['surface_feature_seconds'], 2), stats, flush=True)
        del piece, xyz, sequences


if __name__ == '__main__':
    main()
