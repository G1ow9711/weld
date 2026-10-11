"""Read-only audit of existing data against the 2026-09-30 clarification."""
import argparse
import json
from collections import Counter
from pathlib import Path

from training.labels import sha256
from weld_pipeline.truth import excel_rows


def pcd_header(path):
    header = {}
    with Path(path).open('rb') as stream:
        for _ in range(40):
            line = stream.readline().decode('ascii').strip()
            if not line or line.startswith('#'):
                continue
            key, *values = line.split()
            header[key] = ' '.join(values)
            if key == 'DATA':
                break
        else:
            raise ValueError('No PCD DATA header: ' + str(path))
    return {k: header.get(k) for k in ['FIELDS', 'POINTS', 'DATA']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='weld_pipeline/data/partial_augmented_surface_v4')
    ap.add_argument('--outdir', default='weld_pipeline/reports/customer_definition_20260930')
    args = ap.parse_args()
    root, out = Path(args.data), Path(args.outdir)
    split = json.loads((root / 'split.json').read_text(encoding='utf-8'))
    original = split['augmentation_policy']['original_train']
    if set(original) & set(split['validation']):
        raise ValueError('Original TRAIN and DEV overlap')
    rows = []
    for name in original + split['validation']:
        folder = (root / split.get('piece_roots', {}).get(name, name)).resolve()
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        record, truths = manifest['record'], manifest['truth']
        pcd = Path(record['input'])
        if sha256(pcd) != record['pcd_sha256']:
            raise ValueError('Original point cloud changed: ' + name)
        for source, wanted in record['reference_hashes'].items():
            if sha256(source) != wanted:
                raise ValueError('Original reference changed: ' + source)
        excel = next(Path(p) for p in record['reference_hashes'] if p.endswith('.xlsx'))
        labels = excel_rows(excel)
        declared_segments = sum(int(r['SegmentCount']) for r in labels)
        reference_segments = sum(len(w['segments']) for w in truths)
        if {str(r['ObjectId']) for r in labels} != {str(w['id']) for w in truths}:
            raise ValueError('Excel identity and cached truth differ: ' + name)
        if declared_segments != reference_segments:
            raise ValueError('Declared and reference segment counts differ: ' + name)
        keys = sorted({k for r in labels for k in r})
        extra = [k for k in keys if any(q in k.lower() for q in ['thickness', 'normal', '板厚', '法向'])]
        rows.append({
            'group': name, 'role': 'training' if name in original else 'development',
            'cache_manifest_sha256': sha256(folder / 'manifest.json'),
            'input_pcd': str(pcd), 'input_sha256': record['pcd_sha256'],
            'pcd_header': pcd_header(pcd), 'reference_hashes': record['reference_hashes'],
            'parent_welds': len(truths), 'declared_segments': declared_segments,
            'available_geometric_segments': reference_segments,
            'multi_segment_parents': sum(len(w['segments']) > 1 for w in truths),
            'parent_counts_by_type': dict(Counter(w['type'] for w in truths)),
            'segment_counts_by_type': dict(Counter(w['type'] for w in truths for s in w['segments'])),
            'segment_counts_by_kind': dict(Counter(s['kind'] for w in truths for s in w['segments'])),
            'single_segment_vertical_parents': sum(w['type'] == 'vertical' and len(w['segments']) == 1 for w in truths),
            'thickness_or_normal_columns': extra,
            'blank_existing_process_values': dict(Counter(k for r in labels for k in [
                'FrontWallDistance', 'BackWallDistance', 'startHeight', 'endHeight',
                'StartViaHoleHeight', 'EndViaHoleHeight', 'StartBasePlateEdgeDistance',
                'EndBasePlateEdgeDistance', 'HasEave (Any)', 'IsSplitedForDualRobot (Any)'] if r.get(k) is None)),
            'ambiguous_wall_minus_one_count': sum(r.get(k) == -1 for r in labels for k in ['FrontWallDistance', 'BackWallDistance']),
            'scope': 'Geometry availability only. Existing parent attributes do not establish internal child-end attributes.',
        })
        print(json.dumps({k: rows[-1][k] for k in ['group', 'role', 'parent_welds', 'available_geometric_segments', 'single_segment_vertical_parents']}, ensure_ascii=False), flush=True)
    summaries = {}
    for role in ['training', 'development']:
        selected = [r for r in rows if r['role'] == role]
        summaries[role] = {'independent_groups': len(selected),
            **{k: sum(r[k] for r in selected) for k in ['parent_welds', 'available_geometric_segments', 'multi_segment_parents', 'single_segment_vertical_parents', 'ambiguous_wall_minus_one_count']},
            'segments_by_kind': dict(sum((Counter(r['segment_counts_by_kind']) for r in selected), Counter())),
            'groups_with_thickness_or_normal_columns': [r['group'] for r in selected if r['thickness_or_normal_columns']],
            'groups_with_arc_references': [r['group'] for r in selected if r['segment_counts_by_kind'].get('arc')],
            'groups_with_single_vertical_references': [r['group'] for r in selected if r['single_segment_vertical_parents']],
        }
    report = {'status': 'EXISTING_GEOMETRY_CAN_PROCEED', 'scope': 'Data readiness only; not extraction accuracy or full contract acceptance.',
        'data_split_sha256': sha256(root / 'split.json'), 'original_files_hash_verified': True,
        'original_supervision_policy': 'Customer confirms full annotation; unlabeled contacts in original supplied workpieces are non-welds.',
        'augmentation_supervision_policy': 'Derived partial-annotation variants keep unknown masks; customer full-label statement does not turn unknown augmented context into negatives.',
        'input_for_current_development': '_GW1_AllFaces.pcd; final deployment filename still pending.',
        'segment_count_status': 'Geometric segment inventory; final mapping to customer process-count units requires concrete example confirmation.',
        'performance_priority': 'Accuracy and completeness first. Runtime optimization deferred, not declared passed.',
        'summaries': summaries, 'groups': rows,
        'source_sha256': {str(p): sha256(p) for p in [Path(__file__), Path('weld_pipeline/truth.py'), Path('weld_pipeline/customer_label_corrections.py')]}}
    out.mkdir(parents=True, exist_ok=True)
    (out / 'data_readiness.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('READINESS', json.dumps(summaries, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
