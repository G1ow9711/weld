"""Train physical local extents from offline CAD; keep original labels intact."""
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.cad_contact_height_sections import ContactFaceSections
from weld_pipeline.contact_plate_extent_model import VERSION, physical_span_baseline
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.finite_contact_plate_profiles import FiniteContactPlates, FEATURE_COUNT
from weld_pipeline.prepare_contract_child import child_attributes
from weld_pipeline.train_candidate_tree import mapped_references
from weld_pipeline.train_endpoint_process_attributes import known_label


def records(split, names, out, role):
    features, values, rows, coverage = [], [], [], {}
    for name in names:
        reference = Path('weld_pipeline/reports/step_contact_continuity_candidates_v1') / (name + '_attribute_candidates.json')
        if not reference.exists():
            coverage[name] = {'state': 'NO_THIS_ROUND_CAD_REFERENCE'}
            continue
        ref = json.loads(reference.read_text('utf8'))
        if not ref['pose_verified_for_reference']:
            raise ValueError('Unverified CAD reference alignment')
        mesh = next((Path(root) / (name + '_mesh.npz') for root in (
            'weld_pipeline/reports/step_reference_pose_repair_v1',
            'weld_pipeline/reports/all_step_references_20260930',
            'weld_pipeline/reports/step_alignment_20260930')
            if (Path(root) / (name + '_mesh.npz')).exists()), None)
        if mesh is None:
            raise ValueError('Aligned reference mesh missing')
        with np.load(mesh) as m:
            pose = np.array(ref['alignment']['transform'])
            vertices = m['vertices'] @ pose[:3, :3].T + pose[:3, 3]
            sections = ContactFaceSections(vertices, m['triangles'], m['owners'])
        piece = load_piece(split['piece_roots'][name], verify=True)
        input_path = piece['record']['input']
        if sha256(input_path) != piece['record']['pcd_sha256']:
            raise ValueError('Original XYZ source changed')
        profiles = FiniteContactPlates(read_xyz_pcd(input_path) * 1000.)
        mapping = {w['id']: i for i, refs in mapped_references(piece).items() for w in refs}
        welds = {w['id']: w for w in piece['truth']}
        counts = Counter(); piece_rows = []
        for child in ref['children']:
            if child['weld_id'] not in mapping:
                counts['unmapped_contact_children'] += 1
                continue
            sequence = piece['sequences'][mapping[child['weld_id']]]
            normals = sequence['features'][len(sequence['features']) // 2, 60:66].reshape(2, 3)
            attrs = child_attributes(welds[child['weld_id']], child['child_index'])
            for endpoint, index, field in (('start', 0, 'start_height'), ('end', -1, 'end_height')):
                sample = child['samples'][index]
                if len(sample['candidate_pairs']) != 1:
                    counts['nonunique_endpoint_pair'] += 1
                    continue
                q = np.array(sample['xyz_mm'])
                old = known_label(field, attrs[field])
                for side, board in enumerate(sample['candidate_pairs'][0]):
                    ref_normal = np.array(board['outward_normal_world'])
                    if abs(ref_normal[2]) >= .4:
                        counts['horizontal_contact_face'] += 1
                        continue
                    counts['expected_nonhorizontal_endpoint_boards'] += 1
                    target = sections.section(int(board['volume']), q, ref_normal)
                    similarities = np.abs(normals @ ref_normal)
                    selected = normals[int(similarities.argmax())]
                    observation = None; feature = None; state = target['state']
                    if similarities.max() < np.cos(np.deg2rad(3.)):
                        state = 'PCD_CONTACT_NORMAL_UNRESOLVED'
                    elif state == 'finite_contact_face_section_candidate':
                        feature, observation = profiles.observe(q, selected)
                        if feature[0] == 0:
                            state = 'PCD_CONTACT_FACE_UNRESOLVED'
                    record = {'piece': name, 'parent': child['weld_id'], 'child': child['child_index'],
                        'endpoint': endpoint, 'board_side': side + 1, 'type': child['type'],
                        'xyz_mm': q.tolist(), 'pcd_normal': selected.tolist(), 'reference_volume': int(board['volume']),
                        'reference_board_name': board['board_name'], 'state': state,
                        'cad_section': target, 'pcd_observation': observation,
                        'old_height_label_mm': old['value'] if old else None,
                        'reviewed_customer_truth': False, 'base_datum_applied': False}
                    if state == 'finite_contact_face_section_candidate':
                        record['sample_index'] = len(features)
                        record['physical_baseline_mm'] = float(physical_span_baseline(feature))
                        record['old_label_to_span_difference_mm'] = (
                            abs(old['value'] - target['vertical_span_mm']) if old else None)
                        features.append(feature); values.append(target['vertical_span_mm']); rows.append(record)
                        counts['supported_reference_samples'] += 1
                    else:
                        counts[state] += 1
                    piece_rows.append(record)
        coverage[name] = {'counts': dict(counts), 'reference_sha256': sha256(reference),
            'mesh_sha256': sha256(mesh), 'pcd_sha256': piece['record']['pcd_sha256']}
        (out / (role + '_physical_' + name + '.json')).write_text(json.dumps({
            'coverage': coverage[name], 'rows': piece_rows, 'scope': 'candidate_not_contract_truth'},
            indent=2, ensure_ascii=False), encoding='utf8')
        print(role, 'PHYSICAL_ROWS', name, counts['supported_reference_samples'], '/',
              counts['expected_nonhorizontal_endpoint_boards'], flush=True)
    return np.asarray(features, np.float32).reshape(-1, FEATURE_COUNT), np.asarray(values, float), rows, coverage


def score(bundle, features, targets, rows, coverage, role, out):
    baseline = physical_span_baseline(features)
    prediction = baseline + bundle['span_residual'].predict(features)
    error = np.abs(prediction - targets)
    expected = sum(c.get('counts', {}).get('expected_nonhorizontal_endpoint_boards', 0) for c in coverage.values())
    counts = {endpoint: {'supported': 0, 'within_3': 0} for endpoint in ('start', 'end')}
    source = {}
    for row, got, difference in zip(rows, prediction, error):
        counts[row['endpoint']]['supported'] += 1
        counts[row['endpoint']]['within_3'] += int(difference <= 3.)
        piece = source.setdefault(row['piece'], {'supported': 0, 'within_3': 0, 'baseline_within_3': 0})
        piece['supported'] += 1; piece['within_3'] += int(difference <= 3.)
        piece['baseline_within_3'] += int(abs(row['physical_baseline_mm'] - row['cad_section']['vertical_span_mm']) <= 3.)
    summary = {'role': role, 'supported_reference_boards': len(rows), 'expected_reference_boards': expected,
        'within_3': int((error <= 3.).sum()), 'baseline_within_3': int((np.abs(baseline - targets) <= 3.).sum()),
        'endpoint_counts': counts, 'per_piece': source, 'coverage': coverage,
        'max_supported_error_mm': float(error.max()) if len(error) else None,
        'mean_supported_error_mm': float(error.mean()) if len(error) else None,
        'geometry_misses_included': False, 'local_base_datum_resolved': False,
        'reviewed_customer_truth': False, 'full_contract_pass': False}
    np.savez(out / (role + '_predictions.npz'), targets=targets, predictions=prediction,
             physical_baseline=baseline, errors=error)
    (out / (role + '_summary.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf8')
    print(role, 'EXTENT_CANDIDATE', summary['within_3'], '/', expected, 'BASELINE', summary['baseline_within_3'], flush=True)
    return summary


def main():
    split_path = Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split = json.loads(split_path.read_text('utf8'))
    names = [n for n in split['train'] if '__' not in n]
    out = Path('weld_pipeline/runs/contact_plate_extent_seed138'); out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        raise ValueError('Existing frozen contact extent run')
    sources = [__file__, 'weld_pipeline/cad_contact_height_sections.py',
        'weld_pipeline/finite_contact_plate_profiles.py', 'weld_pipeline/contact_plate_extent_model.py',
        'weld_pipeline/prepare_contract_child.py', 'weld_pipeline/train_candidate_tree.py',
        'weld_pipeline/train_endpoint_process_attributes.py']
    config = {'version': VERSION, 'seed': 138, 'trees': 128, 'max_features': .75, 'min_leaf': 1,
        'input_features': FEATURE_COUNT, 'train': names, 'observed_check': split['validation'],
        'split_sha256': sha256(split_path), 'sources': {p: sha256(p) for p in sources},
        'user_height_definition': '实际接触板及局部底板基准',
        'target': 'offline_CAD_finite_contact_face_local_vertical_outer_span_candidate',
        'original_labels_modified': False, 'runtime_input': 'XYZ_and_PCD_derived_contact_normal_only',
        'names_ids_or_absolute_coordinates_used_as_features': False,
        'datum_relation': 'local_base_candidates_recorded_separately_not_yet_frozen_as_final_height',
        'reference_review_status': 'NOT_REVIEWED', 'geometry_gate_verified': False, 'full_contract_pass': False}
    (out / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf8')
    archive = out / 'source_snapshot'; archive.mkdir(exist_ok=True)
    for path in sources:
        shutil.copy2(path, archive / Path(path).name)
    started = time.perf_counter()
    x, y, rows, coverage = records(split, names, out, 'FIT')
    if not len(y) or not np.isfinite(x).all():
        raise ValueError('No finite supported FIT extent supervision')
    np.savez(out / 'fit_training.npz', features=x, targets=y)
    counts = Counter(row['piece'] for row in rows)
    weights = np.array([1. / counts[row['piece']] for row in rows])
    model = ExtraTreesRegressor(n_estimators=128, max_features=.75, min_samples_leaf=1,
                               random_state=138, n_jobs=3)
    model.fit(x, y - physical_span_baseline(x), sample_weight=weights)
    bundle = {'span_residual': model, 'config': config}
    joblib.dump(bundle, out / 'heads.joblib')
    fit = score(bundle, x, y, rows, coverage, 'FIT', out)
    (out / 'fit_decision.json').write_text(json.dumps({'candidate_physics_fit_gate':
        fit['within_3'] > .95 * fit['expected_reference_boards'],
        'may_replace_contract_height': False, 'check_used_for_decision': False}, indent=2), encoding='utf8')
    del x, y, rows
    x, y, rows, coverage = records(split, split['validation'], out, 'OBSERVED_CHECK')
    check = score(bundle, x, y, rows, coverage, 'OBSERVED_CHECK', out)
    (out / 'complete.json').write_text(json.dumps({'elapsed_seconds': time.perf_counter() - started,
        'fit_summary': fit, 'observed_check_summary': check, 'heads_sha256': sha256(out / 'heads.joblib'),
        'original_labels_modified': False, 'local_base_datum_resolved': False,
        'candidate_not_reviewed': True, 'not_integrated_into_v4': True, 'full_contract_pass': False},
        indent=2, ensure_ascii=False), encoding='utf8')
    print('CONTACT_EXTENT_COMPLETE_FULL_CONTRACT_NOT_ACHIEVED', flush=True)


if __name__ == '__main__':
    main()
