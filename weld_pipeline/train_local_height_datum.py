"""FIT-only physical height candidate supervision with fixed local CAD datum."""
import csv
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
from weld_pipeline.contact_plate_extent_model import physical_span_baseline
from weld_pipeline.curved_cache import load_piece
from weld_pipeline.finite_contact_plate_profiles import FiniteContactPlates, FEATURE_COUNT
from weld_pipeline.fixed_cad_height_datum import FixedContactDatum, VERSION
from weld_pipeline.prepare_contract_child import child_attributes
from weld_pipeline.train_candidate_tree import mapped_references
from weld_pipeline.train_endpoint_process_attributes import known_label


def records(split, names, out, role):
    x, y, supported, all_rows, coverage = [], [], [], [], {}
    for name in names:
        ref_path = Path('weld_pipeline/reports/step_contact_continuity_candidates_v1') / (name + '_attribute_candidates.json')
        if not ref_path.exists():
            coverage[name] = {'state': 'NO_CURRENT_CAD_REFERENCE'}
            continue
        ref = json.loads(ref_path.read_text('utf8'))
        if not ref['pose_verified_for_reference']:
            raise ValueError('Unverified CAD reference pose')
        mesh = next(Path(root) / (name + '_mesh.npz') for root in (
            'weld_pipeline/reports/step_reference_pose_repair_v1',
            'weld_pipeline/reports/all_step_references_20260930',
            'weld_pipeline/reports/step_alignment_20260930') if (Path(root) / (name + '_mesh.npz')).exists())
        with np.load(mesh) as m:
            pose = np.array(ref['alignment']['transform'])
            v = m['vertices'] @ pose[:3, :3].T + pose[:3, 3]
            datum = FixedContactDatum(v, m['triangles'], m['owners'], m['face_ids'])
        piece = load_piece(split['piece_roots'][name], verify=True)
        path = piece['record']['input']
        if sha256(path) != piece['record']['pcd_sha256']:
            raise ValueError('Original PCD hash changed')
        profile = FiniteContactPlates(read_xyz_pcd(path) * 1000.)
        mapping = {w['id']: i for i, refs in mapped_references(piece).items() for w in refs}
        parents = {w['id']: w for w in piece['truth']}
        counts = Counter(); local_rows = []
        for child in ref['children']:
            if child['type'] != 'horizontal':
                counts['vertical_height_board_role_unfrozen_children'] += 1
                continue
            weld = parents[child['weld_id']]; points = np.array(weld['segments'][child['child_index']]['points'])
            attrs = child_attributes(weld, child['child_index'])
            sequence = piece['sequences'][mapping[child['weld_id']]] if child['weld_id'] in mapping else None
            normals = (sequence['features'][len(sequence['features']) // 2, 60:66].reshape(2, 3)
                       if sequence is not None else None)
            for endpoint, index, field in (('start', 0, 'start_height'), ('end', -1, 'end_height')):
                counts['expected_horizontal_endpoints'] += 1
                sample = child['samples'][index]; q = np.array(sample['xyz_mm'])
                inward = points[-1] - points[0] if index == 0 else points[0] - points[-1]
                old = known_label(field, attrs[field])
                row = {'piece': name, 'parent': child['weld_id'], 'child': child['child_index'],
                    'endpoint': endpoint, 'xyz_mm': q.tolist(),
                    'old_height_label_mm': old['value'] if old else None,
                    'original_label_modified': False, 'reviewed_customer_truth': False}
                if len(sample['candidate_pairs']) != 1:
                    measurement = {'state': 'CONTACT_PAIR_NONUNIQUE'}
                else:
                    pair = sample['candidate_pairs'][0]
                    standing = [b for b in pair if abs(b['outward_normal_world'][2]) < .4]
                    bases = [b for b in pair if abs(b['outward_normal_world'][2]) >= np.cos(np.deg2rad(3.))]
                    if len(standing) != 1 or len(bases) != 1:
                        measurement = {'state': 'LOCAL_HEIGHT_BOARD_ROLE_UNRESOLVED'}
                    else:
                        row['standing_board'] = standing[0]; row['local_base_board'] = bases[0]
                        measurement = datum.height(standing[0], bases[0], q, inward)
                row['measurement'] = measurement; state = measurement['state']
                if state == 'LOCAL_CONTACT_HEIGHT_CANDIDATE' and normals is not None:
                    agreement = np.abs(normals @ np.array(row['standing_board']['outward_normal_world']))
                    if agreement.max() >= np.cos(np.deg2rad(3.)):
                        normal = normals[int(agreement.argmax())]
                        feature, observation = profile.observe(q, normal)
                        if feature[0] > 0:
                            row['pcd_normal'] = normal.tolist(); row['pcd_observation'] = observation
                            row['sample_index'] = len(x); x.append(feature); y.append(measurement['height_mm'])
                            supported.append(row); counts['supported_training_samples'] += 1
                        else:
                            row['pcd_state'] = 'NO_FINITE_PCD_CONTACT_FACE'
                    else:
                        row['pcd_state'] = 'PCD_NORMAL_NOT_ASSOCIATED_WITH_FIXED_FACE'
                if state == 'LOCAL_CONTACT_HEIGHT_CANDIDATE':
                    counts['cad_local_datum_resolved'] += 1
                    if old is not None:
                        counts['original_labels_comparable'] += 1
                        row['old_label_difference_mm'] = abs(old['value'] - measurement['height_mm'])
                        counts['original_labels_differ_over_3'] += row['old_label_difference_mm'] > 3.
                else:
                    counts[state] += 1
                local_rows.append(row); all_rows.append(row)
        coverage[name] = {'counts': dict(counts), 'pcd_sha256': piece['record']['pcd_sha256'],
                          'reference_sha256': sha256(ref_path), 'mesh_sha256': sha256(mesh)}
        (out / (role + '_datum_' + name + '.json')).write_text(json.dumps({'coverage': coverage[name],
            'rows': local_rows, 'user_height_definition': '实际接触板及局部底板基准',
            'scope': 'CAD-derived candidate audit, original labels unchanged'}, indent=2, ensure_ascii=False), encoding='utf8')
        print(role, 'LOCAL_DATUM', name, counts['cad_local_datum_resolved'], '/',
              counts['expected_horizontal_endpoints'], 'OLD_DIFF', counts['original_labels_differ_over_3'], flush=True)
    columns = []
    for r in all_rows:
        m = r['measurement']; s = m.get('section', {}); b = m.get('local_base', {})
        columns.append({'piece': r['piece'], 'parent': r['parent'], 'child': r['child'],
            'endpoint': r['endpoint'], 'original_height_mm': r['old_height_label_mm'],
            'local_datum_candidate_mm': m.get('height_mm'), 'difference_mm': r.get('old_label_difference_mm'),
            'height_direction': m.get('height_direction'), 'measurement_state': m['state'],
            'standing_plate': r.get('standing_board', {}).get('board_name'),
            'local_base_plate': r.get('local_base_board', {}).get('board_name'),
            'standing_projection_mm': s.get('projection', {}).get('distance_mm'),
            'local_base_projection_mm': b.get('distance_mm'),
            'review_status': 'NOT_REVIEWED_ORIGINAL_UNCHANGED'})
    if columns:
        with (out / (role + '_板高原标签与局部底板候选.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(columns[0])); writer.writeheader(); writer.writerows(columns)
    return np.asarray(x, np.float32).reshape(-1, FEATURE_COUNT), np.array(y), supported, coverage


def score(bundle, x, y, rows, coverage, role, out):
    prediction = physical_span_baseline(x) + bundle['span_residual'].predict(x)
    errors = abs(prediction - y)
    expected = sum(v.get('counts', {}).get('expected_horizontal_endpoints', 0) for v in coverage.values())
    groups = {endpoint: {'supported': 0, 'within_3': 0} for endpoint in ('start', 'end')}
    for row, error in zip(rows, errors):
        groups[row['endpoint']]['supported'] += 1; groups[row['endpoint']]['within_3'] += int(error <= 3.)
    summary = {'role': role, 'expected_horizontal_endpoints': expected, 'supported_endpoint_references': len(y),
        'candidate_within_3': int((errors <= 3.).sum()), 'endpoint_groups': groups, 'coverage': coverage,
        'max_supported_error_mm': float(errors.max()) if len(errors) else None,
        'physical_baseline_within_3': int((abs(physical_span_baseline(x) - y) <= 3.).sum()),
        'geometry_misses_included': False, 'vertical_and_curve_height_fields_verified': False,
        'customer_reviewed_truth': False, 'full_contract_pass': False}
    np.savez(out / (role + '_height_predictions.npz'), targets=y, predictions=prediction, errors=errors)
    (out / (role + '_summary.json')).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf8')
    print(role, 'LOCAL_HEIGHT_CANDIDATE', summary['candidate_within_3'], '/', expected, flush=True)
    return summary


def main():
    split_path = Path('weld_pipeline/data/full_available_fit_v1/split.json')
    split = json.loads(split_path.read_text('utf8')); names = [n for n in split['train'] if '__' not in n]
    out = Path('weld_pipeline/runs/local_height_datum_seed140'); out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        raise ValueError('Frozen local height run already exists')
    sources = [__file__, 'weld_pipeline/fixed_cad_height_datum.py', 'weld_pipeline/finite_contact_plate_profiles.py',
               'weld_pipeline/contact_plate_extent_model.py', 'weld_pipeline/prepare_contract_child.py',
               'weld_pipeline/train_candidate_tree.py', 'weld_pipeline/train_endpoint_process_attributes.py']
    config = {'version': VERSION, 'seed': 140, 'trees': 128, 'max_features': .75, 'min_leaf': 1,
        'features': FEATURE_COUNT, 'train': names, 'observed_check': split['validation'],
        'split_sha256': sha256(split_path), 'sources': {p: sha256(p) for p in sources},
        'user_height_definition': '实际接触板及局部底板基准', 'candidate_height_definition':
        'actual_contact_board_opposite_boundary_relative_to_its_same_joint_horizontal_base_face',
        'fixed_face_projection_max_mm': 3., 'projection_limit_basis': 'contract_geometry_tolerance_not_check_tuning',
        'endpoint_one_sided_query_mm': .01, 'original_labels_modified': False,
        'reference_review_status': 'NOT_REVIEWED', 'input': '39_pure_PCD_physical_features',
        'no_names_ids_absolute_xyz_or_CAD_runtime_features': True,
        'selection_policy': 'Fixed single training; FIT receipt before reading observed check references',
        'whole_contract_output_verified': False}
    (out / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf8')
    archive = out / 'source_snapshot'; archive.mkdir(exist_ok=True)
    for p in sources:
        shutil.copy2(p, archive / Path(p).name)
    started = time.perf_counter(); x, y, rows, coverage = records(split, names, out, 'FIT')
    if not len(y) or not np.isfinite(x).all():
        raise ValueError('No finite local height supervision')
    np.savez(out / 'FIT_training.npz', features=x, targets=y)
    counts = Counter(r['piece'] for r in rows); weights = [1. / counts[r['piece']] for r in rows]
    model = ExtraTreesRegressor(n_estimators=128, max_features=.75, min_samples_leaf=1, n_jobs=3, random_state=140)
    model.fit(x, y - physical_span_baseline(x), sample_weight=weights)
    bundle = {'span_residual': model, 'config': config}; joblib.dump(bundle, out / 'heads.joblib')
    fit = score(bundle, x, y, rows, coverage, 'FIT', out)
    (out / 'FIT_decision.json').write_text(json.dumps({'candidate_local_height_fit_gate':
        fit['candidate_within_3'] > .95 * fit['expected_horizontal_endpoints'],
        'original_labels_or_v4_replaced': False, 'check_used_for_decision': False}, indent=2), encoding='utf8')
    del x, y, rows
    x, y, rows, coverage = records(split, split['validation'], out, 'OBSERVED_CHECK')
    check = score(bundle, x, y, rows, coverage, 'OBSERVED_CHECK', out)
    receipt = {'elapsed_seconds': time.perf_counter() - started, 'fit_summary': fit,
        'observed_check_summary': check, 'heads_sha256': sha256(out / 'heads.joblib'),
        'original_labels_modified': False, 'not_integrated_into_v4': True,
        'customer_reviewed_truth': False, 'full_contract_pass': False}
    (out / 'complete.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding='utf8')
    print('LOCAL_HEIGHT_DATUM_COMPLETE_FULL_CONTRACT_NOT_ACHIEVED', flush=True)


if __name__ == '__main__':
    main()
