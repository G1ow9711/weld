"""Fixed full-FIT coverage audit; no learned predictor or runtime label access."""
import json
import time
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.physical_event_candidates import (
    VERSION, candidate_positions, closest_event, physical_checks)

OUT = Path('weld_pipeline/reports/physical_event_candidates_161_20261002')
CONFIG = Path('weld_pipeline/runs/near_event_consistency_160/config.json')
RELATION = Path('weld_pipeline/data/joint_relation_v1')
PARENT = Path('weld_pipeline/data/expanded_surface_v3')
GEOMETRY_SOURCES = ('bounded_sequences.py', 'robust_surface_planes.py',
                    'point_frame.py', 'geometry_candidates.py')


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def main():
    if OUT.exists():
        raise ValueError('Preserve preregistered physical-event audit')
    checks = physical_checks(); cfg = json.loads(CONFIG.read_text('utf8'))
    OUT.mkdir(parents=True); start = time.perf_counter()
    protocol = dict(version=VERSION, train=cfg['train'], expected_truth_children=1221,
        finite_range_tolerance_mm=2., finite_z_tolerance_mm=.05, inner_padding_mm=20.,
        threshold_from_contract_mm=3., unlimited_supporting_plane_control_diagnostic_only=True,
        exact_duplicate_positions_only=True, full_grid_not_added=True,
        proposal_function_accepts_geometry_only=True, labels_only_for_offline_coverage=True,
        source_config_sha256=sha256(CONFIG), physical_checks=checks,
        sources={p:sha256(p) for p in [__file__, 'weld_pipeline/physical_event_candidates.py']},
        observed_or_new_held_read=False, diagnostic_only=True, full_contract_pass=False)
    dump(OUT/'config.json', protocol); results = {}
    for name in cfg['train']:
        relation = json.loads((RELATION/name/'manifest.json').read_text('utf8'))
        root = Path(relation['source_cache_root']); mf = root/'manifest.json'
        manifest = json.loads(mf.read_text('utf8'))
        parent_path = PARENT/name/'manifest.json'
        parent = json.loads(parent_path.read_text('utf8'))
        if (sha256(mf) != relation['original_cache_manifest_sha256']
                or sha256(parent_path) != manifest['provenance']['parent_manifest_sha256']
                or not manifest['record']['parent_geometry_exact']):
            raise ValueError('Frozen parent/mixed geometry chain changed: '+name)
        for file in GEOMETRY_SOURCES:
            path = Path('weld_pipeline')/file
            stored = parent['record']['source_hashes'][str(path)]
            if sha256(path) != stored:
                raise ValueError('Point-only geometry generator changed: '+str(path))
        pcd = Path(parent['record']['input']); pcd_sha = sha256(pcd)
        if pcd_sha != parent['record']['pcd_sha256'] or pcd_sha != manifest['record']['pcd_sha256']:
            raise ValueError('Raw input identity changed: '+name)
        xyz_path = root/'xyz.npy'
        if sha256(xyz_path) != manifest['files']['xyz.npy']['sha256']:
            raise ValueError('Frozen candidate XYZ changed: '+name)
        xyz = np.load(xyz_path, mmap_mode='r', allow_pickle=False)
        planes = parent['record']['geometry']['vertical_planes']
        sequences = []
        for entry in manifest['sequences']:
            sequences.append(dict(xyz=xyz[entry['start']:entry['stop']], direction=entry['direction'],
                                  type=entry['type'], sample_step_mm=entry['sample_step_mm']))
        # All pools are constructed without passing metadata, labels or mappings.
        pools = {}; stats = {}; arrays = {}; curve_metadata = json.loads((root/'curve_metadata.json').read_text('utf8'))
        for i, (seq, extra) in enumerate(zip(sequences, curve_metadata['sequences'])):
            if seq['type'] != 'horizontal' or extra.get('curve_kind') == 'arc':
                continue
            seq.update(extra)
            finite, counts = candidate_positions(seq, planes, finite=True)
            unlimited, _ = candidate_positions(seq, planes, finite=False)
            pools[i] = dict(finite=finite, unlimited=unlimited); stats[i] = counts
            arrays[str(i)+'_finite'] = finite; arrays[str(i)+'_unlimited'] = unlimited
        pool_path = OUT/(name+'_pools.npz'); np.savez_compressed(pool_path, **arrays)
        line_indices = [i for i, extra in enumerate(curve_metadata['sequences']) if extra.get('curve_kind') != 'arc']
        mapping = manifest['record']['mapping']['line_mapping']['weld_to_sequence']
        lines = [w for w in manifest['truth'] if all(s['kind'] == 'line' for s in w['segments'])]
        children = []; roles = []; misses = []
        for wi, weld in enumerate(lines):
            if weld['type'] != 'horizontal':
                continue
            si = line_indices[int(mapping[str(wi)])]; seq = sequences[si]
            for ci, segment in enumerate(weld['segments']):
                ends = np.asarray(segment['points'], float)[[0, -1]]
                row = dict(parent_id=weld['id'], segment=ci, sequence=si)
                for mode in ('finite', 'unlimited'):
                    scored = [closest_event(seq, pools[si][mode], p) for p in ends]
                    row[mode] = dict(both_ends_covered=all(s['covered_3mm'] for s in scored), endpoints=scored)
                children.append(row)
                if not row['finite']['both_ends_covered']:
                    misses.append(dict(row, reference_points=ends.tolist(), metadata=weld.get('metadata', {})))
            points = np.asarray([p for segment in weld['segments'] for p in
                                 (segment['points'][0], segment['points'][-1])], float)
            pos = (points-np.asarray(seq['xyz'][0], float))@seq['direction']
            for role, point in [('start', points[np.argmin(pos)]), ('end', points[np.argmax(pos)])]:
                roles.append(dict(role=role, **{mode:closest_event(seq, pools[si][mode], point)
                                                for mode in ('finite', 'unlimited')}))
            for segment in weld['segments'][:-1]:
                point = np.asarray(segment['points'][-1], float)
                roles.append(dict(role='cut', **{mode:closest_event(seq, pools[si][mode], point)
                                                for mode in ('finite', 'unlimited')}))
        coverage = {mode:dict(children=len(children), covered_children=sum(r[mode]['both_ends_covered'] for r in children),
                             covered_endpoints=sum(s['covered_3mm'] for r in children for s in r[mode]['endpoints']))
                    for mode in ('finite', 'unlimited')}
        for mode in coverage:
            coverage[mode]['covered_child_fraction'] = coverage[mode]['covered_children']/max(1, len(children))
        role_counts = {role:{mode:dict(total=sum(r['role'] == role for r in roles),
                    covered=sum(r['role'] == role and r[mode]['covered_3mm'] for r in roles))
                    for mode in ('finite', 'unlimited')} for role in ('start', 'end', 'cut')}
        result = dict(name=name, source_manifest_sha256=sha256(mf), parent_manifest_sha256=sha256(parent_path),
            input_sha256=pcd_sha, pool_sha256=sha256(pool_path), vertical_planes=len(planes),
            candidate_sequences=len(pools), candidate_positions=sum(len(p['finite']) for p in pools.values()),
            full_H_line_grid_points=sum(len(sequences[i]['xyz']) for i in pools), coverage=coverage,
            roles=role_counts, sequence_stats=stats, children=children, missing_children=misses,
            all_truth_children_retained=True, learned_model_used=False, diagnostic_only=True, full_contract_pass=False)
        dump(OUT/(name+'.json'), result); results[name] = result
        print('PHYSICAL_EVENT_COVERAGE', name, coverage, 'POOL', result['candidate_positions'], flush=True)
        del xyz, sequences, pools, arrays
    total = {mode:{key:sum(r['coverage'][mode][key] for r in results.values())
                   for key in ('children', 'covered_children', 'covered_endpoints')} for mode in ('finite', 'unlimited')}
    for mode, record in total.items():
        if record['children'] != 1221:
            raise ValueError('Full scoped child denominator changed')
        record['covered_child_fraction'] = record['covered_children']/record['children']
    roles = {role:{mode:{key:sum(r['roles'][role][mode][key] for r in results.values())
                       for key in ('total', 'covered')} for mode in ('finite', 'unlimited')}
             for role in ('start', 'end', 'cut')}
    receipt = dict(version=VERSION, config_sha256=sha256(OUT/'config.json'), coverage=total, roles=roles,
        candidate_positions=sum(r['candidate_positions'] for r in results.values()),
        full_H_line_grid_points=sum(r['full_H_line_grid_points'] for r in results.values()),
        piece_receipts={n:sha256(OUT/(n+'.json')) for n in results},
        physical_checks=checks, candidate_coverage_not_training_or_contract_acceptance=True,
        pure_pool_limited_runtime_adopted=False, observed_or_new_held_read=False,
        elapsed_seconds=time.perf_counter()-start, full_contract_pass=False)
    dump(OUT/'complete.json', receipt); print('PHYSICAL_EVENT_COVERAGE_COMPLETE', total, flush=True)


if __name__ == '__main__':
    main()
