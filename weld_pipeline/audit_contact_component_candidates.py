"""Full scoped child/endpoint coverage using observed contact components."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from training.labels import sha256
from weld_pipeline.contact_component_event_candidates import VERSION, component_positions, physical_checks
from weld_pipeline.physical_event_candidates import closest_event

BASE = Path('weld_pipeline/reports/physical_event_candidates_161_20261002')
FLAGS = Path('weld_pipeline/reports/physical_event_process_flags_162_20261002')
OUT = Path('weld_pipeline/reports/contact_component_candidates_163_20261002')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def main():
    if OUT.exists():
        raise ValueError('Preserve contact-component coverage audit')
    checks = physical_checks(); cfg = json.loads((BASE/'config.json').read_text('utf8'))
    done = json.loads((BASE/'complete.json').read_text('utf8'))
    flag_done = json.loads((FLAGS/'complete.json').read_text('utf8'))
    if sha256(FLAGS/'endpoint_rows.json') != flag_done['rows_sha256']:
        raise ValueError('Original endpoint diagnostic changed')
    endpoint_rows = json.loads((FLAGS/'endpoint_rows.json').read_text('utf8'))
    by_piece = defaultdict(list)
    for row in endpoint_rows:
        by_piece[row['piece']].append(row)
    OUT.mkdir(parents=True)
    dump(OUT/'config.json', dict(version=VERSION, train=cfg['train'], physical_checks=checks,
        base_sha256=sha256(BASE/'complete.json'), flags_sha256=sha256(FLAGS/'complete.json'),
        matching_plane_absdot_min=.99999, plane_offset_tolerance_mm=.25, level_tolerance_mm=.015,
        original_targets_unchanged=True, workrange_planes_not_added=True, invented_cutback_not_added=True,
        geometry_only_proposal_function=True, sources={p:sha256(p) for p in
            [__file__, 'weld_pipeline/contact_component_event_candidates.py', 'weld_pipeline/physical_event_candidates.py']},
        observed_or_new_held_read=False, diagnostic_only=True, full_contract_pass=False))
    results = {}; all_rows = []
    for name in cfg['train']:
        audited_path = BASE/(name+'.json')
        if sha256(audited_path) != done['piece_receipts'][name]:
            raise ValueError('Original full coverage changed')
        audited = json.loads(audited_path.read_text('utf8'))
        relation = json.loads((Path('weld_pipeline/data/joint_relation_v1')/name/'manifest.json').read_text('utf8'))
        root = Path(relation['source_cache_root']); mf = root/'manifest.json'
        parent_path = Path('weld_pipeline/data/expanded_surface_v3')/name/'manifest.json'
        if sha256(mf) != audited['source_manifest_sha256'] or sha256(parent_path) != audited['parent_manifest_sha256']:
            raise ValueError('Frozen geometry chain changed')
        manifest = json.loads(mf.read_text('utf8')); parent = json.loads(parent_path.read_text('utf8'))
        if (sha256(root/'xyz.npy') != manifest['files']['xyz.npy']['sha256']
                or sha256(parent['record']['input']) != audited['input_sha256']):
            raise ValueError('Pure candidate/raw XYZ changed')
        xyz = np.load(root/'xyz.npy', mmap_mode='r', allow_pickle=False)
        extra = json.loads((root/'curve_metadata.json').read_text('utf8'))['sequences']
        geometry = parent['record']['geometry']; sequences = {}; pools = {}; stats = {}
        for i, entry in enumerate(manifest['sequences']):
            if entry['type'] != 'horizontal' or extra[i].get('curve_kind') == 'arc':
                continue
            seq = dict(xyz=xyz[entry['start']:entry['stop']], direction=entry['direction'], type=entry['type'])
            sequences[i] = seq; pools[i], stats[i] = component_positions(seq, geometry)
        npz = OUT/(name+'_pools.npz'); np.savez_compressed(npz, **{str(i):p for i, p in pools.items()})
        children_by_key = {(row['parent_id'], row['segment']):row for row in audited['children']}
        child_coverage = defaultdict(list); rows = []
        for original in by_piece[name]:
            key = original['parent_id'], original['segment']; si = children_by_key[key]['sequence']
            score = closest_event(sequences[si], pools[si], np.asarray(original['point_mm'], float))
            row = dict(original, component_error_mm=score['error_mm'], component_covered_3mm=score['covered_3mm'])
            rows.append(row); all_rows.append(row); child_coverage[key].append(score['covered_3mm'])
        covered = sum(len(ends) == 2 and all(ends) for ends in child_coverage.values())
        result = dict(piece=name, input_sha256=audited['input_sha256'], source_manifest_sha256=sha256(mf),
            parent_manifest_sha256=sha256(parent_path), pool_sha256=sha256(npz),
            truth_children=len(child_coverage), covered_children=covered,
            truth_endpoints=len(rows), covered_endpoints=sum(r['component_covered_3mm'] for r in rows),
            candidate_positions=sum(len(p) for p in pools.values()), sequence_stats=stats,
            original_geometry_pool_covered=audited['coverage']['finite']['covered_children'],
            endpoints=rows, diagnostic_only=True, full_contract_pass=False)
        dump(OUT/(name+'.json'), result); results[name] = result
        print('CONTACT_COMPONENT_COVERAGE', name, covered, '/', result['truth_children'],
              'POINTS', result['covered_endpoints'], '/', len(rows), flush=True)
        del xyz, sequences, pools
    total = {key:sum(r[key] for r in results.values()) for key in
             ('truth_children', 'covered_children', 'truth_endpoints', 'covered_endpoints', 'candidate_positions')}
    if total['truth_children'] != 1221 or total['truth_endpoints'] != 2442:
        raise ValueError('All original children/endpoints must be retained')
    groups = defaultdict(list)
    for row in all_rows:
        for key in ('role', 'explicit_endpoint_workrange_split', 'parent_workrange_any', 'parent_dual_robot_any'):
            groups[key+'='+str(row[key])].append(row)
    grouped = {key:dict(total=len(rows), covered=sum(r['component_covered_3mm'] for r in rows),
        missing=sum(not r['component_covered_3mm'] for r in rows)) for key, rows in groups.items()}
    dump(OUT/'complete.json', dict(version=VERSION, config_sha256=sha256(OUT/'config.json'), totals=total,
        groups=grouped, original_IsSplitedForDualRobot_means_side_standing_plate_split=True,
        covered_child_fraction=total['covered_children']/1221.,
        piece_receipts={n:sha256(OUT/(n+'.json')) for n in results}, physical_checks=checks,
        diagnostic_not_training_or_contract_acceptance=True, pool_limited_runtime_adopted=False,
        observed_or_new_held_read=False, full_contract_pass=False))
    print('CONTACT_COMPONENT_COVERAGE_COMPLETE', total, grouped, flush=True)


if __name__ == '__main__':
    main()
