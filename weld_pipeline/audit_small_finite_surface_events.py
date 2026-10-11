"""Full 1221-child audit of raw-PCD small-face event evidence."""
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from training.dataio import read_xyz_pcd
from training.labels import sha256
from weld_pipeline.physical_event_candidates import closest_event
from weld_pipeline.small_finite_surface_events import (VERSION, physical_checks,
    small_surface_planes, small_face_positions)

BASE = Path('weld_pipeline/reports/contact_component_candidates_163_20261002')
MAPPING = Path('weld_pipeline/reports/physical_event_candidates_161_20261002')
OUT = Path('weld_pipeline/reports/small_finite_surface_events_165_20261002')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')


def main():
    if OUT.exists():
        raise ValueError('Preserve fixed small-face audit')
    checks = physical_checks()
    config = json.loads((BASE/'config.json').read_text('utf8'))
    complete = json.loads((BASE/'complete.json').read_text('utf8'))
    mapped_complete = json.loads((MAPPING/'complete.json').read_text('utf8'))
    OUT.mkdir(parents=True); began = time.perf_counter()
    dump(OUT/'config.json', dict(version=VERSION, train=config['train'], all_children=1221,
        all_endpoints=2442, neighbor_scales=[8, 16, 32], raw_points_subsampled=False,
        PCA_first_to_second_eigenvalue_max=.05, PCA_second_eigenvalue_min=1e-4,
        vertical_normal_Z_max=.15, direction_alignment_min=.995, offset_vote_mm=.02,
        fitted_direction_alignment_min=.99995, plane_tolerance_mm=.12,
        surface_span_evidence_mm=.5, source_sha256={p:sha256(p) for p in [__file__,
            'weld_pipeline/small_finite_surface_events.py', 'weld_pipeline/contact_component_event_candidates.py',
            'weld_pipeline/physical_event_candidates.py']}, physical_checks=checks,
        base_complete_sha256=sha256(BASE/'complete.json'),
        original_targets_unchanged=True, workrange_or_cutbacks_not_added=True,
        no_proposal_is_declared_a_weld=True, continuous_decoder_not_restricted=True,
        observed_or_new_held_read=False, diagnostic_only=True, full_contract_pass=False))
    results = {}; all_rows = []
    for name in config['train']:
        source_path = BASE/(name+'.json'); mapping_path = MAPPING/(name+'.json')
        if sha256(source_path) != complete['piece_receipts'][name] or sha256(mapping_path) != mapped_complete['piece_receipts'][name]:
            raise ValueError('Fixed full FIT source or mapping changed')
        source = json.loads(source_path.read_text('utf8'))
        mapping = json.loads(mapping_path.read_text('utf8'))
        parent_path = Path('weld_pipeline/data/expanded_surface_v3')/name/'manifest.json'
        parent = json.loads(parent_path.read_text('utf8'))
        relation = json.loads((Path('weld_pipeline/data/joint_relation_v1')/name/'manifest.json').read_text('utf8'))
        root = Path(relation['source_cache_root']); mf = root/'manifest.json'
        if sha256(mf) != source['source_manifest_sha256'] or sha256(parent_path) != source['parent_manifest_sha256']:
            raise ValueError('Pure candidate geometry chain changed')
        manifest = json.loads(mf.read_text('utf8')); pcd = Path(parent['record']['input'])
        if sha256(pcd) != source['input_sha256'] or sha256(root/'xyz.npy') != manifest['files']['xyz.npy']['sha256']:
            raise ValueError('Raw source or straight candidate XYZ changed')
        raw = read_xyz_pcd(pcd)*1000.; geometry = parent['record']['geometry']
        planes, evidence = small_surface_planes(raw, geometry)
        xyz = np.load(root/'xyz.npy', mmap_mode='r', allow_pickle=False)
        extra = json.loads((root/'curve_metadata.json').read_text('utf8'))['sequences']
        sequences = {}; pools = {}; pool_stats = {}
        for i, entry in enumerate(manifest['sequences']):
            if entry['type'] != 'horizontal' or extra[i].get('curve_kind') == 'arc':
                continue
            sequence = dict(xyz=xyz[entry['start']:entry['stop']], direction=entry['direction'], type=entry['type'])
            sequences[i] = sequence
            pools[i], pool_stats[i] = small_face_positions(sequence, geometry, planes)
        npz = OUT/(name+'_pools.npz')
        np.savez_compressed(npz, **{str(i):pool for i, pool in pools.items()})
        mapped = {(r['parent_id'], r['segment']):r['sequence'] for r in mapping['children']}
        child_flags = defaultdict(list); rows = []
        for original in source['endpoints']:
            key = original['parent_id'], original['segment']; si = mapped[key]
            score = closest_event(sequences[si], pools[si], np.asarray(original['point_mm'], float))
            row = dict(original, small_face_error_mm=score['error_mm'], small_face_covered_3mm=score['covered_3mm'])
            if original['component_covered_3mm'] and not score['covered_3mm']:
                raise ValueError('Additional hypotheses removed an original event')
            rows.append(row); all_rows.append(row); child_flags[key].append(score['covered_3mm'])
        result = dict(piece=name, input_sha256=source['input_sha256'], source_manifest_sha256=sha256(mf),
            parent_manifest_sha256=sha256(parent_path), parent_coverage_sha256=sha256(source_path),
            planes=planes, plane_evidence=evidence, pool_stats=pool_stats, pool_sha256=sha256(npz),
            truth_children=len(child_flags), covered_children=sum(len(v) == 2 and all(v) for v in child_flags.values()),
            truth_endpoints=len(rows), covered_endpoints=sum(r['small_face_covered_3mm'] for r in rows),
            original_covered_children=source['covered_children'], original_covered_endpoints=source['covered_endpoints'],
            candidate_positions=sum(map(len, pools.values())), endpoints=rows,
            runtime_inputs_are_raw_PCD_and_point_derived_geometry=True, diagnostic_only=True, full_contract_pass=False)
        if sha256(pcd) != source['input_sha256']:
            raise ValueError('Raw input changed during small-face audit')
        dump(OUT/(name+'.json'), result); results[name] = result
        print('SMALL_FACE_EVENTS', name, result['covered_children'], '/', result['truth_children'],
              'ENDPOINTS', result['covered_endpoints'], '/', len(rows), 'PLANES', len(planes), flush=True)
        del raw, xyz, sequences, pools
    total = {key:sum(r[key] for r in results.values()) for key in
        ['truth_children', 'covered_children', 'truth_endpoints', 'covered_endpoints', 'candidate_positions',
         'original_covered_children', 'original_covered_endpoints']}
    if total['truth_children'] != 1221 or total['truth_endpoints'] != 2442:
        raise ValueError('Full fixed child denominator required')
    groups = {}
    for key, predicate in [('unmarked_workrange', lambda r:r['parent_workrange_any'] is False),
                           ('marked_workrange', lambda r:r['parent_workrange_any'] is True),
                           ('explicit_workrange_cut', lambda r:r['explicit_endpoint_workrange_split']),
                           ('internal', lambda r:r['role'] == 'internal')]:
        selected = [r for r in all_rows if predicate(r)]
        groups[key] = dict(total=len(selected), covered=sum(r['small_face_covered_3mm'] for r in selected))
    sufficient = total['covered_children']/total['truth_children'] > .95
    dump(OUT/'complete.json', dict(config_sha256=sha256(OUT/'config.json'), total=total, groups=groups,
        piece_receipts={n:sha256(OUT/(n+'.json')) for n in results},
        geometry_pool_coverage_sufficient=sufficient, runtime_output_not_limited_to_pool=True,
        new_model_trained=False, elapsed_seconds=time.perf_counter()-began,
        original_targets_unchanged=True, observed_or_new_held_read=False, full_contract_pass=False))
    print('SMALL_FACE_EVENTS_COMPLETE', total, groups, 'POOL_SUFFICIENT', sufficient, flush=True)


if __name__ == '__main__':
    main()
