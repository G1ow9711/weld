"""Replay every early finite line region through local property recovery.

Outputs are separate unlabelled geometry hypotheses. Frozen learned predictions,
training sources and archived review files are not rewritten or relabelled.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

from weld_pipeline.server_supported_joint_regions import collect_regions, recover_region, POLICY
from weld_pipeline.server_plate_pair_evidence import POLICY as PLATE_POLICY
from weld_pipeline.sjg_local_runtime_904 import digest, read, write

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ['tools/recover_all_finite_joint_regions.py',
    'weld_pipeline/server_supported_joint_regions.py', 'weld_pipeline/server_plate_pair_evidence.py']


def iter_array(path):
    decoder = json.JSONDecoder()
    buffer = ''
    first = True
    with path.open(encoding='utf8') as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            buffer += chunk
            if first:
                buffer = buffer.lstrip()
                if not buffer.startswith('['):
                    raise ValueError('Candidate source must be a complete JSON array')
                buffer = buffer[1:]
                first = False
            while True:
                buffer = buffer.lstrip()
                if buffer.startswith(','):
                    buffer = buffer[1:].lstrip()
                if buffer.startswith(']'):
                    if buffer[1:].strip() or stream.read().strip():
                        raise ValueError('Unexpected data after the complete candidate array')
                    return
                try:
                    value, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if not chunk:
                        raise ValueError('Candidate source array is incomplete')
                    break
                yield value
                buffer = buffer[end:]


def recover_case(source_root, key, out, manifest, sources):
    started = time.perf_counter()
    folder = out / key
    folder.mkdir(exist_ok=False)
    source = source_root / key / 'run132' / ('source_' + key)
    paths = dict(xyz=source_root / key / 'run98/xyz_mm.npy',
                 faces=source / 'surfaces.json', profiles=source / 'profiles.json')
    hashes = {name: digest(path) for name, path in paths.items()}
    if manifest is not None and hashes['xyz'] != manifest[key]:
        raise ValueError('Dense input changed from its authenticated conversion evidence')
    write(folder / 'protocol.json', dict(sources=sources,
        input_files={name: str(path) for name, path in paths.items()}, input_sha256=hashes,
        unit='mm', reference_geometry_or_labels_read=False, raw_input_unchanged=True,
        geometry_and_plate_policy=POLICY, original_plate_evidence_policy=PLATE_POLICY,
        learned_model_modified_or_executed=False, geometric_candidates_used_as_training_labels=False))
    faces = read(paths['faces'])
    regions, coverage = collect_regions(iter_array(paths['profiles']), faces)
    write(folder / 'candidate_region_coverage.json', dict(coverage=coverage,
        every_source_profile_visited=True, no_first_pair_acceptance_gate=True,
        infinite_plane_aliases_used_as_plate_identity=False, profile_order_geometric_invariance_required=True))
    print('ALL_FINITE_SOURCE_REGIONS_COLLECTED', key, coverage, flush=True)
    xyz = np.load(paths['xyz'], mmap_mode='r', allow_pickle=False)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 64:
        raise ValueError('Complete dense XYZ input must have a supported shape')
    tree = cKDTree(xyz, copy_data=False)
    recovered = []
    audit = []
    try:
        for index, region in enumerate(regions):
            result = recover_region(xyz, tree, region)
            recovered.append(result)
            children = result['children']
            audit.append(dict(region_index=index, observed_face_regions=result['observed_face_regions'],
                source_profiles=len(region['source_profile_indices']),
                child_candidates=len(children), known_local_attribute_children=sum(
                    child['plate_attributes'] is not None for child in children),
                local_attribute_stations=result['local_attribute_stations'],
                local_property_regions=result['local_property_regions'],
                unresolved_reasons=dict(Counter(item['reason'] for item in result['unresolved_intervals'])),
                child_endpoints_mm=[child['endpoints_mm'] for child in children],
                known_plate_thicknesses_mm=[[p['thickness_mm'] for p in child['plate_attributes']]
                    for child in children if child['plate_attributes']]))
            write(folder / 'progress.json', dict(controller_pid=os.getpid(), status='REPLAYING_ALL_FINITE_SOURCE_REGIONS',
                completed_regions=index+1, registered_regions=len(regions), case=key,
                output_geometry_hypotheses=sum(len(r['children']) for r in recovered),
                learned_model_fit_calls=0, reference_geometry_reads=0, unix_time=time.time()))
            print('FINITE_REGION_REPLAY', key, index+1, '/', len(regions),
                  'CHILDREN', len(children), 'STATIONS', result['local_attribute_stations'], flush=True)
        if sources != {name: digest(ROOT / name) for name in sources}:
            raise ValueError('Bound region recovery implementation changed')
        if hashes != {name: digest(path) for name, path in paths.items()}:
            raise ValueError('Original frozen geometric input changed during replay')
        predictions = folder / 'geometric_contact_hypotheses.json'
        write(predictions, dict(geometric_traces=recovered, experimental=True,
            learned_final_predictions=False, confirmed_welds=False, geometry_fallback=True,
            reference_labels_used=False, STEP_used=False, full_contract_pass=False))
        write(folder / 'region_replay_audit.json', audit)
        children = [child for result in recovered for child in result['children']]
        directions = [np.asarray(child['endpoints_mm'])[1]-np.asarray(child['endpoints_mm'])[0] for child in children]
        axes = Counter('XYZ'[int(np.argmax(abs(direction)))] for direction in directions)
        summary = dict(status='ALL_SOURCE_FINITE_LINE_REGIONS_REPLAYED_UNLABELLED_HYPOTHESES_NOT_MODEL_QUALITY',
            case=key, input_points=len(xyz), input_sha256=hashes, source_profile_coverage=coverage,
            replayed_regions=len(regions), child_geometry_hypotheses=len(children),
            total_geometry_length_mm=sum(child['length_mm'] for child in children),
            known_local_plate_attribute_children=sum(child['plate_attributes'] is not None for child in children),
            unknown_attribute_children=sum(child['plate_attributes'] is None for child in children),
            principal_axis_hypothesis_counts=dict(axes),
            hypotheses_sha256=digest(predictions), replay_audit_sha256=digest(folder / 'region_replay_audit.json'),
            protocol_sha256=digest(folder / 'protocol.json'), input_coordinate_transform_applied=False,
            no_short_length_output_filter=True, unsupported_observation_gaps_filled=False,
            extraction_accuracy_percent=None, precision_percent=None, truth_error_mm=None,
            business_signed_frames_or_material_identity_certified=False,
            learned_fit_calls=0, reference_geometry_reads=0, full_contract_pass=False,
            seconds=time.perf_counter()-started)
        write(folder / 'receipt.json', summary)
        print('COMPLETE_ALL_FINITE_REGION_REPLAY', key, len(regions), len(children), flush=True)
        return dict(path=str((folder / 'receipt.json').relative_to(ROOT)), sha256=digest(folder / 'receipt.json'))
    finally:
        del tree
        xyz._mmap.close()


def main(args):
    if Path(args.experiment_name).name != args.experiment_name or args.experiment_name in ('', '.', '..'):
        raise ValueError('An independent output experiment is required')
    if args.declared_unit != 'mm':
        raise ValueError('Declared millimetre input is required; units are not guessed')
    if len(set(args.case)) != len(args.case) or any(Path(key).name != key or key in ('', '.', '..') for key in args.case):
        raise ValueError('Each case must name one distinct input folder')
    source_root = Path(args.source_root).resolve(strict=True)
    manifest = None
    if args.input_sha_manifest:
        rows = read(Path(args.input_sha_manifest).resolve(strict=True))
        manifest = {row['case']: row['xyz_sha256'] for row in rows}
        if not set(args.case) <= set(manifest):
            raise ValueError('Every replay input needs its original dense XYZ fingerprint')
    out = ROOT / 'server_rebuild' / args.experiment_name
    out.mkdir(exist_ok=False)
    sources = {name: digest(ROOT / name) for name in SOURCES}
    write(out / 'protocol.json', dict(sources=sources, cases=args.case,
        original_source_root=str(source_root), unit=args.declared_unit,
        no_case_specific_coordinate_constants=True, every_candidate_region_required=True,
        model_fit_calls=0, inference_labels_or_STEP_used=False))
    try:
        with threadpool_limits(limits=2):
            receipts = [recover_case(source_root, key, out, manifest, sources) for key in args.case]
        write(out / 'receipt.json', dict(status='ALL_REGISTERED_UNLABELLED_REGION_REPLAYS_COMPLETE_NOT_MODEL_ACCURACY',
            records=receipts, cases=args.case, sources=sources, model_fit_calls=0,
            reference_geometry_reads=0, model_optimization_round_completed=False,
            full_contract_pass=False))
    except BaseException:
        write(out / 'failure.json', dict(traceback=traceback.format_exc(), unix_time=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--case', required=True, action='append')
    parser.add_argument('--experiment-name', required=True)
    parser.add_argument('--declared-unit', required=True)
    parser.add_argument('--input-sha-manifest')
    main(parser.parse_args())
