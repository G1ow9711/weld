"""Input event provenance and finite interval controls, including real XYZ."""
import argparse
from copy import deepcopy
import math

import numpy as np

from tools.export_full_dataset_split_view import local_name
from tools.qualify_full_dataset_labels import ply_points
from tools.record_training_experiment import git_head
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.server_supported_interval_events import FiniteInputIntervalEvents, interval_child, shared_event_records
from weld_pipeline.server_supported_directed_parent_pool import directed_children, assemble_fixed_parent
from weld_pipeline.server_partitioned_full_dataset import PartitionedFullShardDataset
from weld_pipeline.sjg_local_runtime_904 import ROOT, digest, read, write


SOURCES = ['tools/check_supported_interval_events.py', 'weld_pipeline/server_supported_interval_events.py',
    'weld_pipeline/server_supported_directed_parent_pool.py', 'weld_pipeline/server_finite_carrier_grid.py',
    'weld_pipeline/sjg_curve_queries_907.py', 'weld_pipeline/contract_curve_geometry.py']


def main(args):
    base = ROOT / 'server_rebuild/full_dataset_v2'
    out = base / local_name(args.experiment_name); out.mkdir(exist_ok=False)
    sources = {name: digest(ROOT / name) for name in SOURCES}
    write(out / 'protocol.json', dict(source_commit=git_head(ROOT), sources=sources,
        scope='Synthetic line/arc evidence and real registered TRAIN input-only profiles',
        labels_or_model_weights_read=False, full_contract_pass=False))
    xyz = np.array([[0., 0., 0.], [3.3, 0., 0.], [10., 0., 0.]])
    profile = dict(representation=dict(kind='line', endpoints_mm=xyz[[0, 2]].tolist()),
                   support_intervals_mm=[[0., 3.3], [7., 10.]], raw_support_point_ids=[0, 1, 2])
    before = deepcopy(profile)
    bank = FiniteInputIntervalEvents(xyz, 'a' * 64).build(profile, [[0., 5.], [5., 10.]])
    assert before == profile
    q = bank['positions_mm']; cut = int(np.flatnonzero(q == 3.3)[0])
    assert np.array_equal(q[[0, -1]], [0., 10.]) and np.diff(q).max() <= 1.
    gap = int(np.flatnonzero(q == 5.)[0])
    assert bank['provenance'][gap, 5] and not bank['provenance'][gap, 11]
    assert not shared_event_records(bank, [gap])
    children = [interval_child(bank, 0, cut), interval_child(bank, cut, len(q) - 1)]
    events = shared_event_records(bank, [0, cut, len(q) - 1])
    parent = assemble_fixed_parent(directed_children(children), [0, 2], [1], events)
    assert parent['joint_boundary_max_displacement_mm'] == 0.
    arc = CircularArc([1., 2., 3.], 5., [0., 0., 1.], [1., 0., 0.], math.pi, False)
    points = arc.point_at(np.linspace(0., arc.length_mm, 17))
    ap = dict(representation=dict(kind='arc', arc_parameters=arc.to_parameters()),
              support_intervals_mm=[[0., arc.length_mm]], raw_support_point_ids=list(range(17)))
    ab = FiniteInputIntervalEvents(points, 'b' * 64).build(ap, [[0., arc.length_mm]])
    ac = interval_child(ab, 0, len(ab['positions_mm']) - 1)
    generated = CircularArc.from_parameters(ac['arc_parameters'])
    error = float(np.linalg.norm(generated.point_at(np.linspace(0., generated.length_mm, 65))
                                 - arc.point_at(np.linspace(0., arc.length_mm, 65)), axis=1).max())
    assert error <= 1e-10
    try:
        FiniteInputIntervalEvents(xyz, 'a' * 64, max_events=3).build(profile, [[0., 10.]])
    except ValueError:
        pass
    else:
        raise ValueError('Explicit event resource overflow was silently truncated')

    dataset = PartitionedFullShardDataset(ROOT, base)
    job_key = min(dataset.jobs, key=lambda k: (dataset.jobs[k]['original_points'], k))
    job = dataset.jobs[job_key]; raw = ROOT.parent / 'data/G_Data' / job['workpiece_source']
    if digest(raw) != job['workpiece_sha256']:
        raise ValueError('Actual complete raw input changed')
    xyz = ply_points(raw); inp = dataset.input_case(job_key)
    producer = FiniteInputIntervalEvents(xyz, digest(dataset.shards / job_key / 'receipt.json'))
    chosen = {}
    for pid, p in enumerate(inp.profiles):
        if inp.offsets[pid + 1] > inp.offsets[pid]:
            chosen.setdefault(p['representation']['kind'], pid)
    records = []
    for kind, pid in chosen.items():
        begin, end = inp.offsets[pid:pid + 2]
        b = producer.build(inp.profiles[pid], inp.spans[begin:end, :2])
        c = interval_child(b, 0, len(b['positions_mm']) - 1)
        directed_children([c])  # The actual new assembly geometry validator.
        records.append(dict(kind=kind, profile=pid, events=len(b['positions_mm']),
                            interval_length_mm=c['length_mm'], audit=b['audit']))
    if not records:
        raise ValueError('The actual registered input produced no checked profiles')
    if sources != {name: digest(ROOT / name) for name in sources}:
        raise ValueError('Input event source changed')
    write(out / 'receipt.json', dict(status='PASS_SHARED_FINITE_INPUT_EVENT_AND_INTERVAL_CONTROLS',
        protocol_sha256=digest(out / 'protocol.json'), job_key=job_key, raw_xyz_sha256=job['workpiece_sha256'],
        raw_points=len(xyz), actual_profiles=records, synthetic_arc_curve_max_error_mm=error,
        observed_support_gap_not_filled=True, grid_only_event_not_common_observed_boundary=True,
        original_observed_events_retained=True, finite_pair_constructed_without_end_regression=True,
        shared_new_parent_endpoint_displacement_mm=0., explicit_resource_overflow_rejected=True,
        label_geometry_reads=0, weights_read=0, full_TRAIN_preparation_claimed=False,
        full_contract_pass=False))
    print('PASS_FINITE_INPUT_INTERVAL_EVENTS', [(r['kind'], r['events']) for r in records], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', required=True)
    main(parser.parse_args())
