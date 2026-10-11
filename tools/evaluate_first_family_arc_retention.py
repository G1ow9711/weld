"""Post-freeze geometry diagnostic; fixed horizontal role is NOT a type model."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read, write, digest
from weld_pipeline.contract_curve_geometry import CircularArc, sample_arc, CurveGeometryError
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
from weld_pipeline.contract_geometry_d3_409 import score_source

BASE = Path(__file__).resolve().parents[1] / 'server_rebuild/expansion_v1'


def main(model_name='curved_group_comparison_grouped_v1', output_name='arc_pointer_first_family_retention_v1'):
    key = 'new_ac2008a03607'
    folder = BASE / output_name
    receipt = read(folder / 'receipt.json')
    assert digest(folder / 'protocol.json') == receipt['protocol_sha256']
    protocol = read(folder / 'protocol.json')
    events = BASE / 'training_arc_vertex_events_v1' / key
    candidates = BASE / 'union_nearest_arc_candidates_v1' / key
    assert digest(events / 'receipt.json') == protocol['event_receipt_sha256']
    ep = read(events / 'protocol.json')
    assert ep['candidate_receipt_sha256'] == digest(candidates / 'receipt.json')
    emap = {r['face']: r for r in read(events / 'receipt.json')['faces']}
    cmap = {r['face']: r for r in read(candidates / 'receipt.json')['faces']}
    parents, invalid, identities, predictions = [], [], [], {}
    # Role fixed for this geometry diagnostic before reading any target labels.
    role = 'horizontal'
    for record in receipt['faces']:
        face = record['face']
        assert digest(folder / record['file']) == record['sha256']
        er, cr = emap[face], cmap[face]
        assert digest(events / er['file']) == er['sha256']
        assert digest(candidates / cr['file']) == cr['sha256']
        with np.load(events / er['file']) as z:
            offsets, positions = z['offsets'], z['positions_mm']
        arcs = read(candidates / cr['file'])['candidates']
        rows = read(folder / record['file'])['rows']
        assert len(rows) == len(arcs)
        for index, row in enumerate(rows):
            assert row['candidate'] == index
            path = row['selected_local_event_indices']
            predictions[(face, index)] = path
            a, b = offsets[index:index+2]
            q = positions[a:b]
            assert all(0 <= i < len(q) for i in path)
            assert all(q[j] > q[i] for i, j in zip(path, path[1:]))
            if not path:
                continue
            if len(path) == 1:
                invalid.append(dict(counted_false_children=1, predicted_weld_types=[0], predicted_kinds=[2], reason='ONLY_ONE_SELECTED_BOUNDARY'))
                continue
            arc = CircularArc.from_parameters(arcs[index]['arc_parameters'])
            segments = []
            try:
                for i, j in zip(path, path[1:]):
                    segment = sample_arc(arc.subarc(float(q[i]), float(q[j])))
                    segment['weld_type'] = role
                    reverse_segment(segment)
                    segments.append(segment)
            except CurveGeometryError as error:
                invalid.append(dict(counted_false_children=len(path)-1, predicted_weld_types=[0]*(len(path)-1), predicted_kinds=[2]*(len(path)-1), reason=str(error), face=face, candidate=index))
                continue
            parents.append(dict(child_count=len(segments), is_closed=False, gaps=['SHARED']*(len(segments)-1), segments=segments))
            identities.append(dict(face=face, candidate=index))
    assert read(BASE/model_name/'receipt.json')['model_sha256']==protocol['model_sha256']
    targets = BASE / 'labels' / key / 'targets.json'
    labels = read(BASE / 'training_arc_parent_boundaries_v1' / 'labels.json')
    assert digest(targets) == labels['target_sha256']
    truth = native_targets(read(targets))
    score = score_source(dict(parents=parents, invalid_predictions=invalid, total_prediction_children=sum(p['child_count'] for p in parents)+sum(r['counted_false_children'] for r in invalid)), truth, [False]*len(truth['parents']))
    write(folder / 'geometry_diagnostic.json', dict(score=score, invalid_predictions=invalid, predicted_parent_identities=identities, conditional_parity_not_applicable_different_family=True, prediction_receipt_sha256=digest(folder/'receipt.json'), target_sha256=digest(targets), fixed_role=role, scope='All frozen input carriers with pointer STOP; original D3 geometry and all duplicate/invalid FP retained. Fixed horizontal role is an explicit diagnostic assumption, not predicted semantics. First-family retention after second-family continuation; prior first-family training shared, not independent heldout or deployment.', fit_calls=0, DEV_reads=0, no_automatic_promotion=True))
    print(score['categories']['overall'], flush=True)


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--model-name',default='curved_group_comparison_grouped_v1');p.add_argument('--output-name',default='arc_pointer_first_family_retention_v1');a=p.parse_args();main(a.model_name,a.output_name)
