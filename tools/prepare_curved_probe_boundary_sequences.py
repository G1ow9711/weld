"""Conditional TRAIN boundary sequences, grouped by actual input carrier."""
from pathlib import Path
from collections import defaultdict
import numpy as np
from weld_pipeline.contract_curve_geometry import CircularArc
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    key='new_fcaf2af1abee';events=BASE/'curved_probe_vertex_events_v1'/key;source=BASE/'curved_upright_probe_v3'/key;capacity=BASE/'curved_probe_vertex_event_capacity_v1/report.json';cr=read(capacity);assert cr['event_receipt_sha256']==digest(events/'receipt.json');er=read(events/'receipt.json');assert digest(events/'events.npz')==er['events_sha256'];ep=read(events/'protocol.json');assert digest(source/'profiles.json')==ep['profiles_sha256'];profiles=read(source/'profiles.json');target=BASE/'labels'/key/'targets.json';truth=native_targets(read(target));assert digest(target)==read(source/'d3_interval_capacity.json')['target_sha256']
    with np.load(events/'events.npz') as z:offsets=z['offsets'];positions=z['positions_mm']
    groups=defaultdict(list)
    for row in cr['rows']:
        for w in row['witnesses']:groups[w['profile']].append(dict(w,parent=row['parent'],child=row['child']))
    rows=[]
    for profile,witnesses in sorted(groups.items()):
        a,b=offsets[profile:profile+2];q=positions[a:b];arc=CircularArc.from_parameters(profiles[profile]['representation']['arc_parameters']);points=arc.point_at(q);boundaries=defaultdict(list)
        for w in witnesses:
            child=truth['parents'][w['parent']]['children'][w['child']];ends=np.asarray(reference_segment(child)['points'])[[0,-1]]
            for event,end in zip(w['event_indices'],ends):boundaries[event].append(end)
        teacher=sorted(boundaries,key=lambda i:positions[i]);positive=[]
        for event in teacher:
            targets=np.asarray(boundaries[event]);distance=np.linalg.norm(points[:,None]-targets[None],axis=2).max(1);indices=np.flatnonzero(distance<=3);assert event-a in indices;positive.append((indices+a).tolist())
        rows.append(dict(profile=profile,boundary_event_indices=positive,teacher_event_indices=teacher,reference_segments=[dict(parent=w['parent'],child=w['child'],ordered_event_indices=w['event_indices']) for w in witnesses],unknown_inter_boundary_intervals=[i for i,(left,right) in enumerate(zip(teacher,teacher[1:])) if not any(set(w['event_indices'])=={left,right} for w in witnesses)]))
        print(profile,'known reference segments',len(witnesses),'unique boundaries',len(teacher),flush=True)
    out=BASE/'curved_probe_boundary_sequences_v1';out.mkdir(exist_ok=False);write(out/'labels.json',dict(rows=rows,event_receipt_sha256=digest(events/'receipt.json'),capacity_sha256=digest(capacity),target_sha256=digest(target),scope='Reference-selected TRAIN carriers. One input carrier has multiple independent parents; unique boundary events grouped once, original parent identities retained only in supervision. Unknown gaps are NOT negative weld/contact labels. Endpoint label sets alone do not guarantee every possible combination passes full D3.',DEV_reads=0));write(out/'receipt.json',dict(status='CONDITIONAL_BOUNDARY_SEQUENCES_READY',profiles=len(rows),labels_sha256=digest(out/'labels.json'),reference_parents=len({w['parent'] for v in groups.values() for w in v}),no_automatic_promotion=True))

if __name__=='__main__':main()
