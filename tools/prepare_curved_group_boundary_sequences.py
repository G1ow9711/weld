"""Revalidate TRAIN boundaries after input-only numerical grouping."""
from pathlib import Path
from collections import defaultdict
import numpy as np
from weld_pipeline.contract_curve_geometry import CircularArc,sample_arc
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    key='new_fcaf2af1abee';group=BASE/'curved_probe_event_groups_v1'/key;gr=read(group/'receipt.json');assert digest(group/'events.npz')==gr['files']['events.npz'];original=BASE/'curved_probe_boundary_sequences_v1';lr=read(original/'receipt.json');assert digest(original/'labels.json')==lr['labels_sha256'];labels=read(original/'labels.json');target=BASE/'labels'/key/'targets.json';assert digest(target)==labels['target_sha256'];truth=native_targets(read(target));source=BASE/'curved_upright_probe_v3'/key;profiles=read(source/'profiles.json');assert digest(source/'profiles.json')==read(source/'receipt.json')['profiles_sha256']
    with np.load(group/'events.npz') as z:offsets=z['offsets'];q=z['positions_mm'];members=z['member_offsets']
    rows=[];validated=0
    for row in labels['rows']:
        profile=row['profile'];a,b=offsets[profile:profile+2];arc=CircularArc.from_parameters(profiles[profile]['representation']['arc_parameters']);points=arc.point_at(q[a:b]);targets=defaultdict(list)
        for segment in row['reference_segments']:
            child=truth['parents'][segment['parent']]['children'][segment['child']];ref=reference_segment(child);ends=np.asarray(ref['points'])[[0,-1]];groupids=[]
            for oldid,end in zip(segment['ordered_event_indices'],ends):
                gid=int(np.searchsorted(members,oldid,side='right')-1);assert a<=gid<b;targets[gid].append(end);groupids.append(gid)
            lo,hi=sorted(q[groupids]);pred=sample_arc(arc.subarc(float(lo),float(hi)));pred['weld_type']=child['weld_type']
            if q[groupids[0]]>q[groupids[1]]:pred=dict(pred,**reverse_segment(pred))
            assert child_error(pred,child,ref,False,False) is not None;validated+=1
        teacher=sorted(targets);positive=[]
        for gid in teacher:
            distance=np.linalg.norm(points[:,None]-np.asarray(targets[gid])[None],axis=2).max(1);ids=np.flatnonzero(distance<=3)+a;assert gid in ids;positive.append(ids.tolist())
        rows.append(dict(profile=profile,teacher_event_indices=teacher,boundary_event_indices=positive))
    out=BASE/'curved_group_boundary_sequences_v1';out.mkdir(exist_ok=False);write(out/'labels.json',dict(rows=rows,group_receipt_sha256=digest(group/'receipt.json'),original_label_receipt_sha256=digest(original/'receipt.json'),target_sha256=digest(target),original_reference_segments_revalidated_D3=validated,scope='TRAIN group endpoint sets recomputed against original references. All30 teacher segments revalidated without output correction. No unknown gap/contact negatives.',DEV_reads=0));write(out/'receipt.json',dict(labels_sha256=digest(out/'labels.json'),profiles=len(rows),validated_segments=validated));print('PASS',validated,'segments',len(rows),'carriers')

if __name__=='__main__':main()
