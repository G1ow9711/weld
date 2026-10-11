"""Post-freeze TRAIN capacity on actual raw-point projected endpoint events."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.contract_curve_geometry import CircularArc,sample_arc
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    key='new_fcaf2af1abee';events=BASE/'curved_probe_vertex_events_v1'/key;er=read(events/'receipt.json');assert digest(events/'protocol.json')==er['protocol_sha256'];assert digest(events/'events.npz')==er['events_sha256'];source=BASE/'curved_upright_probe_v3'/key;ep=read(events/'protocol.json');assert digest(source/'profiles.json')==ep['profiles_sha256'];profiles=read(source/'profiles.json');capacity=read(source/'d3_interval_capacity.json');assert capacity['profiles_sha256']==ep['profiles_sha256'];target=BASE/'labels'/key/'targets.json';assert digest(target)==capacity['target_sha256'];truth=native_targets(read(target))
    with np.load(events/'events.npz') as z:offsets=z['offsets'];positions=z['positions_mm'];raw_ids=z['raw_point_ids'];heights=z['source_height_mm']
    rows=[]
    for row in capacity['rows']:
        child=truth['parents'][row['parent']]['children'][row['child']];ref=reference_segment(child);ends=np.asarray(ref['points'])[[0,-1]];witnesses=[];checked=0
        for witness in row['matches']:
            index=witness['profile'];a,b=offsets[index:index+2];q=positions[a:b];arc=CircularArc.from_parameters(profiles[index]['representation']['arc_parameters']);points=arc.point_at(q);distance=np.linalg.norm(points[:,None]-ends[None],axis=2);first=np.flatnonzero(distance[:,0]<=3);second=np.flatnonzero(distance[:,1]<=3)
            pairs=sorted((float(distance[i,0]+distance[j,1]),int(i),int(j)) for i in first for j in second if i!=j)
            for _,i,j in pairs:
                lo,hi=sorted((float(q[i]),float(q[j])));segment=sample_arc(arc.subarc(lo,hi));segment['weld_type']=child['weld_type'];reverse=i>j
                if reverse:segment=dict(segment,**reverse_segment(segment))
                checked+=1;error=child_error(segment,child,ref,False,False)
                if error is not None:
                    witnesses.append(dict(profile=index,event_indices=[int(a+i),int(a+j)],raw_point_ids=[int(raw_ids[a+i]),int(raw_ids[a+j])],source_height_mm=[float(heights[a+i]),float(heights[a+j])],interval_mm=[lo,hi],reverse=reverse,**error));break
        rows.append(dict(parent=row['parent'],child=row['child'],event_pairs_checked=checked,witnesses=witnesses));print(row['parent'],'witnesses',len(witnesses),flush=True)
    out=BASE/'curved_probe_vertex_event_capacity_v1';out.mkdir(exist_ok=False);write(out/'report.json',dict(rows=rows,children_with_input_event_D3_witness=sum(bool(r['witnesses']) for r in rows),reference_arc_children=len(rows),event_receipt_sha256=digest(events/'receipt.json'),carrier_capacity_sha256=digest(source/'d3_interval_capacity.json'),scope='TRAIN reference chooses candidate and frozen events; geometry capacity only, not runtime predicted endpoints, finite contact, continuity, or weld existence. Explicit section gaps remain unknown.',fit_calls=0,DEV_reads=0))

if __name__=='__main__':main()
