"""TRAIN diagnostic using actual frozen arc events on known carrier witnesses."""
from pathlib import Path
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.contract_curve_geometry import CircularArc,sample_arc
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment

BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def main(key='new_ac2008a03607'):
    events=BASE/'training_arc_vertex_events_v1'/key;er=read(events/'receipt.json');assert er['status']=='INPUT_ARC_EVENTS_FROZEN'
    assert digest(events/'protocol.json')==er['protocol_sha256'];ep=read(events/'protocol.json')
    candidates=BASE/'union_nearest_arc_candidates_v1'/key;cr=read(candidates/'receipt.json');assert ep['candidate_receipt_sha256']==digest(candidates/'receipt.json')
    for f in er['faces']:assert digest(events/f['file'])==f['sha256']
    for f in cr['faces']:assert digest(candidates/f['file'])==f['sha256']
    capacity=read(candidates/'d3_interval_capacity.json');assert capacity['candidate_receipt_sha256']==digest(candidates/'receipt.json')
    targetpath=BASE/'labels'/key/'targets.json';assert digest(targetpath)==capacity['target_sha256'];truth=native_targets(read(targetpath));banks={};rows=[]
    for row in capacity['rows']:
        child=truth['parents'][row['parent']]['children'][row['child']];ref=reference_segment(child);ends=np.asarray(ref['points'])[[0,-1]];witnesses=[];checked=0
        for witness in row['matches']:
            face=witness['face'];index=witness['candidate']
            if face not in banks:
                with np.load(events/f'face_{face}.npz') as z:banks[face]=(z['offsets'],z['positions_mm'],read(candidates/f'face_{face}.json')['candidates'])
            offsets,positions,records=banks[face];a,b=offsets[index:index+2];q=positions[a:b];arc=CircularArc.from_parameters(records[index]['arc_parameters']);points=arc.point_at(q)
            distances=np.linalg.norm(points[:,None]-ends[None],axis=2)
            first=np.flatnonzero(distances[:,0]<=3);second=np.flatnonzero(distances[:,1]<=3)
            pairs=sorted(((float(distances[i,0]+distances[j,1]),int(i),int(j)) for i in first for j in second if i!=j))
            found=None
            for _,i,j in pairs:
                lo,hi=sorted((float(q[i]),float(q[j])));segment=sample_arc(arc.subarc(lo,hi));segment['weld_type']=child['weld_type'];reverse=i>j
                if reverse:segment=dict(segment,**reverse_segment(segment))
                checked+=1;error=child_error(segment,child,ref,False,False)
                if error is not None:
                    found=dict(face=face,candidate=index,event_indices=[int(a+i),int(a+j)],interval_mm=[lo,hi],reverse=reverse,**error);break
            if found:witnesses.append(found)
        result=dict(parent=row['parent'],child=row['child'],carriers_checked=len(row['matches']),event_pairs_checked=checked,witnesses=witnesses);rows.append(result)
        print(row['parent'],row['child'],'event witnesses',len(witnesses),flush=True)
    out=BASE/('training_arc_event_capacity_v1' if key=='new_ac2008a03607' else 'training_arc_event_capacity_fc_v1');out.mkdir(exist_ok=False)
    write(out/'report.json',dict(rows=rows,children_with_input_event_D3_witness=sum(bool(r['witnesses']) for r in rows),reference_arc_children=len(rows),event_receipt_sha256=digest(events/'receipt.json'),carrier_capacity_sha256=digest(candidates/'d3_interval_capacity.json'),scope='Post-freeze TRAIN diagnostic on previously GT-projected compatible carrier witnesses only, not exhaustive negative proof. Endpoints restricted to actual input events, but labels choose intervals and weld type here; not runtime prediction or parent assembly.',fit_calls=0,DEV_reads=0))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--key',choices=['new_ac2008a03607','new_fcaf2af1abee'],default='new_ac2008a03607');args=parser.parse_args();main(args.key)
