"""Post-freeze child geometry availability, explicitly ignoring parent assembly."""
from pathlib import Path
import argparse
import numpy as np
from scipy.spatial import cKDTree
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
from weld_pipeline.server_compact_line_predictions import line_segment_array
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment

BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'

def run(arm,key):
    folder=BASE/('sideplate_endpoint_'+arm+'_full_output_v1')/key
    freeze=read(folder/'prediction_freeze.json');path=folder/'compact_predictions.npz'
    assert digest(path)==freeze['files']['compact_predictions.npz']
    with np.load(path) as compact:
        endpoints=compact['endpoints_mm'];roles=compact['weld_type'];profiles=compact['profile_ids']
    valid=np.isfinite(endpoints).all(axis=(1,2));ids=np.flatnonzero(valid)
    tree=cKDTree(endpoints[ids].reshape(-1,3));targetpath=BASE/'labels'/key/'targets.json'
    truth=native_targets(read(targetpath));rows=[]
    for pi,parent in enumerate(truth['parents']):
        children=[];common=None
        for ci,child in enumerate(parent['children']):
            matches=[]
            if child['original_kind']=='line':
                ref=reference_segment(child);re=np.asarray(ref['points'])[[0,-1]]
                possible=np.unique(ids[np.asarray(tree.query_ball_point(re[0],3.),np.int64)//2])
                for index in possible:
                    ends=endpoints[index];direct=np.linalg.norm(ends-re,axis=1).max();back=np.linalg.norm(ends[::-1]-re,axis=1).max()
                    if min(direct,back)>3:continue
                    segment=line_segment_array(*ends);segment['weld_type']='horizontal' if roles[index]==0 else 'vertical'
                    for reverse in (False,True):
                        candidate=dict(segment,**reverse_segment(segment)) if reverse else segment
                        error=child_error(candidate,child,ref,False,False)
                        if error is not None:
                            matches.append(dict(compact_index=int(index),profile=int(profiles[index]),reverse=reverse));break
            compatible={m['profile'] for m in matches};common=compatible if common is None else common&compatible
            children.append(dict(child=ci,matches=matches))
        rows.append(dict(parent=pi,child_count=len(children),children=children,all_children_individually_available=all(c['matches'] for c in children),common_profiles=sorted(common or [])))
    multi=[r for r in rows if r['child_count']>1]
    report=dict(arm=arm,key=key,rows=rows,children_individually_available=sum(bool(c['matches']) for r in rows for c in r['children']),multi_child_parents=len(multi),multi_parents_all_children_available=sum(r['all_children_individually_available'] for r in multi),multi_parents_all_children_same_profile=sum(bool(r['common_profiles']) for r in multi),prediction_freeze_sha256=digest(folder/'prediction_freeze.json'),target_sha256=digest(targetpath),scope='Optimistic child-geometry availability only. Ignores unique assignment, exact shared events, whole-parent direction, gaps, and parent selection; not runtime TP. No prediction altered.',fit_calls=0,DEV_reads=0)
    write(folder/'child_geometry_capacity.json',report)
    print({k:v for k,v in report.items() if k!='rows'},flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['control8','expanded9'],required=True);parser.add_argument('--key',required=True);args=parser.parse_args();run(args.arm,args.key)
