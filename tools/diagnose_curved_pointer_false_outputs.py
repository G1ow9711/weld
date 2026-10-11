"""Attribute all frozen output errors without changing outputs or thresholds."""
from pathlib import Path
from collections import Counter
import numpy as np
from weld_pipeline.contract_curve_geometry import CircularArc,sample_arc,CurveGeometryError
from weld_pipeline.sjg_native_d3_adapter_1056 import native_targets
from weld_pipeline.world_set_evaluation_285 import reference_segment
from weld_pipeline.contract_geometry_d3_409 import child_error
from weld_pipeline.legacy_incremental_event_runtime_393 import reverse_segment
from weld_pipeline.sjg_local_runtime_904 import read,write,digest
BASE=Path(__file__).resolve().parents[1]/'server_rebuild/expansion_v1'


def main():
    key='new_fcaf2af1abee';folder=BASE/'curved_probe_pointer_all_candidates_v1';r=read(folder/'receipt.json');assert digest(folder/'predictions.json')==r['predictions_sha256'];old=read(folder/'geometry_diagnostic.json');events=BASE/'curved_probe_vertex_events_v1'/key;er=read(events/'receipt.json');assert digest(events/'events.npz')==er['events_sha256'];ep=read(events/'protocol.json');source=BASE/'curved_upright_probe_v3'/key;assert digest(source/'profiles.json')==ep['profiles_sha256'];profiles=read(source/'profiles.json');target=BASE/'labels'/key/'targets.json';assert digest(target)==old['target_sha256'];truth=native_targets(read(target));refs=[(pi,p['children'][0],reference_segment(p['children'][0])) for pi,p in enumerate(truth['parents']) if len(p['children'])==1 and p['children'][0]['original_kind']=='arc'];known={row['profile'] for row in read(BASE/'curved_probe_boundary_sequences_v1/labels.json')['rows']};assigned={m['predicted_parent_index'] for m in old['score']['matches']};rows=[];parent_index=0;counts=Counter();levels={};lengths={}
    with np.load(events/'events.npz') as z:offsets=z['offsets'];positions=z['positions_mm']
    for row in read(folder/'predictions.json')['rows']:
        profile=row['profile'];a,b=offsets[profile:profile+2];q=positions[a:b];path=row['selected_local_event_indices'];arc=CircularArc.from_parameters(profiles[profile]['representation']['arc_parameters']);level=profiles[profile]['base_level_mm'];levelkey=format(level,'.9g');levelcounts=levels.setdefault(levelkey,Counter())
        if len(path)==1:
            counts['one_boundary']+=1;levelcounts['one_boundary']+=1;rows.append(dict(profile=profile,kind='one_boundary',known_carrier=profile in known));continue
        for index,(i,j) in enumerate(zip(path,path[1:])):
            lo,hi=float(q[i]),float(q[j]);matches=[]
            try:segment=sample_arc(arc.subarc(lo,hi));segment['weld_type']='horizontal'
            except CurveGeometryError:
                counts['invalid_geometry']+=1;levelcounts['invalid_geometry']+=1;rows.append(dict(profile=profile,interval_index=index,kind='invalid_geometry',known_carrier=profile in known));continue
            ends=np.asarray(segment['points'])[[0,-1]]
            for pi,child,ref in refs:
                targetends=np.asarray(ref['points'])[[0,-1]]
                if min(np.linalg.norm(ends-targetends,axis=1).max(),np.linalg.norm(ends[::-1]-targetends,axis=1).max())>3:continue
                for reverse in [False,True]:
                    candidate=dict(segment,**reverse_segment(segment)) if reverse else segment
                    if child_error(candidate,child,ref,False,False) is not None:matches.append(pi);break
            kind='assigned_true_positive' if parent_index in assigned else ('duplicate_compatible_geometry' if matches else 'reference_unmatched_geometry')
            if parent_index in assigned:assert matches
            counts[kind]+=1;levelcounts[kind]+=1;lengths.setdefault(kind,[]).append(hi-lo);rows.append(dict(profile=profile,interval_index=index,predicted_parent_index=parent_index,kind=kind,compatible_reference_parents=matches,known_carrier=profile in known,length_mm=hi-lo,base_level_mm=level));parent_index+=1
    assert sum(counts.values())==old['score']['categories']['overall']['predicted']
    partition={}
    for known_flag in [False,True]:partition[str(known_flag)]=dict(Counter(row['kind'] for row in rows if row['known_carrier']==known_flag))
    out=BASE/'curved_pointer_false_output_diagnostic_v1';out.mkdir(exist_ok=False);write(out/'report.json',dict(rows=rows,counts=dict(counts),known_carrier_partition=partition,base_level_counts={k:dict(v) for k,v in levels.items()},length_quantiles_mm={k:np.quantile(v,[0,.5,.95,1]).tolist() for k,v in lengths.items()},prediction_receipt_sha256=digest(folder/'receipt.json'),score_sha256=digest(folder/'geometry_diagnostic.json'),scope='TRAIN output-quality error attribution. Reference-unmatched is NOT proof of no physical weld. No threshold tuning/output changes/negative contact labels.',DEV_reads=0));print(dict(counts));print(partition)

if __name__=='__main__':main()
