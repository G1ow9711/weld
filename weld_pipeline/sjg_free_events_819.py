"""Label-free interval decoding, followed by explicitly partial geometry scoring."""
import argparse
import gc
import time
import traceback
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import torch
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import check_resources,utc
from weld_pipeline.sjg_joint_data_816 import load_inputs,point_queries
from weld_pipeline.sjg_joint_model_817 import JointEventModel


def decode_intervals(q,probability,offset):
    on=probability[:,0]>=.5
    starts=np.flatnonzero(on&~np.r_[False,on[:-1]])
    ends=np.flatnonzero(on&~np.r_[on[1:],False])
    intervals=[];invalid=[]
    for first,last in zip(starts,ends):
        values=[]
        for head,index,outside in ((0,first,first-1),(1,last,last+1)):
            value=float(q[index])
            if 0<=outside<len(q):
                a,b=probability[index,0],probability[outside,0]
                if abs(float(a-b))>1e-12:
                    value=float(q[outside]+(.5-b)/(a-b)*(q[index]-q[outside]))
            near=np.flatnonzero(np.abs(q-value)<=4.)
            if len(near):
                peak=int(near[np.argmax(probability[near,head+1])])
                if probability[peak,head+1]>=.5:
                    value=float(np.clip(q[peak]+np.clip(offset[peak,head]*2,-4,4),q[0],q[-1]))
            values.append(value)
        lo,hi=values
        if hi<=lo+1e-6:
            invalid.append(dict(lo=lo,hi=hi,reason='NONPOSITIVE_EVENT_INTERVAL'));continue
        intervals.append(dict(lo=lo,hi=hi,foreground_confidence=float(probability[first:last+1,0].mean())))
    return intervals,invalid


def geometry_only_score(predictions,target):
    children=[(pi,ci,c) for pi,p in enumerate(target['parents']) for ci,c in enumerate(p['children'])]
    points=np.asarray([p['points'] for p in predictions],float).reshape(-1,2,3)
    costs=np.full((len(points),len(children)),1e8,np.float64)
    for j,(pi,ci,child) in enumerate(children):
        if child['original_kind']!='line':continue
        endpoints=np.asarray(child['world_representation']['endpoints_mm'])
        error=np.minimum(np.linalg.norm(points-endpoints,axis=-1).max(-1),
                         np.linalg.norm(points[:,::-1]-endpoints,axis=-1).max(-1))
        # Lines with corresponding endpoints within 3 mm have their complete
        # segments within 3 mm; this does not certify type/parent/side attributes.
        good=error<=3.;costs[good,j]=error[good]-1e5
    matches=[]
    if costs.size:
        rows,cols=linear_sum_assignment(costs)
        for i,j in zip(rows,cols):
            if costs[i,j]<0:
                pi,ci,_=children[j]
                matches.append(dict(prediction=int(i),parent=pi,child=ci,
                    maximum_endpoint_distance_mm=float(costs[i,j]+1e5)))
    return dict(scope='UNTYPED_CHILD_LINE_GEOMETRY_DIAGNOSTIC_ONLY',original_G=target['original_children'],
        predicted=len(predictions),matched_geometry=len(matches),unmatched_predictions=len(predictions)-len(matches),
        matches=matches,parent_order_type_side_E16_not_scored=True,
        curves_retained_as_failures=sum(c['original_kind']!='line' for _,_,c in children),
        contract_extraction_accuracy_not_available=True,full_contract_pass=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);inventory=read(OUT/'inventory.json')
    checkpoint=OUT/'run36/last.pt';model=JointEventModel().cuda()
    with checkpoint.open('rb') as h:state=torch.load(h,map_location='cuda')
    model.load_state_dict(state['model']);del state;model.eval()
    write(dest/'registered_protocol.json',dict(stage='FREE_INPUT_PROFILE_EVENT_DECODING',sources=TRAIN+VALID+ARC,
        original_G=496,checkpoint_sha256=digest(checkpoint),source_sha256=digest(__file__),
        query_grid_step_mm=2.,candidate_threshold=.5,foreground_threshold=.5,event_threshold=.5,
        event_offset_clip_mm=4.,all_predictions_before_GT=True,STEP_reads=0,
        caches_input_derived_training_experiment_only=True,
        deployment_must_recompute_PTV3_and_geometry_from_raw=True,
        predictions_untyped_unparented=True,full_contract_pass=False,start_utc=utc()))
    try:
        frozen=[]
        with torch.inference_mode():
            for si in TRAIN+VALID+ARC:
                check_resources();start=time.perf_counter();xyz,point,faces,profiles=load_inputs(si,inventory)
                input_file=OUT/f'run35/source_{si:02d}_inputs.npz'
                binding=next(r for r in read(OUT/'run35/all_inputs_frozen.json')['sources'] if r['source']==si)
                assert digest(input_file)==binding['sha256']
                with np.load(input_file) as z:graph={k:torch.as_tensor(z[k],device='cuda') for k in z.files}
                h=model.encode_faces(graph['face_x'],graph['neighbors'],graph['relation']);tree=cKDTree(xyz)
                zs=[];ps=[]
                for begin in range(0,len(profiles),4096):
                    joint=model.encode_joints(h,graph['pairs'][begin:begin+4096],graph['geometry'][begin:begin+4096])
                    zs.append(joint);ps.append(model.relevance(joint).squeeze(-1).sigmoid().cpu().numpy())
                joint=torch.cat(zs);relevance=np.concatenate(ps);selected=np.flatnonzero(relevance>=.5)
                predictions=[];invalid=[];queries=0
                for number,ni in enumerate(selected,1):
                    p=profiles[int(ni)];q=np.linspace(0,p['length'],int(np.ceil(p['length']/2))+1)
                    probabilities=[];offsets=[]
                    for begin in range(0,len(q),2048):
                        values=q[begin:begin+2048];nodes=np.full(len(values),ni,np.int64)
                        patch,query=point_queries(profiles,nodes,values,xyz,point,tree)
                        logits,offset=model.query_events(joint[ni:ni+1].expand(len(values),-1),
                            torch.as_tensor(patch,device='cuda'),torch.as_tensor(query,device='cuda'))
                        probabilities.append(logits.sigmoid().cpu().numpy());offsets.append(offset.cpu().numpy())
                    probability=np.concatenate(probabilities);offset=np.concatenate(offsets);queries+=len(q)
                    intervals,bad=decode_intervals(q,probability,offset)
                    for item in bad:invalid.append(dict(profile=int(ni),**item))
                    for interval in intervals:
                        ends=np.asarray(p['p0'])+np.asarray([interval['lo'],interval['hi']])[:,None]*p['direction']
                        predictions.append(dict(profile=int(ni),points=ends.tolist(),kind='line',
                            type=None,parent=None,material_pair=None,attributes_status='UNKNOWN',
                            candidate_probability=float(relevance[ni]),**interval))
                    if number==1 or number%25==0 or number==len(selected):
                        print('FREE_EVENTS',si,'PROFILE',number,'/',len(selected),'QUERIES',queries,'SEGMENTS',len(predictions),flush=True)
                        check_resources()
                path=dest/f'source_{si:02d}_predictions.json'
                write(path,dict(predictions=predictions,invalid_intervals=invalid,
                    no_deduplication_without_material_evidence=True,all_unobserved_gaps_still_unknown=True))
                record=dict(source=si,file=path.name,sha256=digest(path),selected_profiles=len(selected),
                    free_queries=queries,predictions=len(predictions),invalid_intervals=len(invalid),
                    cached_feature_geometry_decode_seconds=time.perf_counter()-start,
                    not_end_to_end_runtime=True)
                frozen.append(record);print('FREE_EVENTS_FROZEN',record,flush=True)
                del xyz,point,faces,profiles,graph,h,tree,joint,zs;gc.collect();torch.cuda.empty_cache()
        write(dest/'all_predictions_frozen.json',dict(sources=frozen,all_before_GT_read=True))
        rows=[]
        for record in frozen:
            si=record['source'];target=ROOT/inventory[si]['target']['target_path']
            assert digest(target)==inventory[si]['target']['target_sha256']
            prediction=read(dest/record['file'])
            score=geometry_only_score(prediction['predictions'],read(target))
            score['invalid_events_counted_separately']=len(prediction['invalid_intervals'])
            write(dest/f'source_{si:02d}_geometry_only.json',score)
            row={k:v for k,v in score.items() if k!='matches'};row['source']=si;rows.append(row)
            print('FREE_GEOMETRY_ONLY',si,score['matched_geometry'],'/',score['original_G'],'FP',score['unmatched_predictions'],flush=True)
        write(dest/'receipt.json',dict(status='FREE_UNTYPED_EVENT_OUTPUT_EVALUATED',end_utc=utc(),rows=rows,
            all_original_G=496,contract_extraction_accuracy_not_available=True,full_contract_pass=False))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',end_utc=utc(),error=traceback.format_exc()));raise


if __name__=='__main__':main()
