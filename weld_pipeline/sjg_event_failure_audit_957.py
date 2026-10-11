"""Frozen TRAIN postmortem: expose event calibration and oracle decode behavior."""
import argparse
from collections import Counter
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import OUT, read, write, digest, utc
from weld_pipeline.sjg_local_curve_supervision_915 import event_targets
from weld_pipeline.sjg_local_event_infer_931 import decode, geometry_nms
from weld_pipeline.sjg_local_event_score_932 import geometric_match
from weld_pipeline.sjg_local_event_train_930 import as_tensors, loss_for
from weld_pipeline.sjg_material_curve_model_953 import MaterialCurveEventModel


def summary(values):
    return np.quantile(values, [0, .25, .5, .75, 1]).tolist() if len(values) else None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2)
    write(dest/'registered_protocol.json',dict(stage='FROZEN_TRAIN_EVENT_POSTMORTEM',
        inputs=['run133','run140','run144/source_21'],source_sha256=digest(__file__),
        no_development_reads=True,no_training=True,no_output_modification=True,start_utc=utc()))
    manifest=read(OUT/'run133/training_manifest.json');graphs={r['source_key']:r for r in read(OUT/'run133/all_graphs_frozen_before_labels.json')['cases']}
    state=torch.load(OUT/'run140/final.pt',map_location='cpu');assert state['data_manifest_sha256']==digest(OUT/'run133/training_manifest.json')
    model=MaterialCurveEventModel();model.load_state_dict(state['state_dict']);model.eval();rows=[]
    with torch.inference_mode():
        for row in manifest['cases']:
            key=row['source_key'];g=graphs[key]
            for path,sha in [(g['file'],g['sha256']),(row['input_file'],row['input_sha256']),(row['label_file'],row['label_sha256'])]:assert digest(OUT/'run133'/path)==sha
            inputs=as_tensors(OUT/'run133'/row['input_file']);labels=as_tensors(OUT/'run133'/row['label_file']);graph=as_tensors(OUT/'run133'/g['file'])
            probability=[];losses=[]
            for begin in range(0,len(inputs['nodes']),256):
                batch={k:v[begin:begin+256] for k,v in inputs.items() if k!='q_mm'}
                lab={k:v[begin:begin+256] for k,v in labels.items()}
                logits,offset,_=model(dict(graph,**batch));probability.append(logits.sigmoid().numpy());losses.append(float(loss_for(logits,offset,lab)[0]))
            p=np.concatenate(probability);y=labels['target'].numpy();mask=labels['target_mask'].numpy();heads={}
            for i,name in enumerate(['weld','start','end','closed','forward']):
                known=mask[:,i];positive=y[:,i]>=.5;guess=p[:,i]>=.5
                heads[name]=dict(known=int(known.sum()),label_positive=int((known&positive).sum()),
                    predicted_positive=int((known&guess).sum()),TP=int((known&guess&positive).sum()),
                    FP=int((known&guess&~positive).sum()),FN=int((known&~guess&positive).sum()),
                    probabilities_positive=summary(p[known&positive,i]),probabilities_negative=summary(p[known&~positive,i]))
            rows.append(dict(source=key,queries=len(p),unique_profiles=int(len(torch.unique(inputs['nodes']))),
                mean_event_offset_loss=float(np.mean(losses)),heads=heads))
            print('TRAIN_EVENT_AUDIT',key,heads,flush=True)
    folder=OUT/'run144/source_21';freeze=read(folder/'input_output_freeze.json')
    for name,sha in freeze['files'].items():assert digest(folder/name)==sha
    profiles=read(OUT/'run132/source_21/profiles.json');sup=read(OUT/'run133/source_21_supervision.json')
    ptr=np.load(folder/'profile_offsets.npy');q=np.load(folder/'q_mm.npy',mmap_mode='r');p=np.load(folder/'probability.npy',mmap_mode='r');off=np.load(folder/'offset.npy',mmap_mode='r')
    details=[];oracle=[];hybrids={k:[] for k in ['true_endpoints','true_occupancy','true_offsets','true_direction']}
    # Only previously frozen TRAIN source. These are diagnostic oracle outputs, never deployable predictions.
    for index,profile in enumerate(profiles):
        a,b=map(int,ptr[index:index+2]);qq=q[a:b];events=sup['by_profile'].get(str(index),[])
        yy,oo,mm=event_targets(profile,events,qq);yy=np.c_[yy,np.ones(len(qq))]
        if events:
            assert len(events)==1;yy[:,4]=float(events[0]['target_direction_forward'])
            oracle.extend(dict(v,profile=index) for v in decode(profile['representation'],qq,yy,oo))
            interior=(yy[:,0]>=.5)&(yy[:,1]<1e-3)&(yy[:,2]<1e-3)
            details.append(dict(profile=index,grid_points=len(qq),true_interval=events[0]['intervals'],
                interior_queries=int(interior.sum()),interior_weld=summary(p[a:b][interior,0]),
                interior_start=summary(p[a:b][interior,1]),interior_end=summary(p[a:b][interior,2]),
                start_nearest_grid_probability=p[a+int(np.abs(qq-events[0]['lo']).argmin())].tolist(),
                end_nearest_grid_probability=p[a+int(np.abs(qq-events[0]['hi']).argmin())].tolist()))
        for mode in hybrids:
            pp=np.asarray(p[a:b,:5]).copy();offset=np.asarray(off[a:b]).copy()
            if mode=='true_endpoints':pp[:,1:3]=yy[:,1:3];offset=oo
            elif mode=='true_occupancy':pp[:,0]=yy[:,0]
            elif mode=='true_offsets':offset[mm]=oo[mm]
            elif mode=='true_direction':pp[:,4]=yy[:,4]
            hybrids[mode].extend(dict(v,profile=index) for v in decode(profile['representation'],qq,pp,offset))
    target=read(OUT/'run96/source_21_targets_local.json');oracle=geometry_nms(oracle)[0]
    comparisons={key:geometric_match(geometry_nms(value)[0],target) for key,value in hybrids.items()}
    result=dict(status='TRAIN_EVENT_FAILURE_AUDITED',training_rows=rows,positive_carrier_details=details,
        oracle_geometry=geometric_match(oracle,target),diagnostic_oracle_interventions=comparisons,
        original_predictions_untouched=True,not_model_accuracy=True,no_development_labels_used=True,
        full_contract_pass=False,end_utc=utc())
    write(dest/'receipt.json',result);print('ORACLE_DIAGNOSIS',result['oracle_geometry'],comparisons,flush=True)

if __name__=='__main__':main()
