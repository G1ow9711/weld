"""All positive-length endpoint pairs; no first-start or top-K truncation."""
import argparse
import gc
import os
import traceback
import numpy as np
import torch
from scipy.signal import find_peaks
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_parent_inputs_829 import interval_inputs,training_targets
from weld_pipeline.sjg_third_surface_838 import adjacency_features

BASE=OUT/'run53/free'


def all_peak_pairs(q,probability,offset):
    spacing=float(q[1]-q[0]);distance=max(1,int(np.ceil(2/max(spacing,1e-6))))
    starts=find_peaks(np.r_[-1.,probability[:,1],-1.],height=.5,distance=distance)[0]-1
    ends=find_peaks(np.r_[-1.,probability[:,2],-1.],height=.5,distance=distance)[0]-1
    lo=np.clip(q[starts]+np.clip(2*offset[starts,0],-4,4),q[0],q[-1])
    hi=np.clip(q[ends]+np.clip(2*offset[ends,1],-4,4),q[0],q[-1])
    prefix=np.r_[0.,np.cumsum(probability[:,0],dtype=np.float64)];result=[]
    for i,s in enumerate(starts):
        valid=np.flatnonzero((ends>=s)&(hi>lo[i]+1e-6))
        means=(prefix[ends[valid]+1]-prefix[s])/(ends[valid]-s+1)
        for j,mean in zip(valid,means):
            if mean<.5:continue
            result.append(dict(lo=float(lo[i]),hi=float(hi[j]),start_peak_index=int(s),end_peak_index=int(ends[j]),
                foreground_confidence=float(mean),start_event_confidence=float(probability[s,1]),
                end_event_confidence=float(probability[ends[j],2])))
    return result


def main(base=BASE,create_labels=True,expected_candidates=12409):
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2);inventory=read(OUT/'inventory.json')
    write(dest/'registered_protocol.json',dict(stage='ALL_INTERNAL_INTERVAL_CANDIDATES_AND_TRAIN_LABELS',
        sources=TRAIN+VALID+ARC,train=TRAIN,original_G=496,expected_candidates_from_frozen_audit=expected_candidates,
        candidates_are_internal_not_published_weld_predictions=True,no_topK=True,field_run=str(base.relative_to(OUT)),
        create_TRAIN_labels=create_labels,
        event_threshold=.5,foreground_threshold=.5,node_features=132,endpoint_context_features=24,
        all_inputs_before_TRAIN_target_reads=True,development_and_arc_targets_not_read=True,
        source_sha256=digest(__file__),feature_source_sha256=digest(ROOT/'weld_pipeline/sjg_parent_inputs_829.py'),
        adjacency_source_sha256=digest(ROOT/'weld_pipeline/sjg_third_surface_838.py'),
        cached_frozen_fields_development_only=True,full_contract_pass=False,start_utc=utc()))
    try:
        candidate_freezes=[];input_freezes=[];rows=[]
        for source in TRAIN+VALID+ARC:
            check_resources();binding=next(r for r in read(base/'all_predictions_frozen.json')['sources'] if r['source']==source)
            assert digest(base/binding['file'])==binding['sha256']
            previous=read(base/binding['file'])['predictions'];probability_by_profile={c['profile']:c['candidate_probability'] for c in previous}
            geometry_folder=OUT/f'run34/source_{source:02d}';g=read(geometry_folder/'input_freeze.json')
            assert digest(geometry_folder/'profiles.json')==g['files']['profiles.json'];profiles=read(geometry_folder/'profiles.json')
            fields=read(base/f'source_{source:02d}_fields.json')['files'];children=[];references=[]
            for item in fields:
                file=base/item['file'];assert digest(file)==item['sha256']
                references.append(dict(item,file=os.path.relpath(file,dest)))
                with np.load(file) as z:q=z['q_mm'];p=z['probability'];offset=z['offset'];kind=z['type_probability']
                ni=item['profile'];profile=profiles[ni]
                for pair in all_peak_pairs(q,p,offset):
                    inside=(q>=pair['lo'])&(q<=pair['hi'])
                    if not inside.any():inside[np.abs(q-(pair['lo']+pair['hi'])/2).argmin()]=True
                    weld_type=['horizontal','vertical'][int(kind[inside].mean(0).argmax())]
                    points=np.asarray(profile['p0'])+np.asarray([pair['lo'],pair['hi']])[:,None]*profile['direction']
                    children.append(dict(profile=ni,points=points.tolist(),kind='line',weld_type=weld_type,
                        parent_hypothesis='internal_candidate',material_pair=None,attributes_status='UNKNOWN',
                        candidate_probability=probability_by_profile[ni],**pair))
            file=dest/f'source_{source:02d}_predictions.json';write(file,dict(predictions=children,invalid_intervals=[],internal_candidates_only=True))
            write(dest/f'source_{source:02d}_fields.json',dict(files=references,all_input_only=True))
            candidate_freezes.append(dict(source=source,file=file.name,sha256=digest(file),candidates=len(children)))
            print('ALL_INTERVAL_CANDIDATES_FROZEN',source,len(children),flush=True)
        total_candidates=sum(r['candidates'] for r in candidate_freezes)
        if expected_candidates is not None:assert total_candidates==expected_candidates
        write(dest/'all_predictions_frozen.json',dict(sources=candidate_freezes,internal_candidates_only=True,all_before_GT=True))
        for source in TRAIN+VALID+ARC:
            check_resources();children,profiles,nodes=interval_inputs(source,dest)
            context_path=base/f'source_{source:02d}_extra_context.json';events=read(context_path)['events']
            extra=np.asarray([adjacency_features([c['lo'],c['hi']],events[str(c['profile'])]).ravel() for c in children],np.float32).reshape(-1,24)
            path=dest/f'source_{source:02d}_inputs.npz';np.savez_compressed(path,nodes=nodes,features=np.c_[nodes,extra])
            input_freezes.append(dict(source=source,file=path.name,sha256=digest(path),candidates=len(children),
                context_sha256=digest(context_path),candidate_sha256=digest(dest/f'source_{source:02d}_predictions.json'),
                profiles_sha256=digest(OUT/f'run34/source_{source:02d}/profiles.json')))
            print('ALL_PAIR_FEATURES_FROZEN',source,len(children),'DIMENSION',156,flush=True)
            del children,profiles,nodes,extra;gc.collect()
        write(dest/'all_inputs_frozen.json',dict(sources=input_freezes,all_before_TRAIN_labels=True))
        for source in (TRAIN if create_labels else []):
            children=read(dest/f'source_{source:02d}_predictions.json')['predictions']
            target_path=ROOT/inventory[source]['target']['target_path'];assert digest(target_path)==inventory[source]['target']['target_sha256']
            node,_,matches=training_targets(children,np.empty((0,4),np.int64),read(target_path))
            path=dest/f'source_{source:02d}_labels.npy';np.save(path,node)
            row=dict(source=source,candidates=len(node),positive=int(node.sum()),label_sha256=digest(path),
                target_sha256=digest(target_path));rows.append(row);print('PAIR_TRAIN_LABELS',row,flush=True)
        write(dest/'receipt.json',dict(status='ALL_INTERVAL_INPUTS_AND_TRAIN_LABELS_COMPLETE' if create_labels else 'ALL_INTERVAL_INPUTS_FROZEN_NO_LABEL_READS',rows=rows,total_candidates=total_candidates,
            optimizer_updates=0,full_contract_pass=False,end_utc=utc()))
    except Exception:
        write(dest/'receipt.json',dict(status='FAILED',error=traceback.format_exc(),end_utc=utc()));raise


if __name__=='__main__':main()
