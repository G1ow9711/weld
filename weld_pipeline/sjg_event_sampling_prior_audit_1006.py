"""Audit training-query sampling versus the fixed input-grid distribution.

Only existing TRAIN supervision is used. No free prediction probabilities or
held-out references are read, and no thresholds or trained model are modified.
"""
import argparse
import numpy as np
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc
from weld_pipeline.sjg_curve_queries_907 import CurveCarrier
from weld_pipeline.sjg_local_curve_supervision_915 import event_targets


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sources',nargs='+',default=['21','10']);args=ap.parse_args()
    manifest=read(OUT/'run186/training_manifest.json');bindings={r['source_key']:r for r in manifest['cases']}
    assert set(args.sources)<=set(bindings)
    rows=[]
    for key in args.sources:
        binding=bindings[key];path=OUT/'run186'/binding['label_file'];assert digest(path)==binding['label_sha256']
        with np.load(path) as z:
            target=z['target'];mask=z['target_mask']
            sampled=(target[:,:4]*mask[:,:4]).sum(0)/mask[:,:4].sum(0)
            sampled_positive=((target[:,:4]>=.5)&mask[:,:4]).sum(0)/mask[:,:4].sum(0)
        folder=OUT/'run185'/('source_'+key);freeze=read(folder/'input_freeze.json')
        assert digest(folder/'profiles.json')==freeze['files']['profiles.json']
        profiles=read(folder/'profiles.json');supervision=read(OUT/'run186'/('source_'+key+'_supervision.json'))
        total=0;positive=np.zeros(4,np.int64);soft=np.zeros(4,np.float64);eligible_profiles=0
        for i,profile in enumerate(profiles):
            curve=CurveCarrier(profile['representation']);total+=max(1,int(np.ceil(curve.length/2.)))+1
            events=supervision['by_profile'].get(str(i),[])
            if not events:continue
            q=curve.grid(2.);labels,_,_=event_targets(profile,events,q);eligible_profiles+=1
            positive+=(labels>=.5).sum(0);soft+=labels.astype(np.float64).sum(0)
        natural=soft/total
        row=dict(source_key=key,total_input_grid_queries=total,training_queries=len(target),profiles=len(profiles),
            compatible_TRAIN_profiles=eligible_profiles,heads=['weld','start','end','closed'],
            sampled_soft_prior=sampled.tolist(),sampled_positive_fraction=sampled_positive.tolist(),
            full_grid_soft_prior=natural.tolist(),full_grid_positive_fraction=(positive/total).tolist(),
            soft_prior_ratio=[float(a/b) if b>0 else None for a,b in zip(sampled,natural)],
            weighted_sampled_constant_optimum=((np.array([1.,8.,8.,1.])*sampled)/(1-sampled+np.array([1.,8.,8.,1.])*sampled)).tolist(),
            sampling_diagnosis_not_validation=True,heldout_labels_read=0,prediction_probabilities_read=0)
        rows.append(row);print('SAMPLING_PRIOR',row,flush=True)
    write(OUT/'run190/train_query_sampling_prior_audit.json',dict(rows=rows,optimizer_updates=0,
        thresholds_and_model_unchanged=True,end_utc=utc()))


if __name__=='__main__':main()
