"""TRAIN21 cached-root diagnosis after frozen free inference failed."""
import argparse
import numpy as np
import torch
from weld_pipeline.sjg_local_runtime_904 import OUT,read,write,digest,utc
from weld_pipeline.sjg_balanced_hierarchy_1043 import load_model

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();dest=OUT/a.run;dest.mkdir(exist_ok=False)
    torch.set_num_threads(2);checkpoint=OUT/'run218/final.pt';model=load_model(torch.load(checkpoint,map_location='cpu'));model.eval()
    manifest=read(OUT/'run206/training_manifest.json');b=next(x for x in manifest['static_bindings'] if x['source_key']=='21');c=next(x for x in manifest['cases'] if x['source_key']=='21')
    assert digest(OUT/'run206'/b['file'])==b['sha256'] and digest(OUT/'run206'/c['file'])==c['sha256']
    with np.load(OUT/'run206'/b['file']) as z:s={k:torch.from_numpy(z[k]) for k in z.files}
    with np.load(OUT/'run206'/c['file']) as z:features=z['interval_x'];nf=z['node_family'];lo=z['node_lo'];hi=z['node_hi']
    selection=read(OUT/'run219/source_21/selection.json');domain=read(OUT/'run201/source_21_joint_domain.json')
    lookup={(int(f),float(l),float(h)):i for i,(f,l,h) in enumerate(zip(nf,lo,hi))}
    roots=[lookup[(f,float(l),float(h))] for f in selection['selected_families'] for l,h in domain['families'][f]['domains_mm']]
    with torch.inference_mode():
        r,_=model.relations(s['relation_x']);f,_=model.families(r,s['family_relation'],s['family_x'])
        outputs=model.intervals(f,torch.from_numpy(nf),torch.from_numpy(features));prob=outputs['state_logits'].softmax(-1).numpy();pred=prob.argmax(1)
    np.save(dest/'cached_node_state_probability.npy',prob)
    write(dest/'cached_prediction_freeze.json',dict(checkpoint_sha256=digest(checkpoint),probability_sha256=digest(dest/'cached_node_state_probability.npy'),
        source='21',cached_training_input_diagnostic_not_new_free_test=True,root_indices=roots))
    with np.load(OUT/'run206'/c['file']) as z:truth=z['state']
    confusion=np.zeros((4,4),np.int64)
    np.add.at(confusion,(truth,pred),1)
    root_truth=truth[roots];root_pred=pred[roots];positive=np.asarray(roots)[root_truth!=0]
    row=dict(status='ROOT_FAILURE_DIAGNOSED',source='21',selected_relations=len(selection['selected_relations']),selected_families=len(selection['selected_families']),
        root_count=len(roots),root_truth_counts=np.bincount(root_truth,minlength=4).tolist(),root_predicted_counts=np.bincount(root_pred,minlength=4).tolist(),
        positive_root_predicted_counts=np.bincount(pred[positive],minlength=4).tolist(),
        positive_root_probability_mean=prob[positive].mean(0).tolist() if len(positive) else None,
        full_teacher_forced_confusion=confusion.tolist(),state_order=['empty','full_single_child','partial_single_child','subdivide'],
        optimizer_updates=0,not_extraction_accuracy=True,full_contract_pass=False,end_utc=utc())
    write(dest/'receipt.json',row);print('ROOT_DIAGNOSIS',row,flush=True)

if __name__=='__main__':main()
