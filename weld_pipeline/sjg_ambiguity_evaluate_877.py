"""Compare material ambiguity with face-boundary gating on identical frozen relations."""
import argparse
import gc
import time
import numpy as np
import torch
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_material_ambiguity_874 import MaterialAmbiguityHead,targets_from_reference
from weld_pipeline.sjg_ambiguity_prepare_875 import confusion
from weld_pipeline.sjg_constrained_material_870 import constrained_memberships
from weld_pipeline.sjg_context_compare_871 import overlap_detection


def scores(counts):
    c=dict(counts);tp,fp,fn,tn=(c[k] for k in ['TP','FP','FN','TN'])
    c.update(precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),F1=2*tp/max(1,2*tp+fp+fn),
        unique_owner_false_uncertainty_rate=fp/max(1,fp+tn))
    return c


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False)
    torch.set_num_threads(2);checkpoint=OUT/'run79/last.pt';training=read(OUT/'run79/receipt.json')
    assert training['status']=='SEPARATE_MATERIAL_AMBIGUITY_HEAD_TRAINED' and digest(checkpoint)==training['checkpoint_sha256']
    model=MaterialAmbiguityHead(np.zeros(32),np.ones(32)).cuda()
    with checkpoint.open('rb') as h:state=torch.load(h,map_location='cuda')
    model.load_state_dict(state['model']);del state;model.eval();base=OUT/'run75/dense_control';data=OUT/'run78'
    inputs={r['source']:r for r in read(data/'all_inputs_frozen.json')['sources']}
    relation={r['source']:r for r in read(base/'all_predictions_frozen.json')['sources']}
    previous={r['source']:r for r in read(base/'constrained_predictions_frozen.json')['sources']}
    write(dest/'registered_protocol.json',dict(stage='MATERIAL_AMBIGUITY_VS_FACE_BOUNDARY_GATING',sources=TRAIN+VALID+ARC,
        checkpoint_sha256=digest(checkpoint),frozen_backbone_sha256=read(data/'registered_protocol.json')['checkpoint_sha256'],
        fixed_material_relations='run75/dense_control',baseline_gates='surface edge probability',new_gates='material ambiguity probability',
        fixed_gate_threshold=.5,fixed_relation_threshold=.5,only_gate_changes=True,
        all8_outputs_frozen_before_reference=True,XYZ_derived_feature_replay_not_full_end_to_end_timing=True,
        source_sha256=digest(__file__),decoder_sha256=digest(ROOT/'weld_pipeline/sjg_constrained_material_870.py'),
        full_contract_G=496,full_contract_pass=False,start_utc=utc()))
    frozen=[]
    with torch.inference_mode():
        for source in TRAIN+VALID+ARC:
            check_resources();b=inputs[source];r=relation[source];old=previous[source]
            assert digest(data/b['file'])==b['sha256'] and digest(base/r['file'])==r['sha256'] and digest(base/old['file'])==old['sha256']
            assert b['raw_sha256']==r['raw_sha256'];x=np.load(data/b['file'],mmap_mode='r');prob=np.empty(len(x),np.float32)
            print('MATERIAL_AMBIGUITY_INFER',source,'ORIGINAL_POINTS',len(x),flush=True);started=time.perf_counter()
            for i in range(0,len(x),65536):prob[i:i+65536]=model(torch.from_numpy(np.array(x[i:i+65536])).cuda()).sigmoid().cpu().numpy()
            assert np.isfinite(prob).all() and np.all((prob>=0)&(prob<=1))
            with np.load(base/r['file']) as p:out,stats=constrained_memberships(p['neighbors'],p['material_distance'],prob)
            out['material_ambiguity_probability']=prob;path=dest/f'source_{source:02d}_memberships.npz';np.savez_compressed(path,**out)
            record=dict(source=source,points=len(x),file=path.name,sha256=digest(path),feature_sha256=b['sha256'],
                relation_sha256=r['sha256'],baseline_structure_sha256=old['sha256'],raw_sha256=r['raw_sha256'],
                gate_semantics='predicted material ambiguity, not face edge',stats=stats,head_and_decoder_seconds=time.perf_counter()-started)
            frozen.append(record);print('MATERIAL_AMBIGUITY_FROZEN',record,flush=True);del x,out,prob;gc.collect();torch.cuda.empty_cache()
    write(dest/'all_predictions_frozen.json',dict(sources=frozen,all_before_reference_reads=True))
    teachers={r['source']:r for r in read(OUT/'run22/label_summary.json')};rows=[]
    for record in frozen:
        source=record['source'];assert digest(dest/record['file'])==record['sha256']
        if source not in teachers or 'teacher_file' not in teachers[source]:
            rows.append(dict(source=source,status='NO_OFFLINE_STRUCTURE_REFERENCE',original_points=record['points']));continue
        teacher=OUT/'run22'/teachers[source]['teacher_file'];assert digest(teacher)==teachers[source]['teacher_sha256']
        with np.load(teacher) as t,np.load(dest/record['file']) as q,np.load(base/relation[source]['file']) as p,np.load(base/previous[source]['file']) as old:
            owner=t['reference_volume'];target=targets_from_reference(owner,t['multiple_material'])
            old_c=scores(confusion(p['boundary'],target));new_c=scores(confusion(q['material_ambiguity_probability'],target))
            old_entity=overlap_detection(old['offsets'],old['membership'],owner);new_entity=overlap_detection(q['offsets'],q['membership'],owner)
        row=dict(source=source,original_points=record['points'],face_boundary_as_ambiguity=old_c,material_ambiguity=new_c,
            old_face_gate_entities=old_entity,new_material_gate_entities=new_entity,decoder_stats=record['stats'],full_contract_pass=False)
        rows.append(row);write(dest/f'source_{source:02d}_metrics.json',row)
        print('MATERIAL_AMBIGUITY_RESULT',source,'F1',old_c['F1'],'->',new_c['F1'],'ENTITY',old_entity['matched_reference_owners_at_IoU_half'],
            '->',new_entity['matched_reference_owners_at_IoU_half'],'/',new_entity['reference_owners'],flush=True)
    write(dest/'receipt.json',dict(status='SEPARATE_AMBIGUITY_GATING_COMPARED',rows=rows,optimizer_updates=0,
        full_contract_G=496,weld_extraction_not_rerun=True,full_contract_pass=False,end_utc=utc()))


if __name__=='__main__':main()
