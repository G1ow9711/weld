"""Freeze both learned relation/soft grouping arms before reading references."""
import argparse
import gc
import time
import numpy as np
import torch
from scipy.special import expit
from weld_pipeline.sjg_prepare_776 import ROOT,OUT,TRAIN,VALID,ARC
from weld_pipeline.sjg_geometry_776 import read,write,digest
from weld_pipeline.sjg_train_776 import utc,check_resources
from weld_pipeline.sjg_relation_features_880 import RelationHead,FEATURE_NAMES
from weld_pipeline.sjg_train_relation_882 import ARMS
from weld_pipeline.sjg_soft_material_883 import decode_soft,controls
from weld_pipeline.sjg_material_evaluate_862 import relation_counts
from weld_pipeline.sjg_context_compare_871 import overlap_detection


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args();dest=OUT/args.run;dest.mkdir(exist_ok=False);torch.set_num_threads(2)
    assert read(OUT/'run84/receipt.json')['status']=='TWO_PAIR_RELATION_HEADS_TRAINED'
    data=OUT/'run83';base=OUT/'run75/dense_control';inputs=read(data/'all_inputs_frozen.json')['sources']
    relations={r['source']:r for r in read(base/'all_predictions_frozen.json')['sources']}
    seeds={r['source']:r for r in read(base/'constrained_predictions_frozen.json')['sources']}
    write(dest/'registered_protocol.json',dict(stage='RELATION_HEAD_SOFT_GROUPING_COMPARISON',arms=ARMS,sources=TRAIN+VALID+ARC,
        all_arms_before_reference_reads=True,seed_single_membership_groups='run75/dense_control',
        soft_weight='clip(-different_logit,-5,5)',merge_rule='largest positive aggregate gain',overlap_attach='positive aggregate score to final group',
        no_fixed_target_entity_count=True,no_small_group_removal=True,no_development_selection=True,
        cached_XYZ_feature_replay_not_end_to_end_timing=True,source_sha256=digest(__file__),decoder_sha256=digest(ROOT/'weld_pipeline/sjg_soft_material_883.py'),
        full_contract_G=496,full_contract_pass=False,start_utc=utc()))
    write(dest/'controls.json',controls());manifests=[]
    for arm in ARMS:
        folder=dest/arm;folder.mkdir();training=OUT/'run84'/arm;checkpoint=training/'last.pt'
        assert digest(checkpoint)==read(training/'receipt.json')['checkpoint_sha256']
        model=RelationHead(np.zeros(len(FEATURE_NAMES)),np.ones(len(FEATURE_NAMES)),arm=='finite_geometry').cuda()
        with checkpoint.open('rb') as h:state=torch.load(h,map_location='cuda')
        model.load_state_dict(state['model']);del state;model.eval();frozen=[]
        with torch.inference_mode():
            for b in inputs:
                check_resources();source=b['source'];print('SOFT_RELATION_INFER',arm,source,flush=True)
                assert digest(data/b['features_file'])==b['features_sha256'] and digest(data/b['flat_file'])==b['flat_sha256']
                r=relations[source];s=seeds[source];assert digest(base/r['file'])==r['sha256'] and digest(base/s['file'])==s['sha256']
                x=np.load(data/b['features_file'],mmap_mode='r');flat=np.load(data/b['flat_file']);logit=np.empty(len(x),np.float32);started=time.perf_counter()
                for i in range(0,len(x),65536):logit[i:i+65536]=model(torch.from_numpy(np.array(x[i:i+65536])).cuda()).cpu().numpy()
                assert np.isfinite(logit).all()
                with np.load(base/r['file']) as p,np.load(base/s['file']) as q:
                    output,stats=decode_soft(p['neighbors'],flat,logit,q['offsets'],q['membership'])
                path=folder/f'source_{source:02d}_soft.npz';np.savez_compressed(path,**output,relation_logit=logit)
                rec=dict(source=source,points=b['points'],file=path.name,sha256=digest(path),raw_sha256=b['raw_sha256'],
                    features_sha256=b['features_sha256'],flat_sha256=b['flat_sha256'],relation_sha256=r['sha256'],seed_sha256=s['sha256'],
                    checkpoint_sha256=digest(checkpoint),stats=stats,head_and_decoder_seconds=time.perf_counter()-started)
                frozen.append(rec);print('SOFT_RELATION_FROZEN',arm,source,stats,flush=True);del x,flat,logit,output;gc.collect();torch.cuda.empty_cache()
        write(folder/'all_predictions_frozen.json',dict(sources=frozen,all_before_reference_reads=True))
        manifests.append(dict(arm=arm,sha256=digest(folder/'all_predictions_frozen.json')));del model;gc.collect();torch.cuda.empty_cache()
    write(dest/'all_arms_frozen_before_reference.json',dict(arms=manifests,all_before_reference_reads=True))
    teachers={r['source']:r for r in read(OUT/'run22/label_summary.json')};all_rows={};input_by_source={r['source']:r for r in inputs}
    for arm in ARMS:
        folder=dest/arm;rows=[]
        for b in read(folder/'all_predictions_frozen.json')['sources']:
            source=b['source'];assert digest(folder/b['file'])==b['sha256']
            if source not in teachers or 'teacher_file' not in teachers[source]:
                rows.append(dict(source=source,status='NO_OFFLINE_STRUCTURE_REFERENCE',original_points=b['points']));continue
            t=OUT/'run22'/teachers[source]['teacher_file'];assert digest(t)==teachers[source]['teacher_sha256']
            flat=np.load(data/input_by_source[source]['flat_file'])
            with np.load(t) as gt,np.load(folder/b['file']) as q,np.load(base/relations[source]['file']) as p,np.load(base/seeds[source]['file']) as seed:
                owner=gt['reference_volume'];near=p['neighbors'];other=near.clip(0)
                eligible=(near>=0)&(owner[:,None]>=0)&(owner[other]>=0);same=owner[:,None]==owner[other]
                probability=np.full(near.shape,np.nan,np.float32);probability.ravel()[flat]=expit(q['relation_logit'])
                relation=relation_counts(probability,same,eligible);entity=overlap_detection(q['offsets'],q['membership'],owner)
                baseline=overlap_detection(seed['offsets'],seed['membership'],owner)
            relation['predicted_different_precision']=relation['TN']/max(1,relation['TN']+relation['FN_observed'])
            row=dict(source=source,original_points=b['points'],relations=relation,soft_grouping=entity,seed_baseline=baseline,
                decoder_stats=b['stats'],full_contract_pass=False)
            rows.append(row);print('SOFT_RELATION_RESULT',arm,source,'DIFFERENT_PRECISION',relation['predicted_different_precision'],
                'RECALL',relation['different_recall'],'ENTITIES',entity['matched_reference_owners_at_IoU_half'],'/',entity['reference_owners'],flush=True)
        write(folder/'metrics.json',rows);all_rows[arm]=rows
    write(dest/'receipt.json',dict(status='PAIR_RELATION_SOFT_GROUPING_COMPARED',rows=all_rows,optimizer_updates=0,
        full_contract_G=496,weld_extraction_not_rerun=True,full_contract_pass=False,end_utc=utc()))


if __name__=='__main__':main()
